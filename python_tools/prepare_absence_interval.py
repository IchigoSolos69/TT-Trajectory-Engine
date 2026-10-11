"""Prepare complete source-bound absence candidates; only explicit review is truth.

No model, optimizer or dataset split assignment is performed. OpenCV decodes the
whole original sequentially to reconcile source count and PTS; every selected
frame is retained at original resolution. Preparation never creates negatives.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading

DISK_RESERVE_BYTES=2*1024**3


class IntervalCanceled(RuntimeError):pass


class OwnerLifetime:
    """Hold a Windows process handle, so a reused PID cannot keep work alive."""
    def __init__(self,owner_pid=None,*,force_exit=False,kernel=None):
        self.event=threading.Event();self.finished=False;self.lock=threading.Lock()
        self.output=None;self.force_exit=force_exit;self.handle=None;self.kernel=kernel
        if owner_pid is not None:
            if type(owner_pid) is not int or owner_pid<=0:raise ValueError("Positive owner PID required")
            if kernel is None:
                if os.name!="nt":raise ValueError("Owner process handle guard requires Windows")
                import ctypes
                from ctypes import wintypes
                self.kernel=ctypes.WinDLL("kernel32",use_last_error=True)
                self.kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
                self.kernel.OpenProcess.restype=wintypes.HANDLE
                self.kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
                self.kernel.WaitForSingleObject.restype=wintypes.DWORD
                self.kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
            self.handle=self.kernel.OpenProcess(0x00100000,False,owner_pid)  # SYNCHRONIZE only
            if not self.handle:raise IntervalCanceled("OWNER_PROCESS_UNAVAILABLE")
            self.thread=threading.Thread(target=self._monitor,daemon=True);self.thread.start()

    def _monitor(self):
        while not self.finished:
            state=self.kernel.WaitForSingleObject(self.handle,200)
            if state==0x102:continue  # WAIT_TIMEOUT
            try:
                with self.lock:
                    if self.finished:return
                    self.event.set()
                    if self.output is not None:
                        # Only this exclusive extraction's incomplete publication is removed.
                        (self.output/"manifest.json").unlink(missing_ok=True)
                        write_json(self.output/"progress.json",dict(status="canceled",error="OWNER_PROCESS_EXITED",contains_ground_truth=False))
            finally:
                # Disk/write failure must not keep an orphan decoder alive.
                if self.event.is_set() and self.force_exit:os._exit(2)
            return

    def attach(self,output):
        with self.lock:self.output=Path(output)
        self.check()

    def check(self):
        if self.event.is_set():raise IntervalCanceled("OWNER_PROCESS_EXITED")
        if self.output is not None and (self.output/"cancel.json").is_file():
            marker=json.loads((self.output/"cancel.json").read_text(encoding="utf-8"))
            if marker.get("schema_version")==1 and marker.get("cancel_requested") is True:
                raise IntervalCanceled("EXPLICIT_CANCEL")

    def publish(self,manifest,progress):
        with self.lock:
            self.check()
            write_json(self.output/"manifest.json",manifest)
            write_json(self.output/"progress.json",progress)
            self.finished=True

    def progress(self,value):
        with self.lock:
            self.check();write_json(self.output/"progress.json",value)

    def close(self):
        self.finished=True
        if self.handle:
            self.thread.join(timeout=1)
            self.kernel.CloseHandle(self.handle);self.handle=None


def sha(path,check=lambda:None):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):check();h.update(chunk)
    return h.hexdigest()


def json_bytes(value):return (json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+"\n").encode()


def write_json(path,value):
    target=Path(path);pending=target.with_name(target.name+f".pending.{os.getpid()}.{threading.get_ident()}")
    pending.write_bytes(json_bytes(value));pending.replace(target)


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


def checked_child(folder,name):
    folder=Path(folder).resolve();path=folder/name;resolved=path.resolve()
    if resolved.parent!=folder or resolved.name!=name:raise ValueError("Capture file alias/junction escapes its validated source folder")
    return path


def _watch(dependencies,path,expected=None,check=lambda:None):
    path=Path(path).resolve();before=path.stat();digest=sha(path,check);after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError("Source changed while hashing")
    if expected is not None and digest!=expected:raise ValueError(f"Source bytes differ from immutable SHA: {path}")
    dependencies[path.as_posix()]=digest;return digest


def request_for(registry,clip,start,end,interval_id):
    if not isinstance(clip,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",clip):raise ValueError("Safe existing capture ID required")
    if not isinstance(interval_id,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}",interval_id):raise ValueError("Safe interval ID required")
    if type(start) is not int or type(end) is not int or not 1<=start<=end or end-start+1>50000:raise ValueError("Select 1–50,000 consecutive source frames")
    assignment=registry.get("captures",{}).get(clip)
    if not assignment:raise ValueError("Assign the original capture's explicit session and device first")
    session=registry.get("sessions",{}).get(assignment.get("session_id"),{})
    if session.get("role")=="test" or session.get("sealed") is True:raise ValueError("Sealed C is excluded from development decoding")
    if (session.get("role") not in ("train","val") or assignment.get("domain") not in ("iphone","pc_camera")
            or session.get("stage") not in (("existing","A") if session.get("role")=="train" else ("B",))):raise ValueError("Explicit compatible train/B-validation metadata required")
    return dict(schema_version=1,interval_id=interval_id,clip=clip,session_id=assignment["session_id"],
        domain=assignment["domain"],role=session["role"],stage=session["stage"],start_source_frame=start,end_source_frame=end)


def reviewed_source_conflicts(captures,source_sha,start,end,remember,registry=None):
    """Past explicit visible/hidden/unsure evidence cannot become bulk absence."""
    conflicts=[]
    if registry is None:
        registry=json.loads((Path(captures)/"training_sessions.json").read_text(encoding="utf-8"))
    base=Path(captures).resolve()
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir():continue
        folder=candidate.resolve()
        if folder.parent!=base or folder.name!=candidate.name:raise ValueError("Capture directory alias/junction escapes its validated source root")
        assignment=registry.get("captures",{}).get(candidate.name)
        if assignment and registry.get("sessions",{}).get(assignment.get("session_id"),{}).get("role")=="test":continue
        capture_file=checked_child(folder,"capture.json")
        if not capture_file.is_file():continue
        # These manifests are provenance metadata, never test annotation content.
        capture=json.loads(capture_file.read_text(encoding="utf-8"))
        if capture.get("original_sha256")!=source_sha:continue
        remember(capture_file)
        for name in ("annotations.json","frames.jsonl"):remember(checked_child(folder,name))
        labels=json.loads(checked_child(folder,"annotations.json").read_text(encoding="utf-8"))
        for text in checked_child(folder,"frames.jsonl").read_text(encoding="utf-8").splitlines():
            row=json.loads(text);fid=row.get("source_frame")
            if type(fid) is not int or not start<=fid<=end:continue
            annotation=labels.get(row.get("file"),{})
            flags=annotation.get("reviewed_classes")
            reviewed=(flags.get("ball") is True if isinstance(flags,dict) else annotation.get("reviewed") is True)
            if not reviewed:continue
            visibility=(annotation.get("geometry") or {}).get("visibility","unknown")
            if annotation.get("ball") or visibility!="absent" or (annotation.get("geometry") or {}).get("blur_streak"):
                conflicts.append(dict(clip=folder.name,source_frame=fid,visibility=visibility,reason="EXPLICIT_NON_ABSENT_BALL_REVIEW"))
    return conflicts


def require_source_binding(captures,registry,request,source_sha,remember):
    """A development alias cannot decode a source assigned to another role/device."""
    wanted=(request["session_id"],request["role"],request["domain"])
    for clip,assignment in sorted(registry.get("captures",{}).items()):
        session=registry.get("sessions",{}).get(assignment.get("session_id"),{})
        if session.get("role") not in ("train","val","test"):continue
        declared=assignment.get("source_sha256")
        if session.get("role")=="test":actual=declared  # C identity metadata only.
        else:
            if not isinstance(clip,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",clip):raise ValueError("Safe registered capture ID required")
            folder=(Path(captures)/clip).resolve()
            if folder.parent!=Path(captures).resolve() or folder.name!=clip:raise ValueError("Capture directory alias escapes source root")
            path=checked_child(folder,"capture.json")
            remember(path);actual=json.loads(path.read_text(encoding="utf-8")).get("original_sha256")
        if declared is not None and declared!=actual and source_sha in (declared,actual):
            raise ValueError("Original source identity conflicts with fixed registry metadata")
        if source_sha==actual and (assignment.get("session_id"),session.get("role"),assignment.get("domain"))!=wanted:
            raise ValueError("Original source is assigned to another session/role/device; development aliases are excluded")


def prepare_interval(captures,clip,start,end,interval_id,*,guard=None):
    from recording_source import resolve_original_path
    import cv2
    guard=guard or OwnerLifetime()
    captures=Path(captures).resolve();dependencies={}
    remember=lambda path,expected=None:_watch(dependencies,path,expected,guard.check)
    remember(Path(__file__))
    remember(Path(__file__).with_name("recording_source.py"))
    registry_path=captures/"training_sessions.json";remember(registry_path)
    registry=json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version")!=1:raise ValueError("Versioned training-session registry required")
    request=request_for(registry,clip,start,end,interval_id)  # C rejected before capture/index/image reads
    folder=(captures/clip).resolve()
    if folder.parent!=captures or folder.name!=clip:raise ValueError("Capture escapes root")
    capture_file=checked_child(folder,"capture.json")
    capture_hash=remember(capture_file)
    capture=json.loads(capture_file.read_text(encoding="utf-8"))
    if capture.get("source")!="recorded_camera" or capture.get("session_id")!=request["session_id"]:raise ValueError("Complete original recorded-camera provenance required")
    source_sha=capture.get("original_sha256")
    if not isinstance(source_sha,str) or not re.fullmatch(r"[0-9a-f]{64}",source_sha):raise ValueError("Immutable original SHA required")
    require_source_binding(captures,registry,request,source_sha,remember)
    source=resolve_original_path(capture,captures,remember)
    conflicts=reviewed_source_conflicts(captures,source_sha,start,end,remember,registry)
    if conflicts:raise ValueError("Requested absence interval conflicts with explicit prior ball review: "+json.dumps(conflicts))
    output=captures/"absence_intervals"/interval_id;output.parent.mkdir(exist_ok=True)
    if output.parent.resolve().parent!=captures or output.resolve().parent!=output.parent.resolve():
        raise ValueError("Interval output escapes original capture root")
    if output.exists():
        existing=json.loads((output/"request.json").read_text(encoding="utf-8"))
        if existing!=request or any(p.name not in ("request.json","job.json","progress.json","cancel.json","decode.log") for p in output.iterdir()):raise FileExistsError("Choose a new interval ID; existing extraction/review is never overwritten")
    else:
        output.mkdir();(output/"request.json").write_bytes(json_bytes(request))
    remember(output/"request.json")
    guard.attach(output)
    progress=output/"progress.json"
    video=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG)
    try:
        if not video.isOpened():raise RuntimeError("Original source cannot be decoded")
        reported=video.get(cv2.CAP_PROP_FRAME_COUNT)
        if not _finite(reported) or reported<end or abs(reported-round(reported))>1e-6:raise RuntimeError("Exact reported source frame accounting unavailable")
        reported=int(reported);source_rows=[];interval_rows=[];size=None
        guard.progress(dict(status="decoding",decoded_frames=0,source_reported_frames=reported,interval_frames=0))
        while True:
            guard.check()
            ok,image=video.read()
            guard.check()
            if not ok:break
            fid=len(source_rows)+1;pts=video.get(cv2.CAP_PROP_POS_MSEC)/1000
            decoder_index=video.get(cv2.CAP_PROP_POS_FRAMES)
            if not _finite(pts) or pts<0 or (source_rows and pts<=source_rows[-1]["source_timestamp_s"]):raise RuntimeError("Missing/nonmonotonic original PTS; no FPS-derived timestamps permitted")
            if source_rows and pts-source_rows[-1]["source_timestamp_s"]>.100000001:raise RuntimeError("Original PTS gap exceeds 100 ms; continuous absence coverage requires a separate source audit")
            if not _finite(decoder_index) or abs(decoder_index-fid)>1e-6:raise RuntimeError("Decoder source frame accounting gap")
            height,width=image.shape[:2]
            if size is None:size=[width,height]
            if size!=[width,height]:raise RuntimeError("Original frame dimensions change inside source")
            source_rows.append(dict(source_frame=fid,source_timestamp_s=pts,frame_duration_s=0.))
            if fid>1:source_rows[-2]["frame_duration_s"]=pts-source_rows[-2]["source_timestamp_s"]
            if start<=fid<=end:
                filename=f"frame_{fid}.jpg";target=output/filename
                if shutil.disk_usage(output).free-image.nbytes < DISK_RESERVE_BYTES:
                    raise OSError("DISK_BUDGET: retain 2 GiB free reserve; incomplete interval is not reviewable")
                if not cv2.imwrite(str(target),image,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError("Full interval image write failed")
                interval_rows.append(dict(file=filename,source_frame=fid,source_timestamp_s=pts,frame_duration_s=0.,
                    image_sha256=sha(target),decoded_index=fid,width=width,height=height))
            if fid%120==0:guard.progress(dict(status="decoding",decoded_frames=fid,source_reported_frames=reported,interval_frames=len(interval_rows)))
        if len(source_rows)!=reported or len(interval_rows)!=end-start+1:raise RuntimeError("Source reported/decoded/selected frame accounting differs; incomplete extraction is not reviewable")
        for row in interval_rows:row["frame_duration_s"]=source_rows[row["source_frame"]-1]["frame_duration_s"]
        source_start=source_rows[start-1]["source_timestamp_s"]
        source_end=source_rows[end]["source_timestamp_s"] if end<len(source_rows) else source_rows[end-1]["source_timestamp_s"]
        source_index=output/"source_frames.jsonl";index=output/"frames.jsonl"
        guard.check()
        if shutil.disk_usage(output).free-(len(source_rows)*300+len(interval_rows)*500) < DISK_RESERVE_BYTES:
            raise OSError("DISK_BUDGET: insufficient reserve for complete source indices")
        source_index.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in source_rows),encoding="utf-8")
        index.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in interval_rows),encoding="utf-8")
        # Recheck originals, registry and prior review evidence before publication.
        for path,digest in list(dependencies.items()):remember(path,digest)
        manifest={**request,"status":"prepared","source_sha256":source_sha,
            "original_path":capture["original_path"],"resolved_source_path":source.as_posix(),
            "source_capture_sha256":capture_hash,"source_decoded_frames":len(source_rows),"source_reported_frames":reported,
            "interval_frames":len(interval_rows),"source_start_s":source_start,"source_end_s":source_end,
            "duration_s":source_end-source_start,"coverage_complete":True,"timestamp_basis":"opencv_ffmpeg_source_pts_ms",
            "index_sha256":sha(index),"source_index_sha256":sha(source_index),"source_size":size,
            "created_at":datetime.now(timezone.utc).isoformat(),"contains_ground_truth":False,
            "decoder":dict(name="opencv_ffmpeg",opencv_version=cv2.__version__,sequential=True,random_seek=False),
            "settings":dict(jpeg_quality=95,full_resolution=True,source_pts=True,terminal_duration_policy="ZERO_WITHOUT_NEXT_PTS"),
            "preparation_dependencies_sha256":dependencies}
        guard.publish(manifest,dict(status="prepared",decoded_frames=len(source_rows),source_reported_frames=reported,
            interval_frames=len(interval_rows),contains_ground_truth=False))
        return manifest
    except BaseException as error:
        (output/"manifest.json").unlink(missing_ok=True)
        write_json(progress,dict(status="canceled" if isinstance(error,IntervalCanceled) else "failed",error=str(error),contains_ground_truth=False));raise
    finally:video.release()


def probe():
    try:
        import cv2
        return dict(available=bool(cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG)),decoder="opencv_ffmpeg",opencv_version=cv2.__version__,gpu_required=False)
    except ImportError as error:return dict(available=False,decoder="opencv_ffmpeg",reason=str(error),gpu_required=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--probe",action="store_true")
    parser.add_argument("--captures",type=Path);parser.add_argument("--clip")
    parser.add_argument("--start-source-frame",type=int);parser.add_argument("--end-source-frame",type=int)
    parser.add_argument("--interval-id");parser.add_argument("--owner-pid",type=int);args=parser.parse_args()
    if args.probe:print(json.dumps(probe()));return
    if any(v is None for v in (args.captures,args.clip,args.start_source_frame,args.end_source_frame,args.interval_id)):parser.error("Explicit captures/clip/source bounds/interval ID required")
    guard=None
    try:
        guard=OwnerLifetime(args.owner_pid,force_exit=True)
        result=prepare_interval(args.captures,args.clip,args.start_source_frame,args.end_source_frame,args.interval_id,guard=guard)
        print(json.dumps(dict(interval_id=result["interval_id"],status=result["status"],contains_ground_truth=False)))
    except IntervalCanceled as error:
        # An API-created request may exist even when the owner died before attach.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}",args.interval_id):
            output=args.captures.resolve()/"absence_intervals"/args.interval_id
            if output.is_dir() and not (output/"manifest.json").exists():
                write_json(output/"progress.json",dict(status="canceled",error=str(error),contains_ground_truth=False))
        print(json.dumps(dict(status="canceled",error=str(error),contains_ground_truth=False)));sys.exit(2)
    finally:
        if guard:guard.close()


if __name__=="__main__":main()
