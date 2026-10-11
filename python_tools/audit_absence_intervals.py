"""Consume only explicitly confirmed complete original absence intervals."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from PIL import Image
from prepare_absence_interval import request_for,reviewed_source_conflicts,require_source_binding,checked_child


def require(condition,message):
    if not condition:raise ValueError(message)


def _rows(path):
    data=Path(path).read_bytes();require(data and data.endswith(b"\n"),"Complete newline-terminated index required")
    return [json.loads(line) for line in data.splitlines()]


def audit_confirmed_interval(folder,captures,registry,remember):
    from recording_source import resolve_original_path
    folder=Path(folder).resolve();captures=Path(captures).resolve()
    require(folder.parent==captures/"absence_intervals","Interval folder escapes source root")
    request_path=checked_child(folder,"request.json");remember(request_path);request=json.loads(request_path.read_text(encoding="utf-8"))
    clip=request.get("clip");assignment=registry.get("captures",{}).get(clip)
    if assignment and registry.get("sessions",{}).get(assignment.get("session_id"),{}).get("role")=="test":
        return dict(status="SEALED_C_NOT_OPENED",records=[])
    expected=request_for(registry,clip,request.get("start_source_frame"),request.get("end_source_frame"),folder.name)
    require(request==expected,"Interval request/session/device/role differs from immutable registry")
    if not (folder/"review.json").is_file():return dict(status="UNREVIEWED",records=[])
    for name in ("request.json","review.json","manifest.json","progress.json","source_frames.jsonl","frames.jsonl"):
        checked_child(folder,name)
    require(not (folder/"cancel.json").exists(),"Canceled interval cannot enter training")
    remember(folder/"progress.json");progress=json.loads((folder/"progress.json").read_text(encoding="utf-8"))
    require(progress.get("status")=="prepared","Successful completed decoder progress required")
    if (folder/"job.json").exists():
        job_path=checked_child(folder,"job.json");remember(job_path)
        require(json.loads(job_path.read_text(encoding="utf-8")).get("status")=="completed","App-owned decoder job must complete successfully")
    manifest_hash=remember(folder/"manifest.json");review_hash=remember(folder/"review.json")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"));review=json.loads((folder/"review.json").read_text(encoding="utf-8"))
    require(all(manifest.get(k)==v for k,v in request.items()),"Prepared interval/request identity differs")
    require(manifest.get("status")=="prepared" and manifest.get("coverage_complete") is True and manifest.get("contains_ground_truth") is False,"Complete unlabelled interval preparation required")
    require(manifest.get("timestamp_basis")=="opencv_ffmpeg_source_pts_ms"
            and manifest.get("decoder",{}).get("name")=="opencv_ffmpeg"
            and manifest.get("decoder",{}).get("sequential") is True
            and manifest.get("decoder",{}).get("random_seek") is False
            and manifest.get("settings")==dict(jpeg_quality=95,full_resolution=True,source_pts=True,
               terminal_duration_policy="ZERO_WITHOUT_NEXT_PTS"),"Sequential original-PTS/full-resolution decoder policy required")
    require(all(isinstance(manifest.get(k),str) and re.fullmatch(r"[0-9a-f]{64}",manifest[k])
                for k in ("source_sha256","source_capture_sha256","index_sha256","source_index_sha256")),"Valid source/index SHA bindings required")
    required_attestations=("reviewed","confirm_entire_interval","confirm_no_ball","no_hidden_or_unsure","watched_full_original_interval")
    require(review.get("schema_version")==1 and review.get("method")=="manual_full_original_interval_absence" and all(review.get(k) is True for k in required_attestations),"Explicit full-original interval absence review required")
    require(review.get("manifest_sha256")==manifest_hash,"Reviewed manifest changed")
    bound=("interval_id","clip","session_id","domain","role","start_source_frame","end_source_frame",
           "source_sha256","source_capture_sha256","source_index_sha256","index_sha256","duration_s")
    require(all(review.get(k)==manifest.get(k) for k in bound),"Review/source/index/range bindings differ")
    timestamp=datetime.fromisoformat(review.get("reviewed_at","").replace("Z","+00:00"))
    require(timestamp.tzinfo is not None,"Explicit reviewed_at timezone required")
    capfolder=(captures/clip).resolve()
    require(capfolder.parent==captures and capfolder.name==clip,"Capture directory alias escapes root")
    capfile=checked_child(capfolder,"capture.json");remember(capfile,manifest["source_capture_sha256"])
    capture=json.loads(capfile.read_text(encoding="utf-8"))
    require(capture.get("source")=="recorded_camera" and capture.get("original_sha256")==manifest["source_sha256"]
            and capture.get("session_id")==manifest["session_id"] and capture.get("original_path")==manifest.get("original_path"),"Original capture/source/session identity differs")
    require_source_binding(captures,registry,request,manifest["source_sha256"],remember)
    source=resolve_original_path(capture,captures,remember)
    remember(folder/"source_frames.jsonl",manifest["source_index_sha256"])
    remember(folder/"frames.jsonl",manifest["index_sha256"])
    source_rows=_rows(folder/"source_frames.jsonl");selected=_rows(folder/"frames.jsonl")
    total=manifest.get("source_decoded_frames")
    require(type(total) is int and total==manifest.get("source_reported_frames")==len(source_rows),"Complete source decode accounting differs")
    start,end=request["start_source_frame"],request["end_source_frame"]
    require(end<=total and len(selected)==end-start+1==manifest.get("interval_frames"),"Selected interval frame coverage differs")
    for index,row in enumerate(source_rows):
        fid=index+1;pts=row.get("source_timestamp_s");duration=row.get("frame_duration_s")
        require(type(row.get("source_frame")) is int and row.get("source_frame")==fid and type(pts) in (int,float) and math.isfinite(pts) and pts>=0,"Source IDs/PTS invalid")
        require(type(duration) in (int,float) and math.isfinite(duration) and duration>=0,"Source duration invalid")
        if index+1<total:
            next_pts=source_rows[index+1].get("source_timestamp_s")
            require(type(next_pts) in (int,float) and 0<next_pts-pts<=.100000001 and abs(duration-(next_pts-pts))<=1e-9,"Source PTS gap/duration differs")
        else:require(duration==0,"EOF duration must be conservative zero without next actual PTS")
    beginning=source_rows[start-1]["source_timestamp_s"]
    ending=source_rows[end]["source_timestamp_s"] if end<total else source_rows[end-1]["source_timestamp_s"]
    require(manifest.get("source_start_s")==beginning and manifest.get("source_end_s")==ending
            and abs(manifest.get("duration_s",-1)-(ending-beginning))<=1e-9,"Interval PTS duration differs")
    conflicts=reviewed_source_conflicts(captures,manifest["source_sha256"],start,end,remember,registry)
    require(not conflicts,"Explicit prior visible/hidden/unknown review conflicts with bulk absence: "+json.dumps(conflicts))
    records=[]
    for row,fid in zip(selected,range(start,end+1)):
        source_row=source_rows[fid-1]
        require(type(row.get("source_frame")) is int and type(row.get("decoded_index")) is int
                and row.get("source_frame")==row.get("decoded_index")==fid and row.get("file")==f"frame_{fid}.jpg","Interval frame IDs/path differ")
        require(all(row.get(k)==source_row[k] for k in ("source_timestamp_s","frame_duration_s")),"Interval source PTS/duration differs")
        require(isinstance(row.get("image_sha256"),str) and re.fullmatch(r"[0-9a-f]{64}",row["image_sha256"]),"Interval image SHA binding required")
        image=checked_child(folder,row["file"]);image_hash=remember(image,row.get("image_sha256"))
        with Image.open(image) as opened:
            require(opened.format=="JPEG","Interval image must retain JPEG extraction format")
            rgb=opened.convert("RGB");width,height=rgb.size
            require([width,height]==manifest.get("source_size")==[row.get("width"),row.get("height")],"Interval frame dimensions differ")
            decoded_hash=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        records.append(dict(image=image.resolve().as_posix(),image_sha256=image_hash,decoded_sha256=decoded_hash,
            labels={"ball":[]},clip=clip,source_id=manifest["source_sha256"],frame_id=fid,
            timestamp_s=row["source_timestamp_s"],source_size=[width,height],domain=request["domain"],role=request["role"],
            stage=request["stage"],session_id=request["session_id"],burst_id=None,burst_index=None,
            original_source_verified=True,resolved_source_path=source.as_posix(),continuity_audited=True,precise_target=None,
            positive=False,rally_id=None,annotation=dict(reviewed=True,visibility="absent",center_px=None,tags=[]),
            absence_interval_id=request["interval_id"],absence_review_sha256=review_hash,frame_duration_s=row["frame_duration_s"]))
    return dict(status="VERIFIED",records=records,summary=dict(interval_id=request["interval_id"],clip=clip,
        source_sha256=manifest["source_sha256"],domain=request["domain"],role=request["role"],
        start_source_frame=start,end_source_frame=end,duration_s=ending-beginning,reviewed_frames=len(records),
        manifest_sha256=manifest_hash,review_sha256=review_hash,index_sha256=manifest["index_sha256"]))


def load_confirmed_intervals(captures,registry,remember):
    root=Path(captures)/"absence_intervals";records=[];summaries=[];blockers=[];issues=[];unreviewed=sealed=0
    if not root.is_dir():return dict(records=[],summaries=[],blockers=[],issues=[],unreviewed=0,sealed_not_opened=0,coverage_seconds={})
    timing={};timing_conflicts=set();coverage={}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():continue
        try:
            result=audit_confirmed_interval(folder,captures,registry,remember)
            if result["status"]=="VERIFIED":
                records.extend(result["records"]);summaries.append(result["summary"])
                for row in result["records"]:
                    source_key=(row["source_id"],row["frame_id"])
                    evidence=(row["timestamp_s"],row["frame_duration_s"])
                    if source_key in timing and any(abs(a-b)>1e-9 for a,b in zip(timing[source_key],evidence)):
                        timing_conflicts.add(source_key);blockers.append("COMPLETE_ABSENCE_SOURCE_TIMING_CONFLICT")
                        issues.append(dict(interval_id=folder.name,source_sha256=row["source_id"],source_frame=row["frame_id"],
                            error="Same original frame has conflicting audited PTS/duration"))
                    timing[source_key]=evidence
                    coverage.setdefault((row["role"],row["domain"]),{})[source_key]=row["frame_duration_s"]
            elif result["status"]=="UNREVIEWED":unreviewed+=1
            else:sealed+=1
        except (ValueError,OSError,KeyError,TypeError) as error:
            blockers.append("COMPLETE_ABSENCE_INTERVAL_AUDIT_FAILED")
            issues.append(dict(interval_id=folder.name,error=str(error)))
    seconds={}
    for (role,domain),frames in coverage.items():
        seconds.setdefault(role,{})[domain]=math.fsum(duration for key,duration in frames.items() if key not in timing_conflicts)
    if timing_conflicts:records=[r for r in records if (r["source_id"],r["frame_id"]) not in timing_conflicts]
    return dict(records=records,summaries=summaries,blockers=sorted(set(blockers)),issues=issues,unreviewed=unreviewed,
        sealed_not_opened=sealed,coverage_seconds=seconds)
