"""CPU-only source decode, lineage, cancellation and export acceptance checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from prepare_absence_interval import prepare_interval,request_for,sha,OwnerLifetime,IntervalCanceled,probe,reviewed_source_conflicts
from audit_absence_intervals import audit_confirmed_interval,load_confirmed_intervals
from yolo_paired_data import inspect_training_sources,prepare_records
from yolo_ball_experiments import remember


class CompleteAbsenceTests(unittest.TestCase):
    def setUp(self):
        temp=Path(__file__).resolve().parents[1]/".tmp";temp.mkdir(exist_ok=True)
        self.root=Path(tempfile.mkdtemp(prefix="absence_acceptance_",dir=temp))
        self.captures=self.root/"captures";self.captures.mkdir()
        self.source=self.root/"original ü gameplay.mp4"
        writer=cv2.VideoWriter(str(self.source),cv2.VideoWriter_fourcc(*"mp4v"),30.,(160,96))
        self.assertTrue(writer.isOpened(),"CPU MP4 fixture writer required")
        for frame in range(45):
            image=np.zeros((96,160,3),np.uint8);image[:,:,0]=50+frame
            cv2.putText(image,str(frame+1),(10,60),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,180,90),2)
            writer.write(image)
        writer.release()
        self.clip=self.captures/"clip_a";self.clip.mkdir()
        self.capture=dict(source="recorded_camera",session_id="sessionA",original_path=str(self.source),original_sha256=sha(self.source))
        (self.clip/"capture.json").write_text(json.dumps(self.capture),encoding="utf-8")
        (self.clip/"frames.jsonl").write_text("",encoding="utf-8")
        (self.clip/"annotations.json").write_text("{}",encoding="utf-8")
        self.registry=dict(schema_version=1,sessions={"sessionA":dict(role="train",stage="A",sealed=False)},
            captures={"clip_a":dict(session_id="sessionA",domain="pc_camera")})
        self.registry_path=self.captures/"training_sessions.json";self.save_registry()

    def tearDown(self):shutil.rmtree(self.root)

    def save_registry(self):self.registry_path.write_text(json.dumps(self.registry),encoding="utf-8")

    def prepare(self,start=5,end=20,id="interval_1",**kw):
        result=prepare_interval(self.captures,"clip_a",start,end,id,**kw)
        return result,self.captures/"absence_intervals"/id

    def confirm(self,folder):
        manifest=json.loads((folder/"manifest.json").read_text())
        keys=("interval_id","clip","session_id","domain","role","start_source_frame","end_source_frame",
            "source_sha256","source_capture_sha256","index_sha256","source_index_sha256","duration_s")
        review={k:manifest[k] for k in keys}
        review.update(schema_version=1,reviewed=True,method="manual_full_original_interval_absence",reviewed_at="2026-10-11T00:00:00+00:00",
            manifest_sha256=sha(folder/"manifest.json"),confirm_entire_interval=True,confirm_no_ball=True,
            no_hidden_or_unsure=True,watched_full_original_interval=True)
        (folder/"review.json").write_text(json.dumps(review),encoding="utf-8")

    def audit(self,folder):
        dependencies={}
        return audit_confirmed_interval(folder,self.captures,self.registry,lambda p,e=None:remember(dependencies,p,e))

    def test_sequential_decode_accounts_every_frame_and_never_creates_labels(self):
        originals={p:p.read_bytes() for p in (self.source,self.clip/"capture.json",self.clip/"annotations.json")}
        manifest,folder=self.prepare()
        self.assertEqual(manifest["source_decoded_frames"],45)
        self.assertEqual(manifest["source_reported_frames"],45)
        self.assertEqual(manifest["interval_frames"],16)
        self.assertAlmostEqual(manifest["duration_s"],16/30,places=8)
        self.assertEqual(len((folder/"source_frames.jsonl").read_text().splitlines()),45)
        self.assertEqual(len(list(folder.glob("frame_*.jpg"))),16)
        self.assertFalse((folder/"review.json").exists())
        self.assertEqual(self.audit(folder)["status"],"UNREVIEWED")
        self.assertFalse(manifest["contains_ground_truth"])
        self.assertTrue(any(p.endswith("recording_source.py") for p in manifest["preparation_dependencies_sha256"]))
        for path,data in originals.items():self.assertEqual(path.read_bytes(),data)

    def test_eof_tail_has_zero_duration_instead_of_nominal_fps(self):
        manifest,folder=self.prepare(30,45)
        rows=[json.loads(line) for line in (folder/"frames.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["frame_duration_s"],0.)
        self.assertAlmostEqual(manifest["duration_s"],15/30,places=8)
        self.confirm(folder);self.assertEqual(self.audit(folder)["status"],"VERIFIED")

    def test_only_bound_explicit_review_enters_training_export_and_duration(self):
        self.registry["sessions"]["sessionA"].update(role="val",stage="B");self.save_registry()
        manifest,folder=self.prepare();self.confirm(folder)
        result=self.audit(folder)
        self.assertEqual(len(result["records"]),16)
        self.assertTrue(all(not r["positive"] and r["annotation"]["visibility"]=="absent" for r in result["records"]))
        dependencies={};report=inspect_training_sources(self.captures,self.registry_path,self.root/"no_match",
            lambda p,e=None:remember(dependencies,p,e))
        self.assertEqual(report["counts"]["val"]["pc_camera"]["absent"],16)
        self.assertAlmostEqual(report["validation_absence_s"]["pc_camera"],manifest["duration_s"],places=8)
        self.assertIn("VAL_PC_CAMERA_120S_VERIFIED_ABSENCE_REQUIRED",report["blockers"])
        prepared=self.root/"export";prepare_records(report,prepared)
        labels=list((prepared/"labels"/"val").glob("*.txt"))
        self.assertEqual(len(labels),16);self.assertTrue(all(p.read_text()=="" for p in labels))

    def test_complete_120s_validation_interval_uses_observed_next_source_pts(self):
        writer=cv2.VideoWriter(str(self.source),cv2.VideoWriter_fourcc(*"mp4v"),15.,(160,96))
        self.assertTrue(writer.isOpened())
        for frame in range(1802):
            image=np.zeros((96,160,3),np.uint8)
            cv2.putText(image,str(frame+1),(2,60),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,180,90),2)
            writer.write(image)
        writer.release();self.capture["original_sha256"]=sha(self.source)
        (self.clip/"capture.json").write_text(json.dumps(self.capture))
        self.registry["sessions"]["sessionA"].update(role="val",stage="B");self.save_registry()
        manifest,folder=self.prepare(1,900);self.confirm(folder)
        second,other=self.prepare(901,1800,id="interval_2");self.confirm(other)
        self.assertEqual(manifest["source_decoded_frames"],1802)
        self.assertAlmostEqual(manifest["duration_s"],60.,places=8)
        self.assertAlmostEqual(second["duration_s"],60.,places=8)
        dependencies={};report=inspect_training_sources(self.captures,self.registry_path,self.root/"no_match",
            lambda p,e=None:remember(dependencies,p,e))
        self.assertAlmostEqual(report["validation_absence_s"]["pc_camera"],120.,places=8)
        self.assertNotIn("VAL_PC_CAMERA_120S_VERIFIED_ABSENCE_REQUIRED",report["blockers"])
        self.assertIn("VAL_PC_CAMERA_1000_VISIBLE_REQUIRED",report["blockers"])

    def test_overlap_counts_source_pts_once_and_timing_disagreement_blocks(self):
        _,first=self.prepare(5,20);self.confirm(first)
        _,second=self.prepare(10,25,id="interval_2");self.confirm(second)
        dependencies={};result=load_confirmed_intervals(self.captures,self.registry,lambda p,e=None:remember(dependencies,p,e))
        self.assertAlmostEqual(result["coverage_seconds"]["train"]["pc_camera"],21/30,places=8)
        original=audit_confirmed_interval
        def inconsistent(folder,*args):
            item=original(folder,*args)
            if folder.name=="interval_2":item["records"][0]["frame_duration_s"]+=.001
            return item
        with patch("audit_absence_intervals.audit_confirmed_interval",side_effect=inconsistent):
            result=load_confirmed_intervals(self.captures,self.registry,lambda p,e=None:remember(dependencies,p,e))
        self.assertIn("COMPLETE_ABSENCE_SOURCE_TIMING_CONFLICT",result["blockers"])
        self.assertFalse(any(r["frame_id"]==10 for r in result["records"]))

    def test_source_index_image_and_capture_changes_fail_closed(self):
        for name in ("frames.jsonl","source_frames.jsonl","frame_5.jpg"):
            with self.subTest(name=name):
                _,folder=self.prepare(id="tamper_"+name.replace(".","_"));self.confirm(folder)
                target=folder/name;target.write_bytes(target.read_bytes()+b" ")
                with self.assertRaises(ValueError):self.audit(folder)
        _,folder=self.prepare(id="tamper_capture");self.confirm(folder)
        (self.clip/"capture.json").write_text(json.dumps({**self.capture,"unused":"changed"}))
        with self.assertRaises(ValueError):self.audit(folder)

    def test_prior_visible_partial_hidden_unknown_reviews_block_bulk_absence(self):
        for visibility in ("visible","partial","occluded","unknown"):
            (self.clip/"frames.jsonl").write_text(json.dumps(dict(file="frame_10.jpg",source_frame=10))+"\n")
            value=dict(reviewed_classes={"ball":True},ball=[],geometry=dict(visibility=visibility))
            (self.clip/"annotations.json").write_text(json.dumps({"frame_10.jpg":value}))
            original=(self.clip/"annotations.json").read_bytes()
            with self.assertRaisesRegex(ValueError,"prior ball review"):self.prepare(id="conflict_"+visibility)
            self.assertEqual((self.clip/"annotations.json").read_bytes(),original)

    def test_unreviewed_suggestions_are_never_ground_truth_or_conflicts(self):
        (self.clip/"frames.jsonl").write_text(json.dumps(dict(file="frame_10.jpg",source_frame=10))+"\n")
        (self.clip/"annotations.json").write_text(json.dumps({"frame_10.jpg":dict(reviewed=False,ball=[[.5,.5,.1,.1]])}))
        _,folder=self.prepare();self.assertEqual(self.audit(folder)["status"],"UNREVIEWED")

    def test_api_owned_startup_directory_is_accepted_but_no_overwrites(self):
        output=self.captures/"absence_intervals"/"interval_1";output.mkdir(parents=True)
        request=request_for(self.registry,"clip_a",5,20,"interval_1")
        (output/"request.json").write_text(json.dumps(request));(output/"job.json").write_text("{}")
        self.prepare()
        with self.assertRaises(FileExistsError):self.prepare()

    def test_cancel_marker_and_disk_budget_never_publish_complete_manifest(self):
        output=self.captures/"absence_intervals"/"interval_1";output.mkdir(parents=True)
        (output/"request.json").write_text(json.dumps(request_for(self.registry,"clip_a",5,20,"interval_1")))
        (output/"cancel.json").write_text(json.dumps(dict(schema_version=1,cancel_requested=True)))
        with self.assertRaises(IntervalCanceled):self.prepare()
        self.assertFalse((output/"manifest.json").exists())
        with patch("prepare_absence_interval.shutil.disk_usage",return_value=type("Disk",(),{"free":1024})()):
            with self.assertRaisesRegex(OSError,"DISK_BUDGET"):self.prepare(id="disk_budget")
        folder=output.parent/"disk_budget"
        self.assertFalse((folder/"manifest.json").exists())
        self.assertEqual(json.loads((folder/"progress.json").read_text())["status"],"failed")

    def test_c_is_skipped_before_capture_or_annotation_or_image_reads(self):
        self.registry["sessions"]["sessionA"].update(role="test",stage="C",sealed=True);self.save_registry()
        with patch.object(Path,"read_text",side_effect=AssertionError("C content opened")):
            with self.assertRaisesRegex(ValueError,"Sealed C"):request_for(self.registry,"clip_a",5,20,"sealed")
        output=self.captures/"absence_intervals"/"sealed";output.mkdir(parents=True)
        (output/"request.json").write_text(json.dumps(dict(schema_version=1,clip="clip_a")))
        dependencies={};result=load_confirmed_intervals(self.captures,self.registry,lambda p,e=None:remember(dependencies,p,e))
        self.assertEqual(result["sealed_not_opened"],1)
        self.assertEqual(list(dependencies),[(output/"request.json").resolve().as_posix()])

    def test_sealed_original_cannot_be_decoded_by_a_development_alias(self):
        self.registry["sessions"]["C"]=dict(role="test",stage="C",sealed=True)
        self.registry["captures"]["sealed_alias"]=dict(session_id="C",domain="pc_camera",source_sha256=self.capture["original_sha256"])
        self.save_registry()
        # No C directory even exists; its contents cannot be consulted.
        with patch("cv2.VideoCapture",side_effect=AssertionError("Sealed original decoded")):
            with self.assertRaisesRegex(ValueError,"another session/role/device"):self.prepare()

    @unittest.skipUnless(os.name=="nt","Windows directory junction semantics")
    def test_unassigned_junction_to_c_is_rejected_before_c_content_read(self):
        sealed=self.captures/"sealed_C";sealed.mkdir()
        (sealed/"capture.json").write_text("C contents must not be parsed")
        self.registry["sessions"]["C"]=dict(role="test",stage="C",sealed=True)
        self.registry["captures"]["sealed_C"]=dict(session_id="C",domain="pc_camera")
        junction=self.captures/"alias_to_C"
        made=subprocess.run(["cmd","/c","mklink","/J",str(junction),str(sealed)],capture_output=True,text=True)
        self.assertEqual(made.returncode,0,made.stdout+made.stderr)
        reads=[]
        with self.assertRaisesRegex(ValueError,"alias/junction"):
            reviewed_source_conflicts(self.captures,self.capture["original_sha256"],5,20,
                lambda p,e=None:reads.append(str(p)),self.registry)
        self.assertEqual(reads,[])

    def test_annotation_alias_is_rejected_before_evidence_read(self):
        real_resolve=Path.resolve;annotation=self.clip/"annotations.json"
        def fake_resolve(path,*args,**kwargs):
            return self.root/"outside_C_annotations.json" if path==annotation else real_resolve(path,*args,**kwargs)
        with patch.object(Path,"resolve",fake_resolve),self.assertRaisesRegex(ValueError,"file alias/junction"):
            reviewed_source_conflicts(self.captures,self.capture["original_sha256"],5,20,lambda p,e=None:sha(p),self.registry)

    def test_app_job_failure_or_cancellation_cannot_credit_a_prepared_manifest(self):
        _,folder=self.prepare();self.confirm(folder)
        (folder/"job.json").write_text(json.dumps(dict(status="failed")))
        with self.assertRaisesRegex(ValueError,"job must complete"):self.audit(folder)
        (folder/"job.json").write_text(json.dumps(dict(status="completed")))
        self.assertEqual(self.audit(folder)["status"],"VERIFIED")
        (folder/"cancel.json").write_text(json.dumps(dict(schema_version=1,cancel_requested=True)))
        with self.assertRaisesRegex(ValueError,"Canceled interval"):self.audit(folder)

    def test_reported_decode_count_mismatch_and_pts_gap_never_publish(self):
        class FakeVideo:
            def __init__(self,wrongcount=False,gap=False):self.frame=0;self.wrongcount=wrongcount;self.gap=gap
            def isOpened(self):return True
            def get(self,key):
                if key==cv2.CAP_PROP_FRAME_COUNT:return 5 if self.wrongcount else 4
                if key==cv2.CAP_PROP_POS_FRAMES:return self.frame
                if key==cv2.CAP_PROP_POS_MSEC:return (self.frame-1)*(200 if self.gap else 30)
            def read(self):
                self.frame+=1
                return (True,np.zeros((8,8,3),np.uint8)) if self.frame<=4 else (False,None)
            def release(self):pass
        for name,video in (("count",FakeVideo(wrongcount=True)),("gap",FakeVideo(gap=True))):
            with patch("cv2.VideoCapture",return_value=video),self.assertRaises(RuntimeError):self.prepare(1,4,id=name)
            self.assertFalse((self.captures/"absence_intervals"/name/"manifest.json").exists())

    @unittest.skipUnless(os.name=="nt","Windows owner process handle semantics")
    def test_actual_owner_death_cancels_app_worker_with_partial_output(self):
        output=self.captures/"absence_intervals"/"owner_test";output.mkdir(parents=True)
        owner=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
        code=("import sys,time;from pathlib import Path;from prepare_absence_interval import OwnerLifetime;"
            "p=Path(sys.argv[2]);g=OwnerLifetime(int(sys.argv[1]),force_exit=True);g.attach(p);"
            "(p/'frame_1.jpg').write_bytes(b'partial');(p/'ready').write_text('ready');time.sleep(30)")
        worker=subprocess.Popen([sys.executable,"-B","-c",code,str(owner.pid),str(output)],cwd=Path(__file__).parent)
        try:
            deadline=time.monotonic()+8
            while not (output/"ready").exists() and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue((output/"ready").exists(),"Owner guard process startup")
            owner.terminate();owner.wait(timeout=3);self.assertEqual(worker.wait(timeout=3),2)
            self.assertEqual(json.loads((output/"progress.json").read_text())["status"],"canceled")
            self.assertFalse((output/"manifest.json").exists())
        finally:
            for process in (worker,owner):
                if process.poll() is None:process.terminate();process.wait(timeout=3)

    def test_cpu_probe_reports_real_decoder_availability(self):
        result=probe();self.assertTrue(result["available"]);self.assertFalse(result["gpu_required"])


if __name__=="__main__":unittest.main()
