import copy
import hashlib
import builtins
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import test_yolo_ball_experiments as fixtures
from yolo_ball_experiments import build_plan,remember
from yolo_paired_data import inspect_training_sources,prepare_records


class PairedTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.YoloExperimentTests()
        self.fixture.setUp()
        self.fixture.ready_fixture()
        self.root=self.fixture.root
        self.captures=self.fixture.captures
        self.fixture.capture("pc_train",180,"pc_train")
        self.fixture.capture("pc_val",220,"pc_val")
        self.registry=self.captures/"training_sessions.json"
        sessions={name:dict(role=role,stage=stage,sealed=role=="test") for name,role,stage in
                  [("train","train","existing"),("pc_train","train","A"),("val","val","B"),("pc_val","val","B"),("test","test","C")]}
        captures={name:dict(session_id=name,domain="pc_camera" if name.startswith("pc") else "iphone") for name in sessions}
        self.registry.write_text(json.dumps(dict(schema_version=1,sessions=sessions,captures=captures)),encoding="utf-8")

    def tearDown(self): self.fixture.tearDown()

    def inspect(self):
        dependencies={}
        report=inspect_training_sources(self.captures,self.registry,self.fixture.match,
            lambda path,expected=None:remember(dependencies,path,expected))
        return report,dependencies

    def test_c_annotation_and_images_are_never_read_or_hashed(self):
        sealed=self.captures/"test"
        original_bytes,original_text=Path.read_bytes,Path.read_text
        def safe_bytes(path,*args,**kwargs):
            if path.is_relative_to(sealed): self.fail("Sealed test file opened")
            return original_bytes(path,*args,**kwargs)
        def safe_text(path,*args,**kwargs):
            if path.is_relative_to(sealed): self.fail("Sealed test file opened")
            return original_text(path,*args,**kwargs)
        with patch.object(Path,"read_bytes",safe_bytes),patch.object(Path,"read_text",safe_text):
            report,deps=self.inspect()
        self.assertFalse(report["test_labels_consumed"])
        self.assertEqual(report["sealed_test_readiness"],"NOT_AUDITED")
        self.assertFalse(any(Path(p).is_relative_to(sealed) for p in deps))

    def test_minimums_are_actual_data_blockers_not_registry_claims(self):
        report,_=self.inspect()
        self.assertIn("TRAIN_2000_VISIBLE_REQUIRED",report["blockers"])
        self.assertIn("TRAIN_500_AUDITED_POSITIVE_WINDOWS_REQUIRED",report["blockers"])
        self.assertIn("VAL_PC_CAMERA_120S_VERIFIED_ABSENCE_REQUIRED",report["blockers"])
        self.assertEqual(report["counts"]["train"]["iphone"],dict(visible=20,absent=10))

    def test_no_registry_blocks_and_does_not_infer_devices(self):
        self.registry.unlink()
        report,_=self.inspect()
        self.assertEqual(report["blockers"],["PAIRED_SESSION_REGISTRY_MISSING"])
        self.assertEqual(report["records"],[])

    def test_session_identity_cannot_be_reassigned_by_metadata(self):
        cap=self.captures/"train"/"capture.json"
        value=json.loads(cap.read_text()); value["session_id"]="someone_else"
        cap.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"session identity"): self.inspect()

    def test_conflicting_copies_are_both_excluded_with_actionable_blockers(self):
        original=self.captures/"train"/"capture.json"
        capture=json.loads(original.read_text());capture["original_sha256"]="a"*64
        original.write_text(json.dumps(capture))
        copied=self.captures/"train_copy"
        shutil.copytree(self.captures/"train",copied)
        registry=json.loads(self.registry.read_text())
        registry["captures"]["train_copy"]=dict(session_id="train",domain="iphone")
        self.registry.write_text(json.dumps(registry))
        annotations=json.loads((copied/"annotations.json").read_text())
        annotations["frame_11.jpg"]["ball"]=[[.5,.5,.1,.1]]
        (copied/"annotations.json").write_text(json.dumps(annotations))
        report,_=self.inspect()
        self.assertIn("CONFLICTING_SOURCE_FRAME_LABELS_OR_SPLIT",report["blockers"])
        self.assertIn("CONFLICTING_DUPLICATE_IMAGE_LABELS",report["blockers"])
        self.assertEqual(report["counts"]["train"]["iphone"],dict(visible=19,absent=10))
        self.assertEqual(report["review_audit"]["conflicting_unique_records_excluded"],1)
        self.assertEqual(report["review_audit"]["identical_reviewed_image_copies_deduplicated"],29)
        self.assertFalse(any(r["domain"]=="iphone" and r["frame_id"]==11 and r["role"]=="train" for r in report["records"]))
        self.assertEqual(len(report["annotation_conflicts"]),2)

    def test_same_original_disjoint_frames_cannot_become_different_sessions_roles_or_devices(self):
        train=self.captures/"train";copied=self.captures/"source_reimport"
        shutil.copytree(train,copied)
        for folder in (train,copied):
            value=json.loads((folder/"capture.json").read_text());value["original_sha256"]="a"*64
            if folder==copied:value["session_id"]="fake_new_session"
            (folder/"capture.json").write_text(json.dumps(value))
        rows=(copied/"frames.jsonl").read_text().splitlines()
        (train/"frames.jsonl").write_text("\n".join(rows[:15])+"\n")
        (copied/"frames.jsonl").write_text("\n".join(rows[15:])+"\n")
        registry=json.loads(self.registry.read_text())
        registry["sessions"]["fake_new_session"]=dict(role="val",stage="B",sealed=False)
        registry["captures"]["source_reimport"]=dict(session_id="fake_new_session",domain="pc_camera")
        self.registry.write_text(json.dumps(registry))
        report,_=self.inspect()
        self.assertIn("ORIGINAL_SOURCE_SESSION_ROLE_DEVICE_CONFLICT",report["blockers"])
        self.assertFalse(any(r["source_id"]=="a"*64 for r in report["records"]))
        self.assertEqual(report["review_audit"]["whole_source_conflicting_captures_excluded"],2)

    def test_sealed_source_registry_metadata_blocks_leak_without_opening_c(self):
        cap=self.captures/"train"/"capture.json";value=json.loads(cap.read_text());value["original_sha256"]="a"*64
        cap.write_text(json.dumps(value));registry=json.loads(self.registry.read_text())
        registry["captures"]["test"]["source_sha256"]="a"*64;self.registry.write_text(json.dumps(registry))
        original=Path.read_bytes
        def no_c(path,*args,**kw):
            if path.is_relative_to(self.captures/"test"):self.fail("C contents opened")
            return original(path,*args,**kw)
        with patch.object(Path,"read_bytes",no_c):report,_=self.inspect()
        self.assertIn("ORIGINAL_SOURCE_SESSION_ROLE_DEVICE_CONFLICT",report["blockers"])
        self.assertFalse(any(r["source_id"]=="a"*64 for r in report["records"]))

    def test_registry_source_identity_change_fails_closed(self):
        registry=json.loads(self.registry.read_text());registry["captures"]["train"]["source_sha256"]="a"*64
        self.registry.write_text(json.dumps(registry));report,_=self.inspect()
        self.assertIn("CAPTURE_SOURCE_SHA_REGISTRY_CONFLICT",report["blockers"])
        self.assertEqual(report["counts"]["train"]["iphone"],dict(visible=20,absent=10))

    def test_missing_sealed_source_identity_blocks_independence_without_reading_c(self):
        report,_=self.inspect()
        self.assertIn("SEALED_C_SOURCE_IDENTITY_METADATA_REQUIRED",report["blockers"])
        self.assertFalse(report["test_labels_consumed"])
        self.assertEqual(report["sealed_test_readiness"],"NOT_AUDITED")

    def test_missing_original_blocks_fitting_but_retains_reviewed_progress_counts(self):
        cap=self.captures/"train"/"capture.json"
        capture=json.loads(cap.read_text());capture.update(source="recorded_camera",
            original_path=str(self.root/"unavailable.MOV"),original_sha256="a"*64)
        cap.write_text(json.dumps(capture))
        report,_=self.inspect()
        self.assertIn("ORIGINAL_RECORDING_UNAVAILABLE",report["blockers"])
        self.assertEqual(report["counts"]["train"]["iphone"],dict(visible=20,absent=10))
        self.assertEqual(report["original_source_issues"][0]["reason"],"original_recording_unavailable")
        self.assertTrue(all(r["original_source_verified"] is False for r in report["records"] if r.get("clip")=="train"))

    def test_hash_checked_relocation_preserves_original_manifest_and_frozen_metadata(self):
        moved=self.root/"new location"/"original.MOV"
        moved.parent.mkdir();moved.write_bytes(b"verified-original-content")
        cap=self.captures/"train"/"capture.json"
        capture=json.loads(cap.read_text());capture.update(source="recorded_camera",
            original_path=str(self.root/"old location"/"original.MOV"),
            original_sha256=hashlib.sha256(moved.read_bytes()).hexdigest())
        cap.write_text(json.dumps(capture));original_bytes=cap.read_bytes()
        sidecar=self.captures/"source_locations.json"
        sidecar.write_text(json.dumps(dict(schema_version=1,sources={capture["original_sha256"]:dict(paths=[str(moved)],reverified_at="2026-10-11T00:00:00Z")})))
        report,deps=self.inspect()
        self.assertNotIn("ORIGINAL_RECORDING_UNAVAILABLE",report["blockers"])
        self.assertEqual(cap.read_bytes(),original_bytes)
        self.assertIn(sidecar.resolve().as_posix(),deps)
        self.assertIn(moved.resolve().as_posix(),deps)
        for r in report["records"]:
            if r.get("clip")=="train":
                self.assertTrue(r["original_source_verified"])
                self.assertEqual(r["resolved_source_path"],moved.resolve().as_posix())
        moved.write_bytes(b"different-content")
        changed,_=self.inspect()
        self.assertIn("ORIGINAL_RECORDING_PROVENANCE_INVALID",changed["blockers"])
        self.assertFalse(any(r.get("original_source_verified") for r in changed["records"] if r.get("clip")=="train"))

    def test_no_windows_from_structural_consecutive_ids_alone(self):
        report,_=self.inspect()
        self.assertEqual(report["positive_train_windows"],0)
        folder=self.captures/"train"
        registry=json.loads(self.registry.read_text())
        registry["sessions"]["train"]["stage"]="A"
        self.registry.write_text(json.dumps(registry))
        import hashlib
        (folder/"continuity_audit.json").write_text(json.dumps(dict(schema_version=1,reviewed=True,
            index_sha256=hashlib.sha256((folder/"frames.jsonl").read_bytes()).hexdigest(),
            source_id="capture:train",source_frame_ranges=[[1,30]])))
        report,_=self.inspect()
        self.assertEqual(report["positive_train_windows"],0)
        continuity=json.loads((folder/"continuity_audit.json").read_text())
        continuity["image_sha256"]={f"frame_{i}.jpg":hashlib.sha256((folder/f"frame_{i}.jpg").read_bytes()).hexdigest() for i in range(1,31)}
        (folder/"continuity_audit.json").write_text(json.dumps(continuity))
        annotations=json.loads((folder/"annotations.json").read_text())
        for i in range(11,31):
            annotations[f"frame_{i}.jpg"]["geometry"]["blur_streak"]=dict(reviewed=True,method="manual_streak_endpoints",
                endpoints=[[.35,.35],[.35,.35]],source_width=32,source_height=24)
        (folder/"annotations.json").write_text(json.dumps(annotations))
        rows=[json.loads(line) for line in (folder/"frames.jsonl").read_text().splitlines()]
        for i,row in enumerate(rows): row["burst_index"]=i
        (folder/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        continuity["index_sha256"]=hashlib.sha256((folder/"frames.jsonl").read_bytes()).hexdigest()
        (folder/"continuity_audit.json").write_text(json.dumps(continuity))
        report,_=self.inspect()
        self.assertEqual(report["positive_train_windows"],18)

    def test_prepared_references_exact_and_no_test_export(self):
        report,_=self.inspect()
        output=prepare_records(report,self.root/"paired_export")
        plan=build_plan(self.captures,prepared=output,weights=self.fixture.weights,match_root=self.fixture.match,registry=self.registry)
        self.assertFalse(plan["ready_for_fitting"])
        self.assertFalse(plan["guarantees"]["test_labels_consumed"])
        self.assertFalse((output/"test.txt").exists())
        label=next((output/"labels"/"train").glob("*.txt"))
        label.write_text("0 .2 .2 .1 .1")
        with self.assertRaisesRegex(ValueError,"provenance"): build_plan(self.captures,prepared=output,weights=self.fixture.weights,match_root=self.fixture.match,registry=self.registry)

    def test_fixed_batch_and_final_seeds(self):
        from yolo_ball_experiments import variant_config,variant_metadata
        self.assertEqual(variant_metadata("data_only640")["stage"],"A")
        self.assertEqual(variant_metadata("high_detail960")["stage"],"C")
        for seed in (42,43,44):
            config=variant_config(f"final640_seed{seed}",8)
            self.assertEqual((config["epochs"],config["patience"],config["seed"],config["batch"]),(60,12,seed,8))
        with self.assertRaisesRegex(ValueError,"16, 8 or 4"): variant_config("data_only640",2)
        self.assertEqual(variant_config("camera_aug640")["close_mosaic"],0)
        self.assertEqual(variant_config("data_only640")["patience"],0)

    def test_final_recipe_keeps_selected_960_input_and_independent_best640(self):
        from yolo_ball_experiments import variant_config,variant_metadata
        best640=dict(freeze=10,geometry="center_square30",augmentation="common_preset")
        selected=dict(freeze=0,geometry="reviewed_extent",augmentation="camera_safe",imgsz=960)
        for seed in (42,43,44):
            for name in (f"final640_seed{seed}",f"final_seed{seed}"):
                config=variant_config(name,8,best640,selected)
                self.assertEqual((config["imgsz"],config["freeze"],config["epochs"],config["patience"],config["seed"]),(960,0,60,12,seed))
                self.assertEqual(config["mosaic"],0)
                self.assertEqual(variant_metadata(name,8,best640,selected)["geometry"],"reviewed_extent")
        self.assertEqual(variant_config("extent640",8,best640,selected)["imgsz"],640)
        self.assertEqual(variant_config("extent640",8,best640,selected)["freeze"],10)
        self.assertTrue(variant_metadata("high_detail960")["native_current_shape_compatible"])
        with self.assertRaisesRegex(ValueError,"Selected recipe"):
            variant_config("final_seed42",8,best640,{**selected,"imgsz":1280})

    def test_installed_runtime_defaults_are_frozen_and_version_drift_rejected(self):
        from yolo_ball_experiments import runtime_policy,validate_saved_plan
        original_import=builtins.__import__
        def no_heavy_import(name,*args,**kwargs):
            if name.startswith(("ultralytics","torch","cv2","numpy")):
                self.fail("Planning runtime snapshot imported a heavy package")
            return original_import(name,*args,**kwargs)
        dependencies={}
        with patch("builtins.__import__",side_effect=no_heavy_import): policy=runtime_policy(dependencies)
        self.assertIsNotNone(policy["distribution_versions"]["ultralytics"])
        self.assertIsNotNone(policy["distribution_versions"]["torch"])
        self.assertEqual(dependencies[policy["ultralytics_defaults_path"]],policy["ultralytics_defaults_sha256"])
        plan=build_plan(self.captures,weights=self.fixture.weights,match_root=self.fixture.match,registry=self.registry)
        self.assertEqual(plan["runtime_policy"],policy)
        validate_saved_plan(plan)
        import yolo_ball_experiments as runner
        actual_version=runner.importlib.metadata.version
        def changed_version(name): return "changed-version" if name=="torch" else actual_version(name)
        with patch.object(runner.importlib.metadata,"version",side_effect=changed_version):
            with self.assertRaisesRegex(ValueError,"runtime/defaults changed"):validate_saved_plan(plan)


if __name__=="__main__": unittest.main()
