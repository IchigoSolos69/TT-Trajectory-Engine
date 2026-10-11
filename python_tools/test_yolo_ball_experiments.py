"""CPU-only controls/provenance tests. Fixture optimizer writes fake bytes only."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yolo_ball_experiments as experiments


class YoloExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.captures = self.root / "captures"
        self.captures.mkdir()
        self.weights = self.root / "baseline" / "best.pt"
        self.weights.parent.mkdir()
        self.weights.write_bytes(b"fixture checkpoint, not a fitted network")
        self.match = self.root / "match"
        self.match.mkdir()
        self.prepared = self.root / "prepared"
        self.splits = self.root / "splits.json"

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, name, seed, session, count=30):
        folder = self.captures / name
        folder.mkdir()
        cap = dict(source="live", session_id=session, sampled_frames=count, split=None)
        (folder / "capture.json").write_text(json.dumps(cap), encoding="utf-8")
        rows, annotations = [], {}
        for i in range(count):
            file = f"frame_{i+1}.jpg"
            Image.new("RGB", (32, 24), (seed, i * 7, 10)).save(folder / file)
            rows.append(dict(file=file, source_frame=i+1, elapsed_ms=i*16.7, burst_id=1))
            absent = i < 10
            annotations[file] = dict(reviewed=True,
                reviewed_classes=dict(ball=True, person=False, table=False),
                ball=[] if absent else [[.3, .3, .1, .1]], person=[], table=[],
                geometry=dict(visibility="absent" if absent else "visible"))
        (folder / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (folder / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
        return folder

    def ready_fixture(self, same_session=False):
        for i, split in enumerate(experiments.SPLITS):
            self.capture(split, 30 + i*50, "same" if same_session else split)
        self.splits.write_text(json.dumps({s: [s] for s in experiments.SPLITS}), encoding="utf-8")
        # Exact preparation policy, isolated from the real dataset/model paths.
        from prepare_live_training import load_reviewed
        rows, _ = load_reviewed(self.captures, json.loads(self.splits.read_text()), ("ball",)) if not same_session else ({}, 0)
        if same_session:
            return
        self.prepared.mkdir()
        audit = dict(schema_version=2, splits=json.loads(self.splits.read_text()),
                     contains_test_in_training=False, prepared_kinds=["ball"],
                     reviewed_counts_by_kind={"ball": {s: len(rows[s]) for s in experiments.SPLITS}}, frames=[])
        for split in experiments.SPLITS:
            images = self.prepared / "ball" / "images" / split
            labels_dir = self.prepared / "ball" / "labels" / split
            images.mkdir(parents=True); labels_dir.mkdir(parents=True)
            paths = []
            for image, labels, digest, clip in rows[split]:
                copied = images / f"{clip}_{image.name}"
                shutil.copyfile(image, copied)
                (labels_dir / f"{copied.stem}.txt").write_text(experiments.label_text(labels), encoding="utf-8")
                paths.append(copied.resolve().as_posix())
                audit["frames"].append(dict(kind="ball", split=split, clip=clip, file=image.name,
                                          sha256=digest, objects={"ball": len(labels["ball"])}))
            (self.prepared / "ball" / f"live_{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
            if split != "test":
                directory = self.match / "images" / split
                directory.mkdir(parents=True)
                label_root = self.match / "labels" / split
                label_root.mkdir(parents=True)
                match_image = directory / f"game1_fixture_{split}.jpg"
                Image.new("RGB", (32, 24), (20, 10, 40 if split == "train" else 200)).save(match_image)
                (label_root / f"{match_image.stem}.txt").write_text("0 .5 .5 .1 .1", encoding="utf-8")
                (self.match / f"{split}.txt").write_text(match_image.as_posix() + "\n", encoding="utf-8")
                (self.prepared / "ball" / f"{split}.txt").write_text(match_image.as_posix() + "\n" + "\n".join(paths) + "\n", encoding="utf-8")
        (self.prepared / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
        yaml = (f"train: {(self.prepared/'ball'/'train.txt').as_posix()}\n"
                f"val: {(self.prepared/'ball'/'val.txt').as_posix()}\nnames:\n  0: ball\n")
        (self.prepared / "ball" / "data.yaml").write_text(yaml, encoding="utf-8")

    def plan(self, splits=True, prepared=True):
        return experiments.build_plan(self.captures, self.splits if splits else None,
            self.prepared if prepared else None, self.weights, match_root=self.match,fixture_only=True)

    def test_single_variable_variants_and_defensive_copy(self):
        baseline = experiments.variant_config("data_only640")
        self.assertFalse(baseline["amp"])
        self.assertFalse(baseline["cache"])
        self.assertFalse(baseline["plots"])
        for variant, difference in (("unfreeze640", "freeze"), ("high_detail960", "imgsz")):
            other = experiments.variant_config(variant)
            self.assertEqual({k for k in baseline if baseline[k] != other[k]}, {difference})
        baseline["freeze"] = 99
        self.assertEqual(experiments.variant_config("data_only640")["freeze"], 10)
        with self.assertRaises(ValueError):
            experiments.variant_config("invented")

    def test_plan_does_not_import_ultralytics_or_assign_splits(self):
        self.capture("one", 20, "one")
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith("ultralytics"):
                raise AssertionError("Planning must not load/train a model")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            result = self.plan(splits=False, prepared=False)
        self.assertFalse(result["ready_for_fitting"])
        self.assertIn("FROZEN_SPLITS_MISSING", result["blockers"])
        self.assertIn("INDEPENDENT_SESSIONS_INSUFFICIENT", result["blockers"])
        self.assertEqual(result["readiness"]["session_splits"], {"one": ["unassigned"]})
        output = self.root / "fit"
        with self.assertRaisesRegex(ValueError, "Fitting blocked"):
            experiments.run_experiment(result, "data_only640", output, model_factory=lambda _: self.fail("Must not start"))
        self.assertFalse(output.exists())

    def test_ready_fixture_and_native_shape_gate(self):
        self.ready_fixture()
        plan = self.plan()
        self.assertTrue(plan["ready_for_fitting"])
        self.assertFalse(plan["variants"]["high_detail960"]["native_current_shape_compatible"])
        self.assertTrue(plan["variants"]["data_only640"]["native_current_shape_compatible"])
        self.assertFalse(plan["guarantees"]["automatic_deployment"])
        self.assertIn(self.weights.as_posix(), plan["dependencies"])
        self.assertIn((self.prepared / "ball" / "labels" / "test" / "test_frame_1.txt").as_posix(), plan["dependencies"])
        self.assertEqual(experiments.validate_saved_plan(plan), plan)

    def test_same_session_split_leakage_blocks_fit(self):
        self.ready_fixture(same_session=True)
        result = self.plan(prepared=False)
        self.assertFalse(result["ready_for_fitting"])
        self.assertIn("SESSION_SPLIT_LEAKAGE", result["blockers"])

    def test_prepared_label_changes_fail_even_when_audit_claims_ready(self):
        self.ready_fixture()
        label = self.prepared / "ball" / "labels" / "train" / "train_frame_1.txt"
        label.write_text("0 .5 .5 .1 .1", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reviewed ball annotation"):
            self.plan()

    def test_test_image_in_train_list_is_rejected(self):
        self.ready_fixture()
        train = self.prepared / "ball" / "train.txt"
        train.write_text(train.read_text() + (self.prepared / "ball" / "images" / "test" / "test_frame_1.jpg").as_posix() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "membership"):
            self.plan()

    def test_plan_and_input_tampering_fail_before_output(self):
        self.ready_fixture()
        plan = self.plan()
        changed = copy.deepcopy(plan)
        changed["variants"]["data_only640"]["training"]["epochs"] = 13
        with self.assertRaisesRegex(ValueError, "plan was changed"):
            experiments.validate_saved_plan(changed)
        self.weights.write_bytes(b"altered baseline")
        with self.assertRaisesRegex(ValueError, "input changed"):
            experiments.run_experiment(plan, "data_only640", self.root / "fit", model_factory=lambda _: self.fail("Must not start"))
        self.assertFalse((self.root / "fit").exists())

    def test_review_revision_invalidates_saved_plan(self):
        self.ready_fixture()
        plan = self.plan()
        annotations = self.captures / "train" / "annotations.json"
        annotations.write_text(annotations.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input changed"):
            experiments.validate_saved_plan(plan)

    def test_match_image_duplicated_from_holdout_is_rejected(self):
        self.ready_fixture()
        original = self.match / "images" / "train" / "game1_fixture_train.jpg"
        shutil.copyfile(self.captures / "test" / "frame_1.jpg", original)
        with self.assertRaisesRegex(ValueError, "image overlaps"):
            self.plan()

    def test_changed_input_during_fit_invalidates_result(self):
        self.ready_fixture()
        plan = self.plan()
        weights = self.weights
        class ChangingModel:
            def __init__(self, _): pass
            def train(self, **kwargs):
                weights.write_bytes(b"unexpected mutation")
                best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
                best.parent.mkdir(parents=True)
                best.write_bytes(b"fixture")
        output = self.root / "invalid_candidate"
        with self.assertRaisesRegex(ValueError, "input changed"):
            experiments.run_experiment(plan, "data_only640", output, model_factory=ChangingModel)
        result = json.loads((output / "status.json").read_text())
        self.assertFalse(result["completed"])
        self.assertFalse(result["accepted_accuracy"])

    def test_separate_candidate_training_records_checkpoint_without_deployment(self):
        self.ready_fixture()
        plan = self.plan()
        original_hash = experiments.sha(self.weights)
        calls = []
        class FakeModel:
            def __init__(self, weights):
                calls.append({"weights": weights})
            def train(self, **kwargs):
                calls.append(kwargs)
                best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
                best.parent.mkdir(parents=True)
                best.write_bytes(b"test candidate, not neural-network evidence")
        output = self.root / "candidate"
        status = experiments.run_experiment(plan, "unfreeze640", output, model_factory=FakeModel)
        self.assertTrue(status["completed"])
        self.assertFalse(status["deployed"])
        self.assertFalse(status["accepted_accuracy"])
        self.assertEqual(calls[1]["freeze"], 0)
        self.assertNotEqual(calls[1]["data"], plan["data_yaml"])
        self.assertTrue(Path(calls[1]["data"]).is_relative_to(output))
        self.assertFalse(status["private_training_snapshot"]["test_copied"])
        self.assertNotIn("test", calls[1])
        self.assertEqual(experiments.sha(self.weights), original_hash)
        self.assertTrue((output / "plan.json").is_file())
        self.assertEqual(status["best_sha256"], experiments.sha(status["best_checkpoint"]))
        with self.assertRaises(FileExistsError):
            experiments.run_experiment(plan, "unfreeze640", output, model_factory=FakeModel)

    def test_stale_label_cache_is_not_inherited_and_source_is_not_training_path(self):
        self.ready_fixture()
        # Cache intentionally isn't an audited annotation or copied input.
        stale = self.prepared / "ball" / "labels" / "train.cache"
        stale.write_bytes(b"stale equal-length labels must not override reviewed targets")
        plan = self.plan()
        output = self.root / "isolated_candidate"
        output.mkdir()
        snapshot = experiments.private_fit_data(plan, output)
        self.assertFalse(list((output / "fit_data").rglob("*.cache")))
        self.assertFalse((output / "fit_data" / "test.txt").exists())
        self.assertFalse((output / "fit_data" / "images" / "test").exists())
        self.assertFalse(snapshot["test_copied"])
        for private, original in snapshot["source_for_private_copy"].items():
            self.assertTrue(Path(private).is_relative_to(output))
            self.assertFalse(Path(original).is_relative_to(output))
            self.assertEqual(experiments.sha(private), experiments.sha(original))
        experiments.verify_private_fit_data(snapshot)
        self.assertTrue(stale.exists())

    def test_private_label_mutation_during_training_rejects_candidate(self):
        self.ready_fixture()
        plan = self.plan()
        class CorruptingModel:
            def __init__(self, _): pass
            def train(self, **kwargs):
                root = Path(kwargs["data"]).parent
                label = next((root / "labels" / "train").glob("*.txt"))
                label.write_text("0 .4 .5 .1 .1", encoding="utf-8")
                best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
                best.parent.mkdir(parents=True)
                best.write_bytes(b"fixture")
        output = self.root / "invalid_private_candidate"
        with self.assertRaisesRegex(ValueError, "Private training input changed"):
            experiments.run_experiment(plan, "data_only640", output, model_factory=CorruptingModel)
        self.assertFalse(json.loads((output / "status.json").read_text())["completed"])
        experiments.validate_saved_plan(plan)

    def test_failure_logged_and_exception_not_swallowed(self):
        self.ready_fixture()
        plan = self.plan()
        class FailingModel:
            def __init__(self, _): pass
            def train(self, **_): raise RuntimeError("fixture CUDA error")
        output = self.root / "failed_candidate"
        with self.assertRaisesRegex(RuntimeError, "fixture CUDA error"):
            experiments.run_experiment(plan, "data_only640", output, model_factory=FailingModel)
        result = json.loads((output / "status.json").read_text())
        self.assertFalse(result["completed"])
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertFalse(result["deployed"])

    def test_model_data_output_separation_and_plan_write_exclusive(self):
        self.ready_fixture()
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "separate"):
            experiments.run_experiment(plan, "data_only640", self.prepared / "candidate", model_factory=lambda _: self.fail("Must not start"))
        path = self.root / "snapshot.json"
        experiments.save_plan(plan, path)
        with self.assertRaises(FileExistsError):
            experiments.save_plan(plan, path)


if __name__ == "__main__":
    unittest.main()
