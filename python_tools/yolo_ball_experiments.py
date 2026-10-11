"""Plan reproducible ball-only YOLO experiments; fitting is an explicit command.

Plans may be written while data is not ready. Production execution requires
unchanged reviewed train/validation captures, explicit paired-session metadata
and a checked prepare_yolo_paired_training.py export. Sealed C test contents are
never opened for planning or fitting. This module never exports or deploys a model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
from yolo_training_policy import BATCH_CHOICES, FINAL_SEEDS, CAMERA_AUGMENTATION, center_square_labels

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
BASE = dict(epochs=12, imgsz=640, batch=16, device=0, workers=1,
            optimizer="AdamW", lr0=.0001, lrf=.1, freeze=10, cos_lr=True,
            mosaic=.1, close_mosaic=3, scale=.3, translate=.08,
            hsv_s=.5, hsv_v=.5, seed=42, deterministic=True, patience=0,
            amp=False, cache=False, plots=False)
VARIANTS = {
    "data_only640": {},
    "unfreeze640": {"freeze": 0},
    "high_detail960": {"imgsz": 960},
    "extent640": {},
    "camera_aug640": {"mosaic": 0., "close_mosaic": 0, "mixup": 0., "copy_paste": 0., "scale": 0., "translate": 0.,
                      "hsv_h": 0., "hsv_s": 0., "hsv_v": 0., "fliplr": 0., "flipud": 0.},
    **{f"final640_seed{seed}": {"epochs": 60, "patience": 12, "seed": seed} for seed in FINAL_SEEDS},
    **{f"final_seed{seed}": {"epochs": 60, "patience": 12, "seed": seed} for seed in FINAL_SEEDS},
}
STAGES = {"data_only640": "A", "unfreeze640": "B", "high_detail960": "C", "extent640": "D", "camera_aug640": "E"}
CODE_FILES = ("yolo_ball_experiments.py", "audit_training_readiness.py",
              "prepare_live_training.py", "review_absence_provenance.py",
              "export_reviewed_streaks.py", "yolo_training_policy.py", "yolo_paired_data.py", "yolo_ultralytics_hooks.py",
              "prepare_yolo_paired_training.py", "recording_source.py",
              "prepare_absence_interval.py", "audit_absence_intervals.py")
RUNTIME_DISTRIBUTIONS = ("ultralytics","torch","numpy","opencv-python","opencv-python-headless","Pillow")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_recipe(recipe, *, final=False):
    if recipe is None:
        return None
    required={"freeze","geometry","augmentation"}
    if (set(recipe) not in (required,required|{"imgsz"})
            or type(recipe["freeze"]) is not int or recipe["freeze"] not in (0,10)
            or recipe["geometry"] not in ("center_square30","reviewed_extent")
            or recipe["augmentation"] not in ("common_preset","camera_safe")
            or type(recipe.get("imgsz",640)) is not int
            or recipe.get("imgsz",640) not in ((640,960) if final else (640,))):
        raise ValueError("Selected recipe requires tested freeze/geometry/augmentation and valid 640/960 final input")
    return {**recipe,"imgsz":recipe.get("imgsz",640)}


def variant_config(name, batch=16, best640=None, selected_recipe=None):
    if name not in VARIANTS:
        raise ValueError("Unknown fixed experiment variant")
    config = copy.deepcopy(BASE)
    config.update(VARIANTS[name])
    if batch not in BATCH_CHOICES:
        raise ValueError("Common comparison batch must be 16, 8 or 4")
    config["batch"] = batch
    best640=checked_recipe(best640)
    selected_recipe=checked_recipe(selected_recipe,final=True)
    recipe=(selected_recipe or best640) if name.startswith("final") else best640
    if recipe and name not in ("data_only640", "unfreeze640", "high_detail960"):
        config["freeze"] = recipe["freeze"]
        if name.startswith("final"):
            config["imgsz"] = recipe["imgsz"]
        if name.startswith("final") and recipe["augmentation"] == "camera_safe":
            config.update(VARIANTS["camera_aug640"])
    return config


def variant_metadata(name, batch=16, best640=None, selected_recipe=None):
    geometry = "reviewed_extent" if name == "extent640" else "center_square30"
    augmentation = "camera_safe" if name == "camera_aug640" else "common_preset"
    recipe=(selected_recipe or best640) if name.startswith("final") else best640
    if recipe and (name == "camera_aug640" or name.startswith("final")):
        geometry = recipe["geometry"]
    if recipe and name.startswith("final"):
        augmentation = recipe["augmentation"]
    return dict(stage=STAGES.get(name,"FINAL"),training=variant_config(name,batch,best640,selected_recipe),
                geometry=geometry,augmentation=augmentation,
                requires_selected_best640=name in ("extent640","camera_aug640"),
                requires_selected_final_recipe=name.startswith("final"),
                native_current_shape_compatible=True,
                native_gate="Shape-aware static FP32 parser supports 640/960; candidate export/native coordinate parity and live timing remain required")


def remember(dependencies, path, expected=None):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed while hashing")
    if expected is not None and digest != expected:
        raise ValueError(f"Input content differs from provenance: {path}")
    dependencies[path.as_posix()] = digest
    return digest


def runtime_policy(dependencies):
    """Freeze installed package versions/defaults without importing models."""
    versions={}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    default_path,default_hash=None,None
    if versions["ultralytics"] is not None:
        candidate=Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics/cfg/default.yaml")).resolve()
        if candidate.is_file():
            default_path=candidate.as_posix()
            default_hash=remember(dependencies,candidate)
    return dict(distribution_versions=versions,ultralytics_defaults_path=default_path,
                ultralytics_defaults_sha256=default_hash,
                implicit_defaults_policy="FROZEN_INSTALLED_DEFAULT_YAML_AND_EXPLICIT_VARIANT_OVERRIDES",
                heavy_packages_imported=False,automatic_upgrade=False)


def label_text(labels):
    return "\n".join(f"0 {x+w/2:.8f} {y+h/2:.8f} {w:.8f} {h:.8f}"
                     for x, y, w, h in labels["ball"])


def read_list(path, dependencies):
    remember(dependencies, path)
    entries = Path(path).read_text(encoding="utf-8").splitlines()
    if not entries or any(not line or not Path(line).is_absolute() for line in entries):
        raise ValueError("Prepared lists require nonempty absolute source paths")
    return [Path(line).resolve() for line in entries]


def inspect_prepared(prepared, captures, split_file, dependencies, match_root):
    """Verify prepared copies/labels/lists against saved human review, not flags."""
    from prepare_live_training import load_reviewed
    from PIL import Image
    prepared = Path(prepared).resolve()
    remember(dependencies, prepared / "audit.json")
    declared = json.loads((prepared / "audit.json").read_text(encoding="utf-8"))
    splits = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if (declared.get("schema_version") != 2 or declared.get("splits") != splits
            or declared.get("contains_test_in_training") is not False
            or "ball" not in declared.get("prepared_kinds", [])):
        raise ValueError("Prepared audit must match frozen ball split provenance")
    rows, _ = load_reviewed(Path(captures), splits, ("ball",))
    expected_frames = []
    image_splits = {}
    def check_image_partition(image, split):
        with Image.open(image) as source:
            rgb = source.convert("RGB")
            decoded = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if decoded in image_splits and image_splits[decoded] != split:
            raise ValueError("Decoded match/live image overlaps partitions")
        image_splits[decoded] = split
    for split in SPLITS:
        if len(rows[split]) < 30 or sum(bool(r[1]["ball"]) for r in rows[split]) < 5 or sum(not r[1]["ball"] for r in rows[split]) < 5:
            raise ValueError("Every independent split needs 30 reviewed ball examples, including five positive and five absent")
        expected_paths = []
        for image, labels, digest, clip in rows[split]:
            copy_path = prepared / "ball" / "images" / split / f"{clip}_{image.name}"
            remember(dependencies, copy_path, digest)
            check_image_partition(copy_path, split)
            label = prepared / "ball" / "labels" / split / f"{copy_path.stem}.txt"
            remember(dependencies, label)
            if label.read_text(encoding="utf-8") != label_text(labels):
                raise ValueError("Prepared target differs from reviewed ball annotation")
            expected_paths.append(copy_path.resolve())
            expected_frames.append(dict(kind="ball", split=split, clip=clip, file=image.name,
                                        sha256=digest, objects={"ball": len(labels["ball"])}))
        if read_list(prepared / "ball" / f"live_{split}.txt", dependencies) != expected_paths:
            raise ValueError("Prepared live list membership/order differs from frozen review")
        if split in ("train", "val"):
            original = read_list(Path(match_root) / f"{split}.txt", dependencies)
            if any("test_" in p.name or not p.is_file() for p in original):
                raise ValueError("Match list includes test sequence or missing source")
            selected = original
            if split == "train":
                count = min(len(original), 3 * len(expected_paths))
                selected = [original[i * len(original) // count] for i in range(count)]
            if read_list(prepared / "ball" / f"{split}.txt", dependencies) != selected + expected_paths:
                raise ValueError("Merged match/live membership differs from preparation policy")
            for image in selected:
                remember(dependencies, image)
                check_image_partition(image, split)
                label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
                remember(dependencies, label)
                for line in label.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 5 or fields[0] != "0":
                        raise ValueError("Match targets must be ball-only YOLO labels")
                    values = [float(value) for value in fields[1:]]
                    if any(not math.isfinite(value) for value in values) or not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                        raise ValueError("Invalid normalized match ball target")
            # Chronological match splits are regression/development data. Their
            # shared recording sessions are not advertised as new blind tests.
    actual_frames = [row for row in declared.get("frames", []) if row.get("kind") == "ball"]
    if actual_frames != expected_frames:
        raise ValueError("Prepared audit frame records differ from reviewed export")
    actual_counts = declared.get("reviewed_counts_by_kind", {}).get("ball")
    if actual_counts != {split: len(rows[split]) for split in SPLITS}:
        raise ValueError("Prepared audit counts differ from saved review")
    config_path = prepared / "ball" / "data.yaml"
    remember(dependencies, config_path)
    expected_yaml = (f"train: {(prepared/'ball'/'train.txt').as_posix()}\n"
                     f"val: {(prepared/'ball'/'val.txt').as_posix()}\n"
                     "names:\n  0: ball\n")
    if config_path.read_text(encoding="utf-8") != expected_yaml:
        raise ValueError("Unexpected YAML; only checked train/val lists and ball class are allowed")
    return config_path.as_posix()


def _legacy_fixture_plan(captures, splits=None, prepared=None, weights=None, *, match_root=None):
    """Read-only plan. Missing partitions are blockers, never inferred splits."""
    from audit_training_readiness import audit
    captures = Path(captures).resolve()
    weights = Path(weights or ROOT / "artifacts/multigame/candidate/weights/best.pt").resolve()
    match_root = Path(match_root or ROOT / "artifacts/multigame").resolve()
    readiness = audit(captures, splits)
    dependencies = {}
    for name in CODE_FILES:
        remember(dependencies, Path(__file__).parent / name)
    if weights.is_file():
        remember(dependencies, weights)
    blockers = list(readiness["ball"]["reason_codes"])
    if not weights.is_file():
        blockers.append("INITIAL_WEIGHTS_MISSING")
    if splits is not None:
        remember(dependencies, splits, readiness["splits_sha256"])
    for lineage in readiness["capture_lineage"]:
        folder = captures / lineage["clip"]
        for filename, field in (("capture.json", "capture_sha256"), ("annotations.json", "annotations_sha256"), ("frames.jsonl", "index_sha256")):
            remember(dependencies, folder / filename, lineage[field])
        capture = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
        if capture.get("source") == "recorded_camera":
            # audit() just hashes each original recording and checks this value.
            dependencies[Path(capture["original_path"]).resolve().as_posix()] = capture["original_sha256"]
        for line in (folder / "frames.jsonl").read_text(encoding="utf-8").splitlines():
            remember(dependencies, folder / json.loads(line)["file"])
        history = folder / "annotation_history"
        if history.is_dir():
            for file in sorted(history.rglob("*.json")):
                remember(dependencies, file)
    data_yaml = None
    if prepared is None:
        blockers.append("CHECKED_PREPARED_BALL_DATASET_MISSING")
    elif splits is None:
        blockers.append("PREPARED_DATA_REQUIRES_FROZEN_SPLITS")
    elif not blockers:
        data_yaml = inspect_prepared(prepared, captures, splits, dependencies, match_root)
    plan = dict(schema_version=1, scope="Experimental training only; accuracy/promotion require separate held-out and native evaluation",
                inputs=dict(captures=captures.as_posix(), splits=Path(splits).resolve().as_posix() if splits else None,
                            prepared=Path(prepared).resolve().as_posix() if prepared else None,
                            weights=weights.as_posix(), match_root=match_root.as_posix()),
                readiness=readiness, ready_for_fitting=not blockers, blockers=sorted(set(blockers)),
                variants={name: dict(training=variant_config(name),
                          native_current_shape_compatible=name != "high_detail960",
                          native_gate="Export/TensorRT/C++ parity required; 960 additionally requires parser/buffer support") for name in VARIANTS},
                data_yaml=data_yaml, dependencies=dict(sorted(dependencies.items())),
                weights_sha256=dependencies.get(weights.as_posix()),
                configuration_sha256=hashlib.sha256(canonical({name: variant_config(name) for name in VARIANTS}).encode()).hexdigest(),
                input_snapshot_sha256=hashlib.sha256(canonical(dependencies).encode()).hexdigest(),
                guarantees=dict(automatic_deployment=False, test_used_by_optimizer=False,
                                unknown_labels_are_negatives=False, teacher_suggestions_are_ground_truth=False))
    plan["plan_sha256"] = hashlib.sha256(canonical(plan).encode()).hexdigest()
    return plan


def build_plan(captures, splits=None, prepared=None, weights=None, *, match_root=None,
               registry=None, batch=16, best640=None, selected_recipe=None, fixture_only=False):
    """Production plans never consume sealed test C annotations or image bytes."""
    if fixture_only:
        plan = _legacy_fixture_plan(captures,splits,prepared,weights,match_root=match_root)
        plan["fixture_only"] = True
        plan["guarantees"]["full_candidate_training_allowed"] = False
        plan.pop("plan_sha256",None)
        plan["plan_sha256"] = hashlib.sha256(canonical(plan).encode()).hexdigest()
        return plan
    from yolo_paired_data import inspect_training_sources
    captures=Path(captures).resolve()
    weights=Path(weights or ROOT/"artifacts/multigame/candidate/weights/best.pt").resolve()
    match_root=Path(match_root or ROOT/"artifacts/multigame").resolve()
    registry=Path(registry or captures/"training_sessions.json").resolve()
    best640=checked_recipe(best640)
    selected_recipe=checked_recipe(selected_recipe,final=True)
    variants={name:variant_metadata(name,batch,best640,selected_recipe) for name in VARIANTS}
    dependencies={}
    for name in CODE_FILES: remember(dependencies,Path(__file__).parent/name)
    metric_file=Path(__file__).parent/"ball_release_metrics.py"
    if metric_file.is_file(): remember(dependencies,metric_file)
    if weights.is_file(): remember(dependencies,weights)
    runtime=runtime_policy(dependencies)
    report=inspect_training_sources(captures,registry,match_root,lambda path,expected=None:remember(dependencies,path,expected))
    blockers=list(report["blockers"])
    if not weights.is_file(): blockers.append("INITIAL_WEIGHTS_MISSING")
    if not metric_file.is_file(): blockers.append("BALL_RELEASE_METRIC_IMPLEMENTATION_MISSING")
    if runtime["ultralytics_defaults_sha256"] is None: blockers.append("ULTRALYTICS_INSTALLED_DEFAULTS_MISSING")
    data_yaml=None
    source_records=report.pop("records")
    records=[r for role in ("train","val") for r in source_records if r["role"]==role]
    if prepared is None:
        blockers.append("CHECKED_PAIRED_TRAINING_EXPORT_MISSING")
    else:
        prepared=Path(prepared).resolve()
        metadata=prepared/"training_metadata.json"
        if not metadata.is_file(): blockers.append("PAIRED_PREPARED_METADATA_MISSING")
        else:
            remember(dependencies,metadata)
            declared=json.loads(metadata.read_text(encoding="utf-8"))
            actual=declared.get("records",[])
            if declared.get("schema_version") != 1 or declared.get("test_exported") is not False or len(actual)!=len(records):
                raise ValueError("Prepared metadata must contain train/val only with exact reference membership")
            for expected,row in zip(records,actual):
                if {k:row.get(k) for k in expected} != expected:
                    raise ValueError("Prepared metadata differs from reviewed source provenance")
                image,label=Path(row["prepared_image"]),Path(row["prepared_label"])
                if not image.resolve().is_relative_to(prepared) or not label.resolve().is_relative_to(prepared):
                    raise ValueError("Prepared image/label escapes new export")
                remember(dependencies,image,expected["image_sha256"])
                remember(dependencies,label,row["prepared_label_sha256"])
                if label.read_text(encoding="utf-8") != label_text(expected["labels"]):
                    raise ValueError("Prepared label differs from reviewed annotation")
            for role in ("train","val"):
                listing=prepared/f"{role}.txt"
                expected=[Path(r["prepared_image"]).resolve() for r in actual if r["role"]==role]
                if read_list(listing,dependencies)!=expected: raise ValueError("Prepared list differs from exact domain references")
            data_yaml=prepared/"data.yaml"
            remember(dependencies,data_yaml)
            wanted=f"train: {(prepared/'train.txt').as_posix()}\nval: {(prepared/'val.txt').as_posix()}\nnames:\n  0: ball\n"
            if data_yaml.read_text(encoding="utf-8") != wanted: raise ValueError("Unchecked prepared YAML")
            data_yaml=data_yaml.as_posix()
    plan=dict(schema_version=2,scope="Paired-camera candidate fitting; C test labels sealed and unread",
        inputs=dict(captures=captures.as_posix(),splits=None,prepared=Path(prepared).as_posix() if prepared else None,
                    weights=weights.as_posix(),match_root=match_root.as_posix(),registry=registry.as_posix()),
        common_batch=batch,best640_recipe=best640,selected_final_recipe=selected_recipe,readiness=report,
        strict_data_blockers=sorted(set(blockers)),ready_for_fitting=not blockers,blockers=sorted(set(blockers)),
        variants=variants,runtime_policy=runtime,data_yaml=data_yaml,dependencies=dict(sorted(dependencies.items())),
        weights_sha256=dependencies.get(weights.as_posix()),
        configuration_sha256=hashlib.sha256(canonical(variants).encode()).hexdigest(),
        input_snapshot_sha256=hashlib.sha256(canonical(dependencies).encode()).hexdigest(),
        sampling_policy=dict(domain_shares=dict(match=.5,iphone=.25,pc_camera=.25),live_visible_absent_ratio=[2,1]),
        validation_policy=dict(selection="MINIMUM_DOMAIN_F1",domains=["match","iphone","pc_camera"],score_threshold=.25,tolerance1080_px=4),
        camera_augmentation=CAMERA_AUGMENTATION,memory_budget_bytes=7*1024**3,
        guarantees=dict(automatic_deployment=False,test_used_by_optimizer=False,test_labels_consumed=False,
                        unknown_labels_are_negatives=False,teacher_suggestions_are_ground_truth=False))
    plan["plan_sha256"]=hashlib.sha256(canonical(plan).encode()).hexdigest()
    return plan


def validate_saved_plan(plan):
    body = copy.deepcopy(plan)
    expected = body.pop("plan_sha256", None)
    if expected != hashlib.sha256(canonical(body).encode()).hexdigest():
        raise ValueError("Saved plan was changed")
    if plan.get("schema_version") == 2:
        if plan.get("runtime_policy") != runtime_policy({}):
            raise ValueError("Installed training runtime/defaults changed since the frozen plan")
        wanted={name:variant_metadata(name,plan["common_batch"],plan["best640_recipe"],plan["selected_final_recipe"]) for name in VARIANTS}
    else:
        wanted={
            name: dict(training=variant_config(name), native_current_shape_compatible=name != "high_detail960",
                       native_gate="Export/TensorRT/C++ parity required; 960 additionally requires parser/buffer support") for name in VARIANTS}
    if plan.get("schema_version") not in (1,2) or plan.get("variants") != wanted:
        raise ValueError("Saved plan configuration differs from this runner")
    for path, expected in plan["dependencies"].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f"Saved input changed or disappeared: {path}")
    return plan


def save_plan(plan, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")


def private_fit_data(plan, output, name="data_only640"):
    """Fresh train/val copies prevent stale Ultralytics caches or source repair.

    cache=False disables image caching, not reuse of label *.cache files. Never
    inherit those files or let the trainer write beside reviewed/baseline data.
    Test images/labels are verified by the plan but not copied for optimization.
    """
    root = Path(output) / "fit_data"
    root.mkdir()
    hashes, lineage, private_records = {}, {}, []
    paired=plan.get("schema_version")==2
    prepared=Path(plan["inputs"]["prepared"])
    metadata={}
    if paired:
        metadata={Path(r["prepared_image"]).resolve().as_posix():r for r in json.loads((prepared/"training_metadata.json").read_text(encoding="utf-8"))["records"]}
    else:
        prepared=prepared/"ball"
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        paths = []
        for index, text in enumerate((prepared / f"{split}.txt").read_text(encoding="utf-8").splitlines(), 1):
            image = Path(text).resolve()
            label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
            copied_image = images / f"frame_{index:08d}{image.suffix.lower()}"
            copied_label = labels / f"{copied_image.stem}.txt"
            for original, copied in ((image, copied_image), (label, copied_label)):
                expected = plan["dependencies"].get(original.as_posix())
                if expected is None:
                    raise ValueError("Private training input is not in the frozen plan")
                shutil.copyfile(original, copied)
                digest = sha(copied)
                if digest != expected:
                    raise ValueError("Source changed while copying private training data")
                hashes[copied.as_posix()] = digest
                lineage[copied.as_posix()] = original.as_posix()
            if paired:
                record=copy.deepcopy(metadata[image.as_posix()])
                record["private_image"]=copied_image.as_posix()
                record["private_label"]=copied_label.as_posix()
                if split=="train" and plan["variants"][name]["geometry"]=="center_square30":
                    centers=[record["annotation"]["center_px"]] if record["positive"] else []
                    copied_label.write_text(label_text(center_square_labels(record["labels"],record["source_size"],centers_px=centers)),encoding="utf-8")
                    hashes[copied_label.as_posix()]=sha(copied_label)
                private_records.append(record)
            paths.append(copied_image.as_posix())
        listing = root / f"{split}.txt"
        listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
        hashes[listing.as_posix()] = sha(listing)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(f"train: {(root/'train.txt').as_posix()}\n"
                         f"val: {(root/'val.txt').as_posix()}\nnames:\n  0: ball\n", encoding="utf-8")
    hashes[data_yaml.as_posix()] = sha(data_yaml)
    snapshot = dict(data_yaml=data_yaml.as_posix(), files_sha256=hashes,
                    source_for_private_copy=lineage, test_copied=False,
                    inherited_label_caches=False)
    if paired:
        domain_lists={}
        for domain in ("match","iphone","pc_camera"):
            listing=root/f"val_{domain}.txt"
            paths=[r["private_image"] for r in private_records if r["role"]=="val" and r["domain"]==domain]
            if not paths: raise ValueError("Every validation domain must have reviewed references")
            listing.write_text("\n".join(paths)+"\n",encoding="utf-8")
            hashes[listing.as_posix()]=sha(listing)
            domain_lists[domain]=listing.as_posix()
        snapshot.update(records=private_records,validation_domain_lists=domain_lists,
                        target_geometry=plan["variants"][name]["geometry"])
    save_plan(snapshot, root / "manifest.json")
    snapshot["manifest_sha256"] = sha(root / "manifest.json")
    return snapshot


def verify_private_fit_data(snapshot):
    manifest = Path(snapshot["data_yaml"]).parent / "manifest.json"
    if sha(manifest) != snapshot["manifest_sha256"]:
        raise ValueError("Private training manifest changed")
    for path, expected in snapshot["files_sha256"].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f"Private training input changed: {path}")


def run_experiment(plan, name, output, *, model_factory=None):
    """Execute a frozen plan. Factory injection exists for CPU control tests."""
    validate_saved_plan(plan)
    if name not in VARIANTS:
        raise ValueError("Unknown fixed experiment variant")
    if not plan["ready_for_fitting"] or plan["blockers"]:
        raise ValueError("Fitting blocked: " + ", ".join(plan["blockers"]))
    fixture=plan.get("fixture_only") is True
    if fixture and model_factory is None:
        raise ValueError("Fixture plans cannot train neural-network candidates")
    if plan.get("schema_version")==2:
        if plan["variants"][name]["requires_selected_best640"] and plan["best640_recipe"] is None:
            raise ValueError("D/E requires an explicitly selected passing 640 recipe")
        if plan["variants"][name]["requires_selected_final_recipe"] and not (plan["selected_final_recipe"] or plan["best640_recipe"]):
            raise ValueError("Final candidates require an explicit selected recipe")
        refreshed=build_plan(**plan["inputs"],batch=plan["common_batch"],best640=plan["best640_recipe"],selected_recipe=plan["selected_final_recipe"])
    else:
        refreshed=build_plan(**{key:plan["inputs"][key] for key in ("captures","splits","prepared","weights")},
                             match_root=plan["inputs"]["match_root"],fixture_only=fixture)
    if refreshed != plan:
        raise ValueError("Review/preparation/provenance changed since the frozen plan")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new experimental output directory")
    forbidden = [Path(plan["inputs"][key]).resolve() for key in ("captures", "prepared", "match_root")]
    forbidden += [ROOT / "models", Path(plan["inputs"]["weights"]).parent]
    if any(output == folder or output.is_relative_to(folder) for folder in forbidden):
        raise ValueError("Experimental output must be separate from data and deployed/baseline models")
    if model_factory is None:
        sys.path.insert(0, str(ROOT / "artifacts/python_deps"))
        from ultralytics import YOLO
        model_factory = YOLO
    output.mkdir(parents=True)
    status = dict(schema_version=1, plan_sha256=plan["plan_sha256"], variant=name,
                  training=plan["variants"][name]["training"], started=True, completed=False,
                  candidate_only=True, deployed=False, accepted_accuracy=False)
    try:
        status["ultralytics_version"] = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        status["ultralytics_version"] = None
    save_plan(plan, output / "plan.json")
    status_path = output / "status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    try:
        private = private_fit_data(plan, output,name)
        status["private_training_snapshot"] = private
        verify_private_fit_data(private)
        validate_saved_plan(plan)
        hooks={}
        if plan.get("schema_version")==2:
            from yolo_ultralytics_hooks import make_trainer
            hooks["trainer"]=make_trainer(private,plan["variants"][name],status["training"]["seed"],plan["common_batch"])
            status["hooks"]={"trainer":"PairedCameraTrainer","sampler":"BalancedDomainSampler",
                             "validation_selection":"MINIMUM_DOMAIN_F1","sealed_test_consumed":False}
        model_factory(plan["inputs"]["weights"]).train(
            data=private["data_yaml"], project=str(output), name="fit", exist_ok=False,
            **status["training"],**hooks)
        validate_saved_plan(plan)
        verify_private_fit_data(private)
        best = output / "fit" / "weights" / "best.pt"
        if not best.is_file():
            raise ValueError("Trainer did not produce a separate best checkpoint")
        status.update(completed=True, inputs_unchanged=True, best_checkpoint=best.as_posix(), best_sha256=sha(best))
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error))
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Write a read-only data/config/provenance snapshot")
    planning.add_argument("output", type=Path)
    planning.add_argument("--captures", type=Path, default=ROOT / "training_data/live")
    planning.add_argument("--splits", type=Path)
    planning.add_argument("--prepared", type=Path)
    planning.add_argument("--weights", type=Path)
    planning.add_argument("--registry",type=Path)
    planning.add_argument("--batch",type=int,choices=BATCH_CHOICES,default=16)
    planning.add_argument("--best640-recipe",type=Path)
    planning.add_argument("--selected-recipe",type=Path,help="Explicit final recipe; imgsz 640 or 960 (default 640)")
    execute = commands.add_parser("train", help="Explicitly fit one experimental candidate from an unchanged ready plan")
    execute.add_argument("plan", type=Path)
    execute.add_argument("variant", choices=tuple(VARIANTS))
    execute.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        if args.output.exists():
            raise FileExistsError("Choose a new plan file")
        result = build_plan(args.captures, args.splits, args.prepared, args.weights,
                            registry=args.registry,batch=args.batch,
                            best640=json.loads(args.best640_recipe.read_text(encoding="utf-8")) if args.best640_recipe else None,
                            selected_recipe=json.loads(args.selected_recipe.read_text(encoding="utf-8")) if args.selected_recipe else None)
        save_plan(result, args.output)
        print(json.dumps(dict(plan=args.output.resolve().as_posix(), plan_sha256=result["plan_sha256"],
                             ready_for_fitting=result["ready_for_fitting"], blockers=result["blockers"]), indent=2))
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        print(json.dumps(run_experiment(plan, args.variant, args.output), indent=2))


if __name__ == "__main__":
    main()
