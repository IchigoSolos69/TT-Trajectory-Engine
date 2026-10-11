"""Exact-input native/PyTorch YOLO diagnostics, never a training/deployment job.

Both native preprocessing paths are captured, including their actual CHW input.
Candidate oracles use references ONLY after inference and cannot select crops.
Outputs and inputs are hash-checked; source images/models are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "artifacts/python_deps").is_dir():
    sys.path.insert(0, str(ROOT / "artifacts/python_deps"))
import cv2
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_input(bgr, shape):
    _, channels, height, width = shape
    if channels != 3:
        raise ValueError("Expected RGB model")
    scale = min(width/bgr.shape[1], height/bgr.shape[0])
    # C++ round is half-away-from-zero for positive dimensions.
    rw, rh = int(np.floor(bgr.shape[1]*scale+.5)), int(np.floor(bgr.shape[0]*scale+.5))
    left, top = (width-rw)//2, (height-rh)//2
    image = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, height-rh-top, left, width-rw-left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None]/255


def candidates(raw, contract, threshold=.1, nms=.45):
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 5 or not np.isfinite(raw).all():
        raise ValueError("Expected finite one-class [1,5,N] output")
    values, mapping = raw[0], contract["letterbox"]
    scale, left, top = mapping["scale"], mapping["pad_left"], mapping["pad_top"]
    width, height = contract["source_size"]
    selected = np.flatnonzero((values[4] > threshold) & (values[2] > 0) & (values[3] > 0))
    boxes, entries = [], []
    for index in selected:
        cx, cy = (float(values[0, index])-left)/scale, (float(values[1, index])-top)/scale
        bw, bh = float(values[2, index])/scale, float(values[3, index])/scale
        box = [cx-bw/2, cy-bh/2, bw, bh]
        boxes.append(box)
        entries.append(dict(anchor_index=int(index), center_px=[cx, cy], box_px=box,
                            score=float(values[4, index]),
                            center_inside_source=0 <= cx < width and 0 <= cy < height))
    kept = cv2.dnn.NMSBoxes(boxes, [e["score"] for e in entries], threshold, nms) if entries else []
    # Native TensorRTEngine performs NMS before candidate normalization rejects
    # centers outside the source. Such a proposal may suppress an in-image box.
    after = sorted([entries[int(i)] for i in np.asarray(kept).flatten()
                    if entries[int(i)]["center_inside_source"]], key=lambda e: e["score"], reverse=True)
    return sorted(entries, key=lambda e: e["score"], reverse=True), after


def raw_parity_metrics(native, exact, scale):
    if (exact.shape != native.shape or exact.ndim != 3 or exact.shape[0] != 1
            or exact.shape[1] != 5 or not np.isfinite(exact).all()
            or not np.isfinite(native).all() or not np.isfinite(scale) or scale <= 0):
        raise ValueError("PyTorch/native raw-output shape/nonfinite mismatch")
    support = (native[0, 4] > .1) | (exact[0, 4] > .1)
    errors = np.linalg.norm(native[0, :2, support]-exact[0, :2, support], axis=1)/scale
    p95 = float(np.percentile(errors, 95)) if errors.size else None
    score_delta = float(np.abs(native[:, 4]-exact[:, 4]).max())
    coordinate_pass = p95 <= 1 if p95 is not None else None
    score_pass = score_delta <= .02
    return dict(exact_input_p95_center_difference_source_px=p95,
                exact_input_max_score_difference=score_delta,
                exact_input_support_anchor_count=int(support.sum()),
                coordinate_parity_gate_passed=coordinate_pass,
                score_parity_gate_passed=score_pass,
                parity_gate_passed=False if not score_pass else coordinate_pass,
                parity_gate_reason=("Raw model score disagreement exceeds 0.02" if not score_pass
                    else "No eligible anchors for coordinate comparison; coordinate parity unavailable" if p95 is None else None))


def candidate_diagnostic(raw, contract, reference):
    before, after = candidates(raw, contract)
    _, accepted = candidates(raw, contract, .25)
    result = {"proposal_threshold": .1, "accepted_threshold": .25, "nms_iou": .45,
              "pre_nms_candidates": before, "post_nms_candidates": after,
              "selected": accepted[0] if accepted else None, "reference": reference,
              "postprocess_policy": "Native proposal thresholds and NMS, followed by source-center validity filtering",
              "oracle_scope": "Diagnostic only; ground truth unavailable at inference"}
    if reference is None:
        result["failure_reason"] = "REFERENCE_UNKNOWN"
    elif reference["visibility"] == "absent":
        result["failure_reason"] = "ABSENT_FALSE_POSITIVE" if accepted else "CORRECT_ABSTENTION"
    else:
        point = np.asarray(reference["center_px"])
        def distance(e):
            return float(np.linalg.norm(np.asarray(e["center_px"])-point))
        eligible_before = [e for e in before if e["center_inside_source"]]
        result["nearest_pre_nms_source_px"] = min(map(distance, eligible_before), default=None)
        result["nearest_post_nms_source_px"] = min(map(distance, after), default=None)
        result["selected_error_source_px"] = distance(accepted[0]) if accepted else None
        for tolerance in (4, 8, 15):
            result[f"oracle_pre_nms_within_{tolerance}px"] = any(distance(e) < tolerance for e in eligible_before)
            result[f"oracle_post_nms_within_{tolerance}px"] = any(distance(e) < tolerance for e in after)
        if accepted and distance(accepted[0]) < 4:
            reason = "CORRECT_TOP_SELECTION"
        elif result["oracle_post_nms_within_4px"]:
            reason = "RANKING_OR_ACCEPTANCE_THRESHOLD"
        elif result["oracle_pre_nms_within_4px"]:
            reason = "NMS_SUPPRESSION"
        elif before:
            reason = "LOCALIZATION_OR_MISSING_PROPOSAL"
        else:
            reason = "NO_ELIGIBLE_PROPOSAL"
        result["failure_reason"] = reason
    return result


def prepare_cases(output):
    output.mkdir(parents=True, exist_ok=False)
    video = ROOT/"artifacts/blurball_research/official_22_003/source.mp4"
    labels = ROOT/"artifacts/blurball_research/official_22_003/labels.csv"
    references = {int(row["Frame"])+1: row for row in csv.DictReader(labels.open(encoding="utf-8"))}
    requested = {1, 3, 40, 80, 120, 160, 200, 240, 270, 272}
    capture, rows = cv2.VideoCapture(str(video)), []
    if not capture.isOpened():
        raise RuntimeError("Official diagnostic source unavailable")
    video_hash = sha(video)
    try:
        for fid in range(1, max(requested)+1):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Decode ended before {fid}")
            if fid not in requested:
                continue
            path = output/f"official_{fid}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("Cannot write isolated source frame")
            label = references[fid]
            reference = dict(visibility="visible", center_px=[float(label["X"]), float(label["Y"])]) if label["Visibility"] == "1" else dict(visibility="absent", center_px=None)
            rows.append(dict(image=str(path.resolve()), image_sha256=sha(path), domain="match", source_id=video_hash,
                             frame_id=fid, timestamp_s=capture.get(cv2.CAP_PROP_POS_MSEC)/1000,
                             reference=reference, reference_method="Published CSV source pixels", label_sha256=sha(labels)))
    finally:
        capture.release()
    if sha(video) != video_hash:
        raise ValueError("Source changed during decode")
    for clip, files in (("phone_e23544b0c4c1a850_1", [f"frame_{i}.jpg" for i in range(16, 21)]),
                        ("phone_452627e98a3c2652_1", [f"frame_{i}.jpg" for i in range(1, 4)]),
                        ("phone_c343f0cdb72df492_3", ["frame_6.jpg"])):
        folder = ROOT/"training_data/live"/clip
        annotations = json.loads((folder/"annotations.json").read_text())
        metadata = json.loads((folder/"capture.json").read_text())
        frames = {r["file"]: r for r in map(json.loads, (folder/"frames.jsonl").read_text().splitlines())}
        for file in files:
            path, annotation, frame = folder/file, annotations[file], frames[file]
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Cannot decode review image")
            geometry, reference, method = annotation.get("geometry", {}), None, "Unknown; never a screenshot hint"
            if annotation.get("reviewed_classes", {}).get("ball") is True:
                if geometry.get("visibility") == "absent" and not annotation.get("ball"):
                    reference, method = dict(visibility="absent", center_px=None), "Explicit saved absence review"
                elif geometry.get("visibility") in ("visible", "partial"):
                    streak = geometry.get("blur_streak") or {}
                    if streak.get("reviewed") is True:
                        point = streak.get("center")
                        if point is not None:
                            reference = dict(visibility="visible", center_px=[point[0]*image.shape[1], point[1]*image.shape[0]])
                            method = "Explicit reviewed exposure-center point/streak"
            rows.append(dict(image=str(path.resolve()), image_sha256=sha(path), domain="iphone", source_id=metadata["original_sha256"],
                             frame_id=frame["source_frame"], timestamp_s=frame["elapsed_ms"]/1000,
                             reference=reference, reference_method=method, label_sha256=sha(folder/"annotations.json")))
    manifest = {"schema_version": 1, "role": "development", "cases": rows,
                "pc_camera_reason": "No new reviewed PC-camera session supplied; not fabricated or silently substituted"}
    (output/"cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output/"cases.json"


def run(args):
    import torch
    from ultralytics import YOLO
    manifest = json.loads(args.cases.read_text(encoding="utf-8"))
    if manifest.get("role") != "development":
        raise ValueError("This exploratory diagnostic cannot consume sealed test cases")
    args.output.mkdir(parents=True, exist_ok=False)
    dependencies = {str(p.resolve()): sha(p) for p in (args.cases, args.weights, args.engine, args.native, Path(__file__))}
    for case in manifest["cases"]:
        if sha(case["image"]) != case["image_sha256"]:
            raise ValueError("Diagnostic source image changed")
        dependencies[case["image"]] = case["image_sha256"]
    model = YOLO(str(args.weights)).model.eval().float().to(args.device)
    results, env = [], dict(os.environ)
    env["PATH"] = "C:\\TensorRT\\bin;C:\\opencv\\build\\x64\\vc16\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.4\\bin\\x64;"+env.get("PATH", "")
    for index, case in enumerate(manifest["cases"]):
        print(f"Diagnostic {index+1}/{len(manifest['cases'])}: {Path(case['image']).name}", flush=True)
        for mode in ("cpu_linear", "cuda_nearest"):
            folder = args.output/f"case_{index+1}_{mode}"
            command = [str(args.native), "--engine", str(args.engine), "--image", case["image"], "--output", str(folder)]
            if mode == "cuda_nearest":
                command.append("--cuda-preprocess")
            completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (args.output/f"case_{index+1}_{mode}.log").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"Native diagnostic failed; full log: {folder.with_suffix('.log')}\n{completed.stdout}")
            contract = json.loads((folder/"contract.json").read_text())
            for key in ("input", "output"):
                if sha(folder/contract[key+"_file"]) != contract[key+"_sha256"]:
                    raise ValueError("Native tensor hash mismatch")
            if (contract["engine_sha256"] != dependencies[str(args.engine.resolve())]
                    or contract["native_binary_sha256"] != dependencies[str(args.native.resolve())]
                    or contract["source_image_sha256"] != case["image_sha256"]):
                raise ValueError("Native model/binary/image identity mismatch")
            width, height = contract["source_size"]
            bgr = np.fromfile(folder/contract["decoded_bgr_file"], dtype=np.uint8).reshape(height, width, 3)
            if sha(folder/contract["decoded_bgr_file"]) != contract["decoded_bgr_sha256"]:
                raise ValueError("Native decoded pixel hash mismatch")
            actual = np.fromfile(folder/contract["input_file"], dtype="<f4").reshape(contract["input_shape"])
            native = np.fromfile(folder/contract["output_file"], dtype="<f4").reshape(contract["output_shape"])
            canonical = canonical_input(bgr, contract["input_shape"])
            with torch.inference_mode():
                raw = model(torch.from_numpy(actual.copy()).to(args.device))
                exact = (raw[0] if isinstance(raw, (tuple, list)) else raw).detach().cpu().numpy()
                raw = model(torch.from_numpy(canonical).to(args.device))
                standard = (raw[0] if isinstance(raw, (tuple, list)) else raw).detach().cpu().numpy()
            parity = raw_parity_metrics(native, exact, contract["letterbox"]["scale"])
            input_difference = np.abs(actual-canonical)
            if mode == "cpu_linear" and float(input_difference.max()) > 1e-6:
                raise ValueError("CPU preprocessing does not match independent RGB/linear/letterbox reference")
            result = dict(case=case, mode=mode, native_contract=str(folder/"contract.json"),
                          input_vs_standard_max_abs=float(input_difference.max()),
                          input_vs_standard_fraction_changed=float(np.mean(input_difference > 1e-6)),
                          **parity,
                          native=candidate_diagnostic(native, contract, case.get("reference")),
                          pytorch_same_input=candidate_diagnostic(exact, contract, case.get("reference")),
                          pytorch_standard_linear_input=candidate_diagnostic(standard, contract, case.get("reference")))
            results.append(result)
    for path, digest in dependencies.items():
        if sha(path) != digest:
            raise ValueError("Diagnostic input/model/source changed during evaluation")
    report = dict(schema_version=1, scope="Fixed development cases; offline diagnostics, not blind accuracy or live latency",
                  cases=len(manifest["cases"]), results=results, input_sha256=dependencies,
                  inputs_reverified=True, promotion_approved=False,
                  pc_camera_reason=manifest.get("pc_camera_reason"),
                  exact_input_parity_evaluable_cases=sum(r["parity_gate_passed"] is not None for r in results),
                  exact_input_parity_all_passed=any(r["parity_gate_passed"] is not None for r in results)
                      and all(r["parity_gate_passed"] is True for r in results if r["parity_gate_passed"] is not None))
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "inputs_reverified", "exact_input_parity_all_passed", "promotion_approved")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("output", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("cases", type=Path)
    execute.add_argument("output", type=Path)
    execute.add_argument("--weights", type=Path, default=ROOT/"artifacts/multigame/candidate/weights/best.pt")
    execute.add_argument("--engine", type=Path, default=ROOT/"models/yolov11_ttgames.engine")
    execute.add_argument("--native", type=Path, default=ROOT/"backend_cpp/build/Release/dump_detector_raw.exe")
    execute.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_cases(args.output))
    else:
        run(args)


if __name__ == "__main__":
    main()
