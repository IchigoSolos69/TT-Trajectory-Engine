"""Ball-center evaluation and explicit release gates; predictions are not detections.

Rows retain original source pixels/PTS. The four-pixel release tolerance is
normalized to 1080 image rows; historical source-pixel metrics are unchanged.
Frame Wilson intervals are descriptive, not independent-session confidence.
Sparse review samples cannot establish false tracks/minute or absence duration.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DOMAINS = ("match", "iphone", "pc_camera")
SUBSETS = ("bright_background", "small_ball", "blur", "partial_occlusion")
MEASURED = {"measured", "reacquired", "recovered"}
STATES = MEASURED | {"predicted", "rejected", "lost"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _point(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(_number(v) and 0 <= v < n for v, n in zip(value, size)))


def validate_rows(rows):
    seen, sources = set(), {}
    for row in rows:
        source, fid = row.get("source_id"), row.get("frame_id")
        if not isinstance(source, str) or not source or type(fid) is not int or fid < 1:
            raise ValueError("Every row needs a source identity and positive frame ordinal")
        if (source, fid) in seen:
            raise ValueError("Duplicate source/frame would inflate evaluation")
        seen.add((source, fid))
        size, domain, timestamp = row.get("source_size"), row.get("domain"), row.get("timestamp_s")
        if (not isinstance(size, (list, tuple)) or len(size) != 2
                or any(type(n) is not int or n <= 0 for n in size)
                or domain not in DOMAINS or not _number(timestamp) or timestamp < 0):
            raise ValueError("Invalid source geometry, domain or timestamp")
        identity = (tuple(size), domain)
        if source in sources and sources[source] != identity:
            raise ValueError("Source geometry/domain changes require a new source identity")
        sources[source] = identity
        annotation = row.get("annotation", {})
        if annotation.get("visibility", "unknown") not in ("visible", "partial", "absent", "unknown"):
            raise ValueError("Unsupported visibility; hidden balls must remain unknown")
        if annotation.get("reviewed") is True:
            visibility = annotation.get("visibility", "unknown")
            if visibility in ("visible", "partial") and not _point(annotation.get("center_px"), size):
                raise ValueError("Reviewed visible targets require an in-image center")
            if visibility == "absent" and annotation.get("center_px") is not None:
                raise ValueError("An absent target cannot contain a ball center")
        tags = annotation.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("Annotation tags must be strings")
        prediction = row.get("prediction")
        if prediction is not None:
            if not isinstance(prediction, dict) or prediction.get("state") not in STATES:
                raise ValueError("Prediction needs an explicit observation state")
            if not _point(prediction.get("center_px"), size):
                raise ValueError("Prediction center must be finite and inside the source")
            score = prediction.get("score")
            if score is not None and (not _number(score) or not 0 <= score <= 1):
                raise ValueError("Invalid model score")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    for source_rows in grouped.values():
        ordered = sorted(source_rows, key=lambda row: row["frame_id"])
        if any(b["timestamp_s"] <= a["timestamp_s"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Source frame order disagrees with source timestamps")
    return rows


def _interval(successes, total):
    if not total:
        return None
    p, z = successes / total, 1.959963984540054
    scale = 1 + z*z / total
    mid = (p + z*z / (2*total)) / scale
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / scale
    return [max(0, mid-half), min(1, mid+half)]


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*fraction
    lower = int(index)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)


def _negative_coverage(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    seconds, episodes, absent_seen, incomplete, unknown_track = 0.0, 0, False, False, False
    for source_rows in grouped.values():
        ordered = sorted(source_rows, key=lambda row: row["frame_id"])
        previous_fp = None
        for index, row in enumerate(ordered):
            annotation, prediction = row.get("annotation", {}), row.get("prediction")
            absent = annotation.get("reviewed") is True and annotation.get("visibility") == "absent"
            if not absent:
                previous_fp = None
                continue
            absent_seen = True
            previous = ordered[index-1] if index else None
            following = ordered[index+1] if index+1 < len(ordered) else None
            if ((previous is not None and row["frame_id"] != previous["frame_id"]+1)
                    or (following is not None and following["frame_id"] != row["frame_id"]+1)):
                incomplete = True
            # Source PTS backs the duration. A supplied per-frame duration must
            # agree with neighboring source timestamps, not inflate a 60 FPS
            # sequence into e.g. six times the actual verified video time.
            observed = (following["timestamp_s"]-row["timestamp_s"]
                        if following is not None and following["frame_id"] == row["frame_id"]+1
                        else row["timestamp_s"]-previous["timestamp_s"]
                        if previous is not None and row["frame_id"] == previous["frame_id"]+1
                        else None)
            duration = row.get("frame_duration_s", observed)
            # Sequential interval manifests use a conservative zero tail at EOF
            # when no next presentation timestamp exists. It adds no coverage.
            conservative_tail = following is None and _number(duration) and duration == 0
            if conservative_tail:
                pass
            elif (not _number(observed) or not 0 < observed <= .1
                    or not _number(duration) or not 0 < duration <= .1
                    or abs(duration-observed) > max(.001, observed*.01)):
                incomplete = True
            else:
                # The tolerance permits rounded metadata, not extra verified
                # footage. Always accumulate the observed source PTS interval.
                seconds += observed
            fp = prediction is not None and prediction["state"] in MEASURED
            if fp:
                identity = prediction.get("track_id")
                known_track = (type(identity) is int and identity >= 0) or (isinstance(identity, str) and bool(identity))
                unknown_track |= not known_track
                same_track = (previous_fp is not None
                              and known_track and identity == previous_fp["prediction"].get("track_id"))
                contiguous = (previous_fp is not None and row["frame_id"] == previous_fp["frame_id"]+1
                              and row["timestamp_s"]-previous_fp["timestamp_s"] <= .1)
                if not (contiguous and same_track):
                    episodes += 1
                previous_fp = row
            else:
                previous_fp = None
    valid = absent_seen and not incomplete and seconds > 0
    return {"verified_absence_seconds": seconds if valid else None,
            "false_track_episodes": episodes if valid and not unknown_track else None,
            "false_track_episodes_per_minute": episodes*60/seconds if valid and not unknown_track else None,
            "absence_coverage_complete": valid,
            "episode_definition": "Consecutive accepted absent-frame detections with the same track identity; a miss, frame gap or >100ms ends an episode",
            "episode_rate_reason": "Accepted false detections lack a known track identity" if unknown_track else None,
            "coverage_reason": None if valid else "No complete consecutive, timestamped verified-absence coverage"}


def score_frames(rows, tolerance1080_px=4):
    rows = validate_rows(list(rows))
    if not _number(tolerance1080_px) or tolerance1080_px <= 0:
        raise ValueError("Positive finite localization tolerance required")
    counts = dict(frames=len(rows), visible_reference_frames=0, absent_reference_frames=0,
                  unknown_reference_frames=0, tp=0, wrong_location_predictions=0,
                  absent_false_positives=0, no_detection_visible_frames=0, true_negatives=0,
                  predicted_frames=0, abstained_frames=0)
    errors = []
    rallies = set()
    for row in rows:
        annotation, prediction = row.get("annotation", {}), row.get("prediction")
        measured = prediction is not None and prediction["state"] in MEASURED
        counts["predicted_frames"] += int(prediction is not None and prediction["state"] == "predicted")
        counts["abstained_frames"] += int(not measured)
        visibility = annotation.get("visibility", "unknown") if annotation.get("reviewed") is True else "unknown"
        if visibility in ("visible", "partial"):
            counts["visible_reference_frames"] += 1
            if row.get("rally_id") is not None:
                rallies.add((row["source_id"], str(row["rally_id"])))
            if measured:
                error = math.dist(prediction["center_px"], annotation["center_px"])*1080/row["source_size"][1]
                errors.append(error)
                counts["tp" if error < tolerance1080_px else "wrong_location_predictions"] += 1
            else:
                counts["no_detection_visible_frames"] += 1
        elif visibility == "absent":
            counts["absent_reference_frames"] += 1
            counts["absent_false_positives" if measured else "true_negatives"] += 1
        else:
            counts["unknown_reference_frames"] += 1
    fp = counts["wrong_location_predictions"] + counts["absent_false_positives"]
    fn = counts["wrong_location_predictions"] + counts["no_detection_visible_frames"]
    precision = counts["tp"]/(counts["tp"]+fp) if counts["tp"]+fp else None
    recall = counts["tp"]/counts["visible_reference_frames"] if counts["visible_reference_frames"] else None
    f1 = 2*counts["tp"]/(2*counts["tp"]+fp+fn) if 2*counts["tp"]+fp+fn else None
    absent_count = counts["absent_reference_frames"]
    return {**counts, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "precision_interval_95": _interval(counts["tp"], counts["tp"]+fp),
            "recall_interval_95": _interval(counts["tp"], counts["visible_reference_frames"]),
            "interval_method": "Frame Wilson; correlated frames are not independent session evidence",
            "sampled_negative_false_positive_rate": counts["absent_false_positives"]/absent_count if absent_count else None,
            "reviewed_rallies": len(rallies),
            "localization_1080_px": {"count": len(errors), "median": _percentile(errors, .5),
                                      "p95": _percentile(errors, .95),
                                      "rmse": math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None},
            "matching": {"tolerance_1080_px": tolerance1080_px, "comparison": "STRICT_LESS_THAN",
                         "error_scale": "source_distance * 1080 / source_height"},
            **_negative_coverage(rows)}


def evaluate_release(rows, baseline_rows=None, live_evidence=None, role="validation"):
    rows = validate_rows(list(rows))
    baseline = validate_rows(list(baseline_rows)) if baseline_rows is not None else None
    if baseline is not None:
        candidate_truth = {(r["source_id"], r["frame_id"]): (r["domain"], r["source_size"], r["timestamp_s"], r.get("annotation", {})) for r in rows}
        baseline_truth = {(r["source_id"], r["frame_id"]): (r["domain"], r["source_size"], r["timestamp_s"], r.get("annotation", {})) for r in baseline}
        if candidate_truth != baseline_truth:
            raise ValueError("Regression comparison must use identical source frames and reference annotations")
    checks, domains = {}, {}
    for domain in DOMAINS:
        selected = [r for r in rows if r["domain"] == domain]
        report = score_frames(selected)
        report["secondary_tolerances"] = {str(t): score_frames(selected, t) for t in (8, 15)}
        report["subsets"] = {}
        for tag in SUBSETS:
            subset = [r for r in selected if tag in r.get("annotation", {}).get("tags", [])
                      or (tag == "partial_occlusion" and r.get("annotation", {}).get("visibility") == "partial")]
            report["subsets"][tag] = score_frames(subset)
        domains[domain] = report
        if domain == "match":
            continue
        requirements = {"precision": report["precision"] is not None and report["precision"] >= .98,
                        "recall": report["recall"] is not None and report["recall"] >= .95,
                        "visible_count": report["visible_reference_frames"] >= 1000,
                        "rally_count": report["reviewed_rallies"] >= 10,
                        "absence_duration": report["verified_absence_seconds"] is not None and report["verified_absence_seconds"] >= 120-1e-6,
                        "absence_false_positive_rate": report["sampled_negative_false_positive_rate"] is not None and report["sampled_negative_false_positive_rate"] <= .01,
                        "false_tracks_per_minute": report["false_track_episodes_per_minute"] is not None and report["false_track_episodes_per_minute"] <= 1}
        for tag in SUBSETS:
            subset = report["subsets"][tag]
            requirements[tag+"_count"] = subset["visible_reference_frames"] >= 100
        checks.update({domain+":"+key: value for key, value in requirements.items()})
    checks["independent_role"] = role in ("validation", "test")
    checks["fixed_regression_comparison"] = baseline is not None
    if baseline is not None:
        for domain in DOMAINS:
            selected = [r for r in baseline if r["domain"] == domain]
            if not selected:
                continue
            reference = score_frames(selected)
            for key in ("precision", "recall"):
                if reference[key] is not None:
                    actual = domains[domain][key]
                    checks[domain+":no_regression_"+key] = actual is not None and actual >= reference[key]
            rate = reference["sampled_negative_false_positive_rate"]
            if rate is not None:
                actual = domains[domain]["sampled_negative_false_positive_rate"]
                checks[domain+":no_regression_absence"] = actual is not None and actual <= rate
    live_checks = {}
    # The iPhone domain is recording footage. The required live soak is for the
    # actual PC camera; an explicitly supplied iPhone live stream is additional
    # evidence, never silently substituted for the PC camera.
    live_domains = ["pc_camera"]
    if "iphone" in (live_evidence or {}):
        live_domains.append("iphone")
    for domain in live_domains:
        evidence = (live_evidence or {}).get(domain, {})
        fps = evidence.get("analyzed_fps")
        latency = evidence.get("p95_capture_to_display_ms")
        live_checks[domain+":live_60fps"] = (_number(fps) and fps >= 59.4
                and _number(evidence.get("supplied_fps")) and evidence["supplied_fps"] >= 59.4
                and evidence.get("capture_frames_dropped") == 0)
        live_checks[domain+":live_latency"] = _number(latency) and 0 <= latency <= 100
        live_checks[domain+":soak"] = _number(evidence.get("duration_s")) and evidence["duration_s"] >= 1800
        live_checks[domain+":timing_scope"] = evidence.get("scope") == "single_session_live_capture_to_display"
    accuracy_passed = all(checks.values())
    return {"schema_version": 1, "role": role, "domains": domains, "checks": checks,
            "failed_checks": [key for key, value in checks.items() if not value],
            "accuracy_gates_passed": accuracy_passed, "live_checks": live_checks,
            "live_gates_passed": all(live_checks.values()),
            "release_candidate_eligible": role == "test" and accuracy_passed and all(live_checks.values()),
            "promotion_approved": False,
            "live_domains_required": live_domains,
            "subset_policy": "Report accuracy separately; require at least 100 visible examples per supported subset",
            "limitation": "Rows need a separately audited frozen source/model/dataset manifest; arithmetic gates alone are not approval to deploy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--live-evidence", type=Path)
    parser.add_argument("--role", choices=("development", "validation", "test"), default="development")
    args = parser.parse_args()
    read = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    result = evaluate_release(read(args.rows), read(args.baseline) if args.baseline else None,
                              json.loads(args.live_evidence.read_text()) if args.live_evidence else None, args.role)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
