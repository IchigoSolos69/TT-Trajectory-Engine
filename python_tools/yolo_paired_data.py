"""Training-only paired-camera export; sealed C labels are never opened here."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from PIL import Image
from prepare_live_training import validate_annotation
from review_absence_provenance import checked_interval_absence
from export_reviewed_streaks import target as precise_ball_target
from recording_source import resolve_original_path
from prepare_absence_interval import checked_child

DOMAINS = ("match", "iphone", "pc_camera")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_training_sources(captures, registry_path, match_root, remember):
    """Read A/train and B/val only. Test readiness is explicitly unaudited."""
    captures, match_root = Path(captures).resolve(), Path(match_root).resolve()
    blockers, records, registry = [], [], None
    if registry_path is None or not Path(registry_path).is_file():
        return {"records": [], "blockers": ["PAIRED_SESSION_REGISTRY_MISSING"],
                "test_labels_consumed": False, "sealed_test_readiness": "NOT_AUDITED"}
    remember(registry_path)
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("sessions"), dict) or not isinstance(registry.get("captures"), dict):
        raise ValueError("Versioned paired session registry required")
    decoded_roles, identities, unique_records = {}, {}, {}
    conflicting_keys, conflicting_images, conflicts, source_issues = set(), set(), [], []
    audit = Counter()
    source_bindings={};conflicting_sources=set();source_binding_conflicts=[]
    # Bind whole-original ownership before opening development labels. C remains
    # unopened: only its registry role is consulted here.
    for clip,assignment in sorted(registry["captures"].items()):
        session=registry["sessions"].get(assignment.get("session_id"),{})
        role=session.get("role");domain=assignment.get("domain")
        if role=="test" and not assignment.get("source_sha256"):
            blockers.append("SEALED_C_SOURCE_IDENTITY_METADATA_REQUIRED")
        if role not in ("train","val","test") or domain not in DOMAINS[1:]:continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+",clip):raise ValueError("Invalid registry capture ID")
        registered_sha=assignment.get("source_sha256")
        if registered_sha is not None and (not isinstance(registered_sha,str) or not re.fullmatch(r"[0-9a-f]{64}",registered_sha)):
            raise ValueError("Valid immutable registry source SHA required")
        if role=="test":
            source_sha=registered_sha  # Metadata already assigned at sealing; no C files opened.
        else:
            folder=(captures/clip).resolve()
            if folder.parent!=captures or folder.name!=clip:raise ValueError("Capture alias escapes training root")
            capture_path=checked_child(folder,"capture.json");remember(capture_path)
            capture=json.loads(capture_path.read_text(encoding="utf-8"))
            source_sha=capture.get("original_sha256")
            if registered_sha is not None and registered_sha!=source_sha:
                blockers.append("CAPTURE_SOURCE_SHA_REGISTRY_CONFLICT")
                conflicting_sources.update(s for s in (registered_sha,source_sha) if isinstance(s,str))
                source_binding_conflicts.append(dict(clip=clip,registered_source_sha256=registered_sha,
                    actual_source_sha256=source_sha,reason="registry_source_identity_changed"))
        if not isinstance(source_sha,str) or not re.fullmatch(r"[0-9a-f]{64}",source_sha):continue
        binding=(assignment["session_id"],role,domain)
        if source_sha in source_bindings and source_bindings[source_sha]!=binding:
            conflicting_sources.add(source_sha);blockers.append("ORIGINAL_SOURCE_SESSION_ROLE_DEVICE_CONFLICT")
            source_binding_conflicts.append(dict(source_sha256=source_sha,clip=clip,
                existing=list(source_bindings[source_sha]),conflicting=list(binding)))
        else:source_bindings[source_sha]=binding
    for clip, assignment in sorted(registry["captures"].items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
            raise ValueError("Invalid registry capture ID")
        session = registry["sessions"].get(assignment.get("session_id"), {})
        role, domain = session.get("role"), assignment.get("domain")
        if role == "test":
            audit["sealed_c_captures_not_opened"] += 1
            if session.get("stage") != "C" or session.get("sealed") is not True:
                blockers.append("TEST_C_NOT_SEALED")
            continue  # no capture, index, image or annotation read for sealed test
        if role not in ("train", "val") or domain not in DOMAINS[1:]:
            audit["unassigned_captures"] += 1
            blockers.append("CAPTURE_ROLE_OR_DEVICE_UNASSIGNED")
            continue
        if session.get("sealed") is True or session.get("stage") not in (("A", "existing") if role == "train" else ("B",)):
            raise ValueError("Train/validation session stage or seal conflicts with role")
        folder = (captures / clip).resolve()
        audit["assigned_train_val_captures"] += 1
        if folder.parent != captures or folder.name!=clip:
            raise ValueError("Capture escapes training root")
        capture_path=checked_child(folder,"capture.json");remember(capture_path)
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if (capture.get("source") not in ("live", "recorded_camera")
                or (capture.get("session_id") is not None and capture["session_id"] != assignment["session_id"])
                or (capture.get("source") == "recorded_camera" and capture.get("session_id") != assignment["session_id"])):
            raise ValueError("Registry must preserve actual camera session identity")
        source_id = capture.get("original_sha256") or f"capture:{clip}"
        if source_id in conflicting_sources:
            audit["whole_source_conflicting_captures_excluded"]+=1
            continue
        for name in ("annotations.json","frames.jsonl"):remember(checked_child(folder,name))
        original_source_verified = capture.get("source") == "live"
        resolved_source_path = None
        if capture.get("source") == "recorded_camera":
            try:
                resolved_source_path=resolve_original_path(capture,captures,remember).as_posix()
                original_source_verified = True
            except FileNotFoundError:
                blockers.append("ORIGINAL_RECORDING_UNAVAILABLE")
                source_issues.append(dict(clip=clip,source_id=source_id,path=capture["original_path"],
                                          reason="original_recording_unavailable"))
            except ValueError as error:
                blockers.append("ORIGINAL_RECORDING_PROVENANCE_INVALID")
                source_issues.append(dict(clip=clip,source_id=source_id,path=capture["original_path"],
                                          reason="original_provenance_invalid",detail=str(error)))
        annotations = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
        index_bytes = (folder / "frames.jsonl").read_bytes()
        index_hash = hashlib.sha256(index_bytes).hexdigest()
        source_rows = [json.loads(line) for line in index_bytes.decode("utf-8").splitlines()]
        continuity_file = folder / "continuity_audit.json"
        continuity = None
        if continuity_file.is_file():
            remember(continuity_file)
            continuity = json.loads(continuity_file.read_text(encoding="utf-8"))
            if (continuity.get("schema_version") != 1 or continuity.get("reviewed") is not True
                    or continuity.get("index_sha256") != index_hash or continuity.get("source_id") != source_id):
                raise ValueError("Continuity audit must match current source and index")
        history = folder / "annotation_history"
        if history.is_dir():
            for path in sorted(history.rglob("*.json")):
                remember(path)
        previous = None
        for raw in source_rows:
            audit["indexed_train_val_frames"] += 1
            filename, frame_id, elapsed = raw.get("file"), raw.get("source_frame"), raw.get("elapsed_ms")
            if not isinstance(filename, str) or not re.fullmatch(r"frame_[0-9]+\.jpg", filename):
                raise ValueError("Invalid source frame filename")
            if type(frame_id) is not int or frame_id <= 0 or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Positive source frame and finite source timestamp required")
            if previous and (frame_id <= previous[0] or elapsed <= previous[1]):
                raise ValueError("Source review index must be chronological")
            previous = frame_id, elapsed
            annotation = annotations.get(filename, {})
            labels = validate_annotation(annotation, ("ball",))
            if labels is None:
                audit["unreviewed_ball_frames_excluded"] += 1
                continue
            image = checked_child(folder,filename)
            image_hash = remember(image)
            checked_interval_absence(annotation, capture, raw, index_hash, image_hash, history)
            with Image.open(image) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                decoded_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if decoded_hash in decoded_roles and decoded_roles[decoded_hash] != role:
                blockers.append("DECODED_IMAGE_TRAIN_VAL_OVERLAP")
                conflicting_images.add(decoded_hash)
                conflicts.append(dict(reason="split_overlap", clip=clip, source_id=source_id,
                                      frame_id=frame_id, decoded_sha256=decoded_hash))
            decoded_roles[decoded_hash] = role
            geometry = annotation.get("geometry", {})
            if geometry.get("visibility") not in ("visible","partial","absent"):
                audit["unknown_visibility_frames_excluded"] += 1
                continue  # explicit hidden/unknown remains excluded despite suggestions/boxes
            if len(labels["ball"]) > 1:
                audit["multiple_ball_frames_excluded"] += 1
                blockers.append("MULTIPLE_BALL_TARGETS_REQUIRE_REVIEW")
                continue
            center = None
            if labels["ball"]:
                x, y, w, h = labels["ball"][0]
                center = [(x+w/2)*width, (y+h/2)*height]
            precise=precise_ball_target(annotation,(width,height))
            if precise and precise["position_px"] is not None:
                center=list(precise["position_px"])
            ranges = continuity.get("source_frame_ranges", []) if continuity else []
            audited = any(type(a) is int and type(b) is int and a <= frame_id <= b for a, b in ranges)
            if audited and continuity.get("image_sha256",{}).get(filename) != image_hash:
                audited=False
            record = dict(image=image.resolve().as_posix(), image_sha256=image_hash,
                          decoded_sha256=decoded_hash, labels=labels, clip=clip,
                          source_id=source_id, frame_id=frame_id, timestamp_s=elapsed/1000,
                          original_source_verified=original_source_verified,
                          resolved_source_path=resolved_source_path,
                          source_size=[width, height], domain=domain, role=role,
                          stage=session["stage"],
                          session_id=assignment["session_id"], burst_id=raw.get("burst_id"),
                          burst_index=raw.get("burst_index"),continuity_audited=audited,
                          precise_target=precise,positive=bool(labels["ball"]),
                          annotation=dict(reviewed=True, visibility=geometry.get("visibility", "visible" if center else "absent"),
                                          center_px=center, tags=["blur" if tag=="motion_blur" else tag for tag in geometry.get("quality_tags", [])]),
                          rally_id=geometry.get("rally_id") if geometry.get("rally_reviewed") is True else None)
            key = (source_id, frame_id)
            signature = json.dumps(dict(labels=labels,center=center), sort_keys=True)
            if key in identities and identities[key] != (role, signature):
                blockers.append("CONFLICTING_SOURCE_FRAME_LABELS_OR_SPLIT")
                conflicting_keys.add(key)
                conflicts.append(dict(reason="source_frame_conflict", clip=clip, source_id=source_id,
                                      frame_id=frame_id, decoded_sha256=decoded_hash))
            identities[key] = (role, signature)
            if decoded_hash in unique_records:
                if unique_records[decoded_hash]["labels"] != labels or unique_records[decoded_hash]["annotation"]["center_px"] != center:
                    blockers.append("CONFLICTING_DUPLICATE_IMAGE_LABELS")
                    conflicting_images.add(decoded_hash)
                    conflicts.append(dict(reason="duplicate_image_conflict", clip=clip, source_id=source_id,
                                          frame_id=frame_id, decoded_sha256=decoded_hash))
                else:
                    audit["identical_reviewed_image_copies_deduplicated"] += 1
                continue
            unique_records[decoded_hash] = record
            records.append(record)
    from audit_absence_intervals import load_confirmed_intervals
    interval_audit=load_confirmed_intervals(captures,registry,remember)
    blockers.extend(interval_audit["blockers"])
    for record in interval_audit["records"]:
        if record["source_id"] in conflicting_sources:continue
        source_id,frame_id,decoded_hash,role=(record[k] for k in ("source_id","frame_id","decoded_sha256","role"))
        labels=record["labels"];center=record["annotation"]["center_px"]
        key=(source_id,frame_id);signature=json.dumps(dict(labels=labels,center=center),sort_keys=True)
        if key in identities and identities[key]!=(role,signature):
            blockers.append("CONFLICTING_SOURCE_FRAME_LABELS_OR_SPLIT");conflicting_keys.add(key)
            conflicts.append(dict(reason="source_frame_conflict",clip=record["clip"],source_id=source_id,
                frame_id=frame_id,decoded_sha256=decoded_hash))
        identities[key]=(role,signature)
        if decoded_hash in decoded_roles and decoded_roles[decoded_hash]!=role:
            blockers.append("DECODED_IMAGE_TRAIN_VAL_OVERLAP");conflicting_images.add(decoded_hash)
            conflicts.append(dict(reason="split_overlap",clip=record["clip"],source_id=source_id,
                frame_id=frame_id,decoded_sha256=decoded_hash))
        decoded_roles[decoded_hash]=role
        if decoded_hash in unique_records:
            other=unique_records[decoded_hash]
            if other["labels"]!=labels or other["annotation"]["center_px"]!=center:
                blockers.append("CONFLICTING_DUPLICATE_IMAGE_LABELS");conflicting_images.add(decoded_hash)
                conflicts.append(dict(reason="duplicate_image_conflict",clip=record["clip"],source_id=source_id,
                    frame_id=frame_id,decoded_sha256=decoded_hash))
            else:audit["identical_reviewed_image_copies_deduplicated"]+=1
            continue
        unique_records[decoded_hash]=record;records.append(record)
    for role in ("train", "val"):
        listing = match_root / f"{role}.txt"
        if not listing.is_file():
            blockers.append("MATCH_REGRESSION_DATA_MISSING")
            continue
        remember(listing)
        for index, text in enumerate(listing.read_text(encoding="utf-8").splitlines()):
            image = Path(text).resolve()
            if "test_" in image.name or not image.is_file():
                raise ValueError("Match fitting lists must exclude test footage")
            image_hash = remember(image)
            label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
            remember(label)
            boxes = []
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError("Match labels must be ball-only boxes")
                x, y, w, h = map(float, fields[1:])
                if not all(math.isfinite(v) for v in (x,y,w,h)):
                    raise ValueError("Nonfinite match label")
                boxes.append([x-w/2, y-h/2, w, h])
            labels = validate_annotation(dict(reviewed=True, ball=boxes, geometry={"visibility":"visible" if boxes else "absent"}), ("ball",))
            with Image.open(image) as img:
                width, height = img.size
                decoded_hash = hashlib.sha256(str(img.size).encode()+img.convert("RGB").tobytes()).hexdigest()
            if decoded_hash in decoded_roles and decoded_roles[decoded_hash] != role:
                blockers.append("DECODED_IMAGE_TRAIN_VAL_OVERLAP")
                conflicting_images.add(decoded_hash)
                conflicts.append(dict(reason="split_overlap", source_id=f"match:{image.stem}",
                                      frame_id=index+1, decoded_sha256=decoded_hash))
            decoded_roles[decoded_hash] = role
            center = None if not boxes else [(boxes[0][0]+boxes[0][2]/2)*width,(boxes[0][1]+boxes[0][3]/2)*height]
            records.append(dict(image=image.as_posix(), image_sha256=image_hash, labels=labels,
                                decoded_sha256=decoded_hash, domain="match", role=role, positive=bool(boxes),
                                source_id=f"match:{image.stem}", frame_id=index+1, timestamp_s=0,
                                source_size=[width,height], session_id="historical_match_development",
                                annotation=dict(reviewed=True,visibility="visible" if boxes else "absent",center_px=center,tags=[]), rally_id=None))
    before = len(records)
    records = [r for r in records if (r["source_id"],r["frame_id"]) not in conflicting_keys
               and r["decoded_sha256"] not in conflicting_images]
    audit["conflicting_unique_records_excluded"] = before-len(records)
    seen_keys, unique = set(), []
    for record in records:
        key = record["source_id"],record["frame_id"]
        if key in seen_keys:
            audit["identical_source_frames_deduplicated"] += 1
            continue
        seen_keys.add(key)
        unique.append(record)
    records = unique
    live_records = [r for r in records if r["domain"] != "match"]
    summary = {role: {domain: {"visible": sum(r["positive"] for r in live_records if r["role"]==role and r["domain"]==domain),
                               "absent": sum(not r["positive"] for r in live_records if r["role"]==role and r["domain"]==domain)}
                      for domain in DOMAINS[1:]} for role in ("train", "val")}
    new_train=[r for r in live_records if r["role"]=="train" and r["stage"]=="A"]
    windows, durations, rallies = 0, Counter(), {}
    # Duration comes only from complete source decode and explicitly reviewed
    # original intervals, never from the cadence of sparse saved bursts.
    for role,domains in interval_audit["coverage_seconds"].items():
        for domain,seconds in domains.items():durations[(role,domain)]=seconds
    if conflicting_sources:
        for role in ("train","val"):
            for domain in DOMAINS[1:]:
                frames={}
                for r in interval_audit["records"]:
                    if r["source_id"] not in conflicting_sources and r["role"]==role and r["domain"]==domain:
                        frames[(r["source_id"],r["frame_id"])]=r["frame_duration_s"]
                durations[(role,domain)]=math.fsum(frames.values())
    for role in ("train", "val"):
        for domain in DOMAINS[1:]:
            subset = [r for r in live_records if r["role"]==role and r["domain"]==domain]
            grouped = {}
            for row in subset:
                grouped.setdefault((row["source_id"],row["clip"]),[]).append(row)
            for group in grouped.values():
                group.sort(key=lambda r:r["frame_id"])
                for a,b,c in zip(group,group[1:],group[2:]):
                    if (role=="train" and all(r["stage"]=="A" and r["positive"] and r["continuity_audited"] for r in (a,b,c))
                            and c["precise_target"] is not None and a["burst_id"] is not None
                            and a["burst_id"]==b["burst_id"]==c["burst_id"]
                            and type(a["burst_index"]) is int and b["burst_index"]==a["burst_index"]+1 and c["burst_index"]==b["burst_index"]+1
                            and a["source_size"]==b["source_size"]==c["source_size"]
                            and b["frame_id"]==a["frame_id"]+1 and c["frame_id"]==b["frame_id"]+1
                            and 0<b["timestamp_s"]-a["timestamp_s"]<=.1 and 0<c["timestamp_s"]-b["timestamp_s"]<=.1):
                        windows += 1
            rallies[(role,domain)] = len({(r["source_id"],r["rally_id"]) for r in subset if r["rally_id"] is not None})
            if not summary[role][domain]["visible"] or not summary[role][domain]["absent"]:
                blockers.append(f"{role.upper()}_{domain.upper()}_BOTH_CLASSES_MISSING")
            if role=="val":
                if summary[role][domain]["visible"] < 1000: blockers.append(f"VAL_{domain.upper()}_1000_VISIBLE_REQUIRED")
                if rallies[(role,domain)] < 10: blockers.append(f"VAL_{domain.upper()}_10_REVIEWED_RALLIES_REQUIRED")
                if durations[(role,domain)] < 120: blockers.append(f"VAL_{domain.upper()}_120S_VERIFIED_ABSENCE_REQUIRED")
    train_rows=[r for r in live_records if r["role"]=="train"]
    if sum(r["positive"] for r in train_rows) < 2000: blockers.append("TRAIN_2000_VISIBLE_REQUIRED")
    if sum(not r["positive"] for r in train_rows) < 1000: blockers.append("TRAIN_1000_VERIFIED_ABSENT_REQUIRED")
    for domain in DOMAINS[1:]:
        if not any(r["domain"]==domain for r in new_train): blockers.append(f"TRAIN_A_{domain.upper()}_MISSING")
    if windows < 500: blockers.append("TRAIN_500_AUDITED_POSITIVE_WINDOWS_REQUIRED")
    audit["conflicting_source_frame_keys"] = len(conflicting_keys)
    audit["conflicting_decoded_image_keys"] = len(conflicting_images)
    return dict(records=records, blockers=sorted(set(blockers)), counts=summary, review_audit=dict(audit),
                annotation_conflicts=conflicts,original_source_issues=source_issues,positive_train_windows=windows,
                original_source_binding_conflicts=source_binding_conflicts,
                complete_absence_intervals=interval_audit["summaries"],
                complete_absence_interval_issues=interval_audit["issues"],
                unreviewed_absence_intervals_skipped=interval_audit["unreviewed"],
                sealed_absence_intervals_not_opened=interval_audit["sealed_not_opened"],
                validation_absence_s={d:durations[("val",d)] for d in DOMAINS[1:]},
                validation_rallies={d:rallies[("val",d)] for d in DOMAINS[1:]},
                test_labels_consumed=False,sealed_test_readiness="NOT_AUDITED")


def prepare_records(report, output):
    """Export an exclusive train/val directory, even while readiness is blocked."""
    output = Path(output).resolve()
    if not report["records"]:
        raise ValueError("No assigned reviewed training/validation records to export")
    output.mkdir(parents=True, exist_ok=False)
    rows=[]
    for role in ("train", "val"):
        paths=[]
        (output/"images"/role).mkdir(parents=True)
        (output/"labels"/role).mkdir(parents=True)
        for index, record in enumerate((r for r in report["records"] if r["role"]==role),1):
            target=output/"images"/role/f"frame_{index:08d}.jpg"
            shutil.copyfile(record["image"],target)
            if digest(target)!=record["image_sha256"]: raise ValueError("Source changed during prepared copy")
            label=output/"labels"/role/f"{target.stem}.txt"
            label.write_text("\n".join(f"0 {x+w/2:.8f} {y+h/2:.8f} {w:.8f} {h:.8f}" for x,y,w,h in record["labels"]["ball"]),encoding="utf-8")
            rows.append({**record,"prepared_image":target.as_posix(),"prepared_label":label.as_posix(),"prepared_label_sha256":digest(label)})
            paths.append(target.as_posix())
        (output/f"{role}.txt").write_text("\n".join(paths)+"\n",encoding="utf-8")
    (output/"data.yaml").write_text(f"train: {(output/'train.txt').as_posix()}\nval: {(output/'val.txt').as_posix()}\nnames:\n  0: ball\n",encoding="utf-8")
    (output/"training_metadata.json").write_text(json.dumps(dict(schema_version=1,records=rows,test_exported=False,strict_data_blockers=report["blockers"]),indent=2)+"\n",encoding="utf-8")
    return output
