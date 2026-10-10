"""Read-only reviewed-label readiness, with source/image duplicate conflict checks.

Counts are structural eligibility, not label accuracy or permission to deploy.
No split is inferred and no model suggestion becomes a training target.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image
from prepare_live_training import validate_annotation
from export_reviewed_streaks import target
from review_absence_provenance import checked_interval_absence

SPLITS = ('train', 'val', 'test')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def audit(captures, splits_path=None):
    root = Path(captures).resolve()
    if not root.is_dir():
        raise ValueError('Capture directory does not exist')
    assignment, reasons, errors = {}, set(), []
    splits = None
    if splits_path is not None:
        splits = json.loads(Path(splits_path).read_text(encoding='utf-8'))
        if not isinstance(splits, dict) or set(splits) != set(SPLITS):
            raise ValueError('Explicit train, val and test capture lists required')
        for split in SPLITS:
            if not isinstance(splits[split], list) or not splits[split]:
                raise ValueError('Each split must name at least one complete recording')
            for clip in splits[split]:
                if not isinstance(clip, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', clip) or clip in assignment:
                    raise ValueError('Invalid or repeated split capture ID')
                assignment[clip] = split
    else:
        reasons.add('FROZEN_SPLITS_MISSING')
    directories = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.') and (p / 'capture.json').is_file())
    if splits is not None:
        available = {p.name for p in directories}
        if set(assignment) - available:
            raise ValueError('Frozen split references a missing complete capture')
        directories = [p for p in directories if p.name in assignment]
    records, clips, session_splits, parent = [], [], {}, []
    checked_sources = {}
    keys, images = {}, {}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        parent[find(b)] = find(a)

    for folder in directories:
        if folder.resolve().parent != root:
            raise ValueError('Capture path escapes root')
        cp, ap, ip = (folder / n for n in ('capture.json', 'annotations.json', 'frames.jsonl'))
        capture_bytes, annotation_bytes = cp.read_bytes(), ap.read_bytes()
        cap, annotations = json.loads(capture_bytes), json.loads(annotation_bytes)
        raw = ip.read_bytes()
        if not raw or not raw.endswith(b'\n') or any(not line.strip() for line in raw.decode('utf-8').splitlines()):
            raise ValueError('Complete newline-terminated frame index required')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        if not isinstance(annotations, dict) or cap.get('sampled_frames', len(rows)) != len(rows):
            raise ValueError('Capture/index/annotations shape mismatch')
        split, session = assignment.get(folder.name, 'unassigned'), cap.get('session_id')
        if cap.get('source') not in ('live', 'recorded_camera'):
            raise ValueError('Only real camera captures belong in this audit')
        if not isinstance(session, str) or not session.strip():
            reasons.add('SESSION_ID_MISSING')
        else:
            session_splits.setdefault(session, set()).add(split)
        if cap.get('split') is not None and cap['split'] != split:
            reasons.add('DECLARED_SPLIT_CONFLICT')
        original = cap.get('original_sha256')
        if original is not None and (not isinstance(original, str) or not re.fullmatch(r'[0-9a-f]{64}', original)):
            raise ValueError('Invalid original source SHA256')
        if cap.get('source') == 'recorded_camera':
            source_path = cap.get('original_path')
            if original is None or not isinstance(source_path, str) or not Path(source_path).is_file():
                raise ValueError('Original recording path and SHA256 are required')
            if source_path not in checked_sources:
                checked_sources[source_path] = sha(source_path)
            if checked_sources[source_path] != original:
                raise ValueError('Original recording SHA256 differs from capture provenance')
        clip_record = {'clip': folder.name, 'split': split, 'session_id': session, 'saved_frames': len(rows),
                       'annotations_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
                       'capture_sha256': hashlib.sha256(capture_bytes).hexdigest(),
                       'index_sha256': hashlib.sha256(raw).hexdigest(),
                       'temporal_continuity': 'NOT_AUDITED', 'frame_records': []}
        filenames, ids = set(), set()
        previous = None
        for row in rows:
            name, fid, pts = row.get('file'), row.get('source_frame'), row.get('elapsed_ms')
            if not isinstance(name, str) or not re.fullmatch(r'frame_[0-9]+\.jpg', name) or name in filenames:
                raise ValueError('Invalid or repeated frame filename')
            if type(fid) is not int or fid <= 0 or fid in ids or type(pts) not in (int, float) or not math.isfinite(pts) or pts < 0:
                raise ValueError('Unique positive frame IDs and finite timestamps required')
            if previous and not (fid > previous[0] and pts > previous[1]):
                raise ValueError('Frame index must be strictly chronological')
            previous = fid, pts
            filenames.add(name); ids.add(fid)
            image = (folder / name).resolve()
            if image.parent != folder.resolve():
                raise ValueError('Frame path escapes capture')
            with Image.open(image) as im:
                if im.format != 'JPEG':
                    raise ValueError('Source review image must be JPEG')
                rgb = im.convert('RGB')
                dimensions = rgb.size
                decoded = hashlib.sha256(str(dimensions).encode() + rgb.tobytes()).hexdigest()
            annotation = annotations.get(name, {})
            ball, cnn = None, None
            try:
                if not isinstance(annotation, dict):
                    raise ValueError('Annotation must be an object')
                checked_interval_absence(annotation, cap, row, clip_record['index_sha256'], sha(image), folder / 'annotation_history')
            except (ValueError, TypeError, KeyError) as ex:
                errors.append({'clip': folder.name, 'file': name, 'kind': 'shared_provenance', 'reason': str(ex)})
                reasons.add('INVALID_REVIEWED_LABEL_OR_PROVENANCE')
            else:
                try:
                    ball = validate_annotation(annotation, ('ball',))
                except (ValueError, TypeError, KeyError) as ex:
                    errors.append({'clip': folder.name, 'file': name, 'kind': 'ball', 'reason': str(ex)})
                    reasons.add('INVALID_BALL_TARGET')
                try:
                    cnn = target(annotation, dimensions)
                    if cnn is not None and cnn['visibility'] == 'partial':
                        cnn = None
                except (ValueError, TypeError, KeyError) as ex:
                    errors.append({'clip': folder.name, 'file': name, 'kind': 'cnn', 'reason': str(ex)})
                    reasons.add('INVALID_CNN_TARGET')
            source_key = (original, fid) if original is not None else (folder.name, fid)
            index = len(records); parent.append(index)
            record = {'clip': folder.name, 'file': name, 'split': split, 'source_key': source_key, 'decoded_sha256': decoded,
                      'source_frame': fid, 'elapsed_ms': pts, 'burst_id': row.get('burst_id'), 'dimensions': dimensions,
                      'saved_reviewed': annotation.get('reviewed') is True if isinstance(annotation, dict) else False,
                      'ball': ball, 'cnn': cnn}
            records.append(record); clip_record['frame_records'].append(index)
            if source_key in keys:
                union(index, keys[source_key])
            if decoded in images:
                union(index, images[decoded])
            keys[source_key], images[decoded] = index, index
        clips.append(clip_record)
    for clip in clips:
        for name, field in [('capture.json', 'capture_sha256'), ('annotations.json', 'annotations_sha256'), ('frames.jsonl', 'index_sha256')]:
            if sha(root / clip['clip'] / name) != clip[field]:
                raise ValueError('Capture/index/annotations changed during readiness audit; retry the saved snapshot')
    if any(len(parts) > 1 for parts in session_splits.values()):
        reasons.add('SESSION_SPLIT_LEAKAGE')
    if len(session_splits) < 3:
        reasons.add('INDEPENDENT_SESSIONS_INSUFFICIENT')
    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(find(index), []).append(row)
    count_names = ('unique_review_images', 'ball_positive', 'ball_verified_absent', 'ball_unknown_or_excluded',
                   'cnn_visible_frames', 'cnn_absent_frames', 'cnn_unknown_or_excluded', 'cnn_structural_three_frame_windows')
    counts = {s: Counter({name: 0 for name in count_names}) for s in (*SPLITS, 'unassigned')}
    conflicts, bad_groups, cnn_bad_groups = [], set(), set()
    for group_id, group in groups.items():
        parts = {r['split'] for r in group}
        if len(parts) > 1:
            reasons.add('SOURCE_OR_DECODED_IMAGE_SPLIT_LEAKAGE'); bad_groups.add(group_id)
        selected = {}
        for kind in ('ball', 'cnn'):
            labels = {canonical(r[kind]): r[kind] for r in group if r[kind] is not None}
            if len(labels) > 1:
                reasons.add('CONFLICTING_DUPLICATE_LABELS')
                if kind == 'cnn':
                    cnn_bad_groups.add(group_id)
                conflicts.append({'kind': kind, 'frames': [{'clip': r['clip'], 'file': r['file']} for r in group]})
            selected[kind] = next(iter(labels.values()), None) if len(labels) == 1 else None
        if group_id in bad_groups:
            continue
        c = counts[next(iter(parts))]; c['unique_review_images'] += 1
        if selected['ball'] is None:
            c['ball_unknown_or_excluded'] += 1
        else:
            c['ball_positive' if selected['ball']['ball'] else 'ball_verified_absent'] += 1
        if selected['cnn'] is None:
            c['cnn_unknown_or_excluded'] += 1
        else:
            c['cnn_' + selected['cnn']['visibility'] + '_frames'] += 1
    seen_windows = set()
    for clip in clips:
        sequence = clip.pop('frame_records')
        for i in range(2, len(sequence)):
            indexes = sequence[i-2:i+1]; group = [records[j] for j in indexes]
            groupids = tuple(find(j) for j in indexes)
            ids = [r['source_frame'] for r in group]
            if (any(j in bad_groups or j in cnn_bad_groups for j in groupids) or any(r['cnn'] is None for r in group)
                    or ids != list(range(ids[0], ids[0] + 3))
                    or group[0]['burst_id'] is None or len({str(r['burst_id']) for r in group}) != 1
                    or len({r['dimensions'] for r in group}) != 1 or groupids in seen_windows):
                continue
            seen_windows.add(groupids)
            counts[clip['split']]['cnn_structural_three_frame_windows'] += 1
    ball_reasons = set(reasons) - {'INVALID_CNN_TARGET'}
    for split in SPLITS:
        c = counts[split]
        if c['ball_positive'] + c['ball_verified_absent'] < 30:
            ball_reasons.add('BALL_REVIEWED_MINIMUM_NOT_MET_' + split.upper())
        if c['ball_positive'] < 5 or c['ball_verified_absent'] < 5:
            ball_reasons.add('BALL_POSITIVE_ABSENT_MINIMUM_NOT_MET_' + split.upper())
    cnn_reasons = (set(reasons) - {'INVALID_BALL_TARGET'}) | {'TEMPORAL_CONTINUITY_NOT_AUDITED', 'BLURBALL_BACKWARD_ENVIRONMENT_NOT_ASSESSED_BY_THIS_AUDIT'}
    for split in SPLITS:
        if counts[split]['cnn_structural_three_frame_windows'] == 0:
            cnn_reasons.add('CNN_REVIEWED_WINDOWS_MISSING_' + split.upper())
    return {'schema_version': 1, 'scope': 'Read-only structural readiness; no fitting, promotion, label correctness or temporal continuity claim.',
            'capture_count': len(clips), 'saved_images': len(records), 'saved_reviewed': sum(r['saved_reviewed'] for r in records),
            'unique_source_frames': len(keys), 'unique_decoded_images': len(images), 'deduplicated_review_images': len(groups),
            'duplicate_saved_images': len(records) - len(groups), 'session_splits': {k: sorted(v) for k, v in session_splits.items()},
            'splits_sha256': sha(splits_path) if splits_path else None, 'counts_by_split': {s: dict(c) for s, c in counts.items()},
            'ball': {'structurally_ready': not ball_reasons, 'reason_codes': sorted(ball_reasons)},
            'blurball': {'structural_windows': len(seen_windows), 'ready_for_fitting': False, 'reason_codes': sorted(cnn_reasons)},
            'conflicts': conflicts, 'errors': errors, 'capture_lineage': clips,
            'minimum_note': 'YOLO trainer minimum: 30 reviewed frames and five positive/five verified absent per independent split; this is not sufficient sample size or acceptance.',
            'continuity_note': 'Consecutive IDs/timestamps and burst IDs supply structural windows only. Cuts, repeated footage and exposure interruptions require a separate continuity audit.',
            'environment_note': 'This audit does not assess the training environment. Separate forward/loss/backward/finite-gradient/optimizer evidence is required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('--splits', type=Path)
    parser.add_argument('--output', type=Path, help='Optional new evidence file; existing output is never overwritten')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError('Use a new report path')
    result = audit(args.captures, args.splits)
    body = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(body)
    print(body, end='')


if __name__ == '__main__':
    main()
