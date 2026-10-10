"""Snapshot reviewed causal CNN windows from one unassigned camera session.

This is a read-only handoff, not a train/validation/test export or permission to
fit/promote a model. No labels, original footage, or split assignments are edited.
Use a new output directory; StagedReviewedDataset revalidates the saved snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from prepare_live_training import validate_annotation
from export_reviewed_streaks import target
from review_absence_provenance import checked_interval_absence
from blurball_reviewed_dataset import (INPUT_WH, MEAN, STD, TARGET_METHOD,
                                       source_to_output_affine, transform_target,
                                       line_target)

SCHEMA = 1
CODE_FILES = ('stage_reviewed_training.py', 'prepare_live_training.py',
              'export_reviewed_streaks.py', 'review_absence_provenance.py',
              'blurball_reviewed_dataset.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def index_rows(raw):
    if not raw or not raw.endswith(b'\n'):
        raise ValueError('Complete newline-terminated frame index required')
    lines = raw.decode('utf-8').splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError('Blank frame-index row')
    rows = [json.loads(line) for line in lines]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Frame-index rows must be objects')
    return rows


def snapshot(captures, session_id):
    """Strict source snapshot and same dedup/conflict grouping as readiness.

    Duplicate labels are not averaged, selected by recency, or silently promoted
    from another capture. Each accepted window retains its own reviewed targets.
    """
    root = Path(captures).resolve()
    if not root.is_dir() or not isinstance(session_id, str) or not session_id.strip():
        raise ValueError('Existing capture root and explicit session identity required')
    paths = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.') and (p / 'capture.json').is_file())
    snapshots, fingerprints, source_sessions, decoded_sessions = {}, {}, {}, {}
    records, clips, catalog, errors, parents = [], [], [], [], []
    source_keys, image_keys = {}, {}

    def remember(path, expected=None):
        path = Path(path).resolve()
        key = path.as_posix()
        if key not in snapshots:
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed while hashing snapshot')
            snapshots[key] = digest
            fingerprints[key] = (after.st_size, after.st_mtime_ns)
        if expected is not None and snapshots[key] != expected:
            raise ValueError('Original recording SHA256 differs from capture provenance')
        return snapshots[key]

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(b)] = find(a)

    for folder in paths:
        if folder.resolve().parent != root or not re.fullmatch(r'[A-Za-z0-9_-]+', folder.name):
            raise ValueError('Capture directory escapes root or has an invalid identity')
        cp, ap, ip = (folder / name for name in ('capture.json', 'annotations.json', 'frames.jsonl'))
        cap = json.loads(cp.read_text(encoding='utf-8'))
        if not isinstance(cap, dict) or cap.get('source') not in ('live', 'recorded_camera'):
            raise ValueError('Only real camera capture metadata is eligible')
        session = cap.get('session_id')
        if not isinstance(session, str) or not session.strip():
            raise ValueError('Explicit session identity required for every camera capture')
        original = cap.get('original_sha256')
        if original is not None and (not isinstance(original, str) or not re.fullmatch(r'[0-9a-f]{64}', original)):
            raise ValueError('Invalid original recording SHA256')
        original_path = cap.get('original_path')
        if cap['source'] == 'recorded_camera':
            if original is None or not isinstance(original_path, str) or not Path(original_path).is_file():
                raise ValueError('Original recording path and SHA256 required')
            remember(original_path, original)
        if original is not None:
            if original in source_sessions and source_sessions[original] != session:
                raise ValueError('Same original recording is declared in different sessions')
            source_sessions[original] = session
        for path in (cp, ap, ip):
            remember(path)
        annotations = json.loads(ap.read_text(encoding='utf-8'))
        rows = index_rows(ip.read_bytes())
        if not isinstance(annotations, dict) or (cap.get('sampled_frames') is not None and
                (type(cap['sampled_frames']) is not int or cap['sampled_frames'] != len(rows))):
            raise ValueError('Capture/index/annotations shape mismatch')
        selected = session == session_id
        if selected and cap.get('split') is not None:
            raise ValueError('Session staging requires unassigned captures; use the frozen-split exporter instead')
        catalog.append({'clip': folder.name, 'session_id': session, 'selected': selected,
                        'original_sha256': original, 'capture_sha256': snapshots[cp.resolve().as_posix()]})
        clip = {'clip': folder.name, 'session_id': session, 'partition': 'unassigned',
                'original_sha256': original, 'original_path': original_path,
                'capture_sha256': snapshots[cp.resolve().as_posix()],
                'annotations_sha256': snapshots[ap.resolve().as_posix()],
                'index_sha256': snapshots[ip.resolve().as_posix()], 'saved_frames': len(rows), 'record_indices': []}
        filenames, ids, previous = set(), set(), None
        for row in rows:
            name, fid, pts = row.get('file'), row.get('source_frame'), row.get('elapsed_ms')
            if not isinstance(name, str) or not re.fullmatch(r'frame_[0-9]+\.jpg', name) or name in filenames:
                raise ValueError('Invalid or repeated frame filename')
            if type(fid) is not int or fid <= 0 or fid in ids or not finite(pts) or pts < 0:
                raise ValueError('Unique positive source-frame IDs and finite timestamps required')
            if previous is not None and (fid <= previous[0] or pts <= previous[1]):
                raise ValueError('Frame index must be strictly chronological')
            previous = fid, pts
            ids.add(fid); filenames.add(name)
            image = (folder / name).resolve()
            if image.parent != folder.resolve():
                raise ValueError('Image path escapes capture')
            image_hash = remember(image)
            with Image.open(image) as im:
                if im.format != 'JPEG':
                    raise ValueError('Review frame must be an original JPEG')
                rgb = im.convert('RGB')
                dimensions = rgb.size
                decoded_hash = hashlib.sha256(str(dimensions).encode() + rgb.tobytes()).hexdigest()
            if decoded_hash in decoded_sessions and decoded_sessions[decoded_hash] != session:
                raise ValueError('Identical decoded image occurs in different sessions')
            decoded_sessions[decoded_hash] = session
            if not selected:
                continue
            annotation = annotations.get(name, {})
            label, ball, exclusion = None, None, None
            try:
                if not isinstance(annotation, dict):
                    raise ValueError('Annotation must be an object')
                if 'source_frame' in annotation and (type(annotation['source_frame']) is not int or annotation['source_frame'] != fid):
                    raise ValueError('Annotation source frame differs from index')
                if 'elapsed_ms' in annotation and (not finite(annotation['elapsed_ms']) or abs(annotation['elapsed_ms'] - pts) > 1e-6):
                    raise ValueError('Annotation timestamp differs from index')
                flags = annotation.get('reviewed_classes')
                if flags is not None and (not isinstance(flags, dict) or any(type(flags.get(k)) is not bool for k in ('ball', 'person', 'table'))):
                    raise ValueError('Explicit boolean review status required for every class')
                absence = checked_interval_absence(annotation, cap, row, clip['index_sha256'], image_hash, folder / 'annotation_history')
                if absence is not None:
                    remember(folder / 'annotation_history' / (absence['previous_annotation_revision'] + '.json'), absence['previous_annotation_revision'])
            except (ValueError, TypeError, KeyError, AttributeError) as ex:
                label, exclusion = None, 'invalid_target_or_review_provenance'
                errors.append({'clip': folder.name, 'file': name, 'kind': 'shared_provenance', 'reason': str(ex)})
            else:
                # Model scopes stay independent: a malformed detector rectangle
                # does not discard valid manually reviewed temporal endpoints.
                try:
                    ball = validate_annotation(annotation, ('ball',))
                except (ValueError, TypeError, KeyError, AttributeError) as ex:
                    errors.append({'clip': folder.name, 'file': name, 'kind': 'ball', 'reason': str(ex)})
                try:
                    label = target(annotation, dimensions)
                    if label is None:
                        exclusion = 'unreviewed_unknown_occluded_or_missing_streak'
                    elif label['visibility'] == 'partial':
                        label, exclusion = None, 'partial_visible_target'
                    elif label['visibility'] == 'absent' and annotation.get('ball') != []:
                        raise ValueError('Absent target requires explicit empty ball annotations')
                    else:
                        transform_target(label, dimensions)
                except (ValueError, TypeError, KeyError, AttributeError) as ex:
                    label, exclusion = None, 'invalid_target_or_review_provenance'
                    errors.append({'clip': folder.name, 'file': name, 'kind': 'cnn', 'reason': str(ex)})
            idx = len(records); parents.append(idx)
            source_key = (original if original else folder.name, fid)
            record = {'clip': folder.name, 'file': name, 'partition': 'unassigned', 'session_id': session,
                      'source_frame': fid, 'elapsed_ms': pts, 'burst_id': row.get('burst_id'),
                      'image': image.as_posix(), 'sha256': image_hash, 'decoded_sha256': decoded_hash,
                      'source_width': dimensions[0], 'source_height': dimensions[1],
                      'target': label, 'ball': ball, 'exclusion': exclusion,
                      'annotation_sha256': hashlib.sha256(canonical(annotation).encode()).hexdigest(),
                      'source_key': source_key}
            if source_key in source_keys:
                other = records[source_keys[source_key]]
                if other['decoded_sha256'] != decoded_hash or abs(other['elapsed_ms'] - pts) > 1e-6:
                    raise ValueError('Same source frame has conflicting decoded pixels or timestamps')
                union(idx, source_keys[source_key])
            if decoded_hash in image_keys:
                union(idx, image_keys[decoded_hash])
            source_keys[source_key], image_keys[decoded_hash] = idx, idx
            records.append(record); clip['record_indices'].append(idx)
        if selected:
            clips.append(clip)
    if not clips:
        raise ValueError('No captures belong to the requested unassigned session')
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(find(i), []).append(record)
    bad, conflicts, counts, exclusions = set(), [], Counter(), Counter()
    for group_id, group in groups.items():
        labels = {canonical(row['target']): row['target'] for row in group if row['target'] is not None}
        ball_labels = {canonical(row['ball']) for row in group if row['ball'] is not None}
        if len(labels) > 1:
            bad.add(group_id)
            conflicts.append({'kind': 'cnn', 'frames': [{'clip': r['clip'], 'file': r['file']} for r in group]})
        if len(ball_labels) > 1:
            conflicts.append({'kind': 'ball', 'frames': [{'clip': r['clip'], 'file': r['file']} for r in group]})
        if len(labels) == 1:
            label = next(iter(labels.values()))
            if label['visibility'] == 'visible':
                counts['visible_point_frames' if label['half_length_px'] == 0 else 'visible_streak_frames'] += 1
            else:
                counts['absent_frames'] += 1
        else:
            counts['unknown_or_excluded_frames'] += 1
    windows, seen = [], set()
    for clip in clips:
        sequence = clip.pop('record_indices')
        for i in range(2, len(sequence)):
            indices = sequence[i-2:i+1]
            frames = [records[j] for j in indices]
            groupids = tuple(find(j) for j in indices)
            if any(g in bad for g in groupids):
                exclusions['conflicting_duplicate_target_window'] += 1; continue
            if len(set(groupids)) != 3:
                exclusions['repeated_source_or_decoded_image_window'] += 1; continue
            if any(frame['target'] is None for frame in frames):
                exclusions['window_has_unreviewed_or_excluded_target'] += 1; continue
            fids = [frame['source_frame'] for frame in frames]
            if fids != list(range(fids[0], fids[0] + 3)):
                exclusions['source_frame_gap'] += 1; continue
            burst = frames[0]['burst_id']
            if not ((type(burst) is int and burst > 0) or (isinstance(burst, str) and burst.strip())) or any(frame['burst_id'] != burst for frame in frames):
                exclusions['missing_or_changed_burst'] += 1; continue
            if len({(frame['source_width'], frame['source_height']) for frame in frames}) != 1:
                exclusions['dimension_boundary'] += 1; continue
            if groupids in seen:
                exclusions['duplicate_window'] += 1; continue
            seen.add(groupids)
            exported = [{k: copy.deepcopy(v) for k, v in frame.items() if k not in ('ball', 'exclusion', 'source_key')} for frame in frames]
            visibilities = [frame['target']['visibility'] for frame in frames]
            kind = 'all_visible' if all(v == 'visible' for v in visibilities) else 'all_absent' if all(v == 'absent' for v in visibilities) else 'mixed_visible_absent'
            counts[kind + '_windows'] += 1
            windows.append({'partition': 'unassigned', 'clip': clip['clip'], 'session_id': session_id,
                            'original_sha256': clip['original_sha256'], 'frames': exported,
                            'latest_source_frame': fids[-1], 'window_kind': kind,
                            'input_order': 'oldest_to_latest_RGB_CHW', 'prediction_time': 'latest_input_frame',
                            'label_time_support': 'all_three_reviewed_frames'})
    for key, expected in snapshots.items():
        current = Path(key).stat()
        if (current.st_size, current.st_mtime_ns) != fingerprints[key]:
            raise ValueError('Source/capture/index/annotations/image changed during staging; retry saved snapshot')
        # Recheck the small metadata files too: same-size writes can have coarse timestamps.
        if Path(key).name in ('capture.json', 'annotations.json', 'frames.jsonl') and sha(key) != expected:
            raise ValueError('Capture/index/annotations changed during staging')
    reasons = ['INDEPENDENT_VALIDATION_TEST_SESSIONS_MISSING', 'FROZEN_SPLITS_MISSING',
               'TEMPORAL_CONTINUITY_NOT_AUDITED', 'NO_ACCURACY_OR_PROMOTION_EVIDENCE']
    if not counts['all_visible_windows']:
        reasons.append('VISIBLE_REVIEWED_WINDOWS_MISSING')
    manifest = {'schema_version': SCHEMA, 'adapter': 'unassigned_reviewed_cnn_stage_v1',
                'captures_root': root.as_posix(), 'session_id': session_id, 'partition': 'unassigned',
                'ready_for_fitting': False, 'reason_codes': reasons,
                'capture_count': len(clips), 'saved_frames': len(records), 'unique_review_images': len(groups),
                'duplicate_saved_images': len(records) - len(groups),
                'counts': {k: counts[k] for k in ('visible_point_frames', 'visible_streak_frames', 'absent_frames',
                           'unknown_or_excluded_frames', 'all_visible_windows', 'all_absent_windows', 'mixed_visible_absent_windows')},
                'window_count': len(windows), 'exclusions': dict(exclusions), 'errors': errors, 'conflicts': conflicts,
                'catalog': catalog, 'capture_lineage': clips, 'source_snapshots': snapshots,
                'code_sha256': {name: sha(Path(__file__).parent / name) for name in CODE_FILES},
                'model_input_shape': [9, 288, 512], 'target_shape': [3, 288, 512],
                'target_method': TARGET_METHOD, 'input_order': 'oldest_to_latest_RGB_CHW',
                'scope': 'Immutable structural real-review handoff only. No fitting, split assignment, label rewrite, validation or default promotion.',
                'continuity': 'NOT_AUDITED: adjacent IDs/PTS/burst IDs do not prove absence of cuts, repeated footage or exposure interruption.',
                'duplicate_policy': 'Union original-source-frame and decoded-RGB identities; conflicting CNN labels excluded, repeated images within windows excluded, duplicate windows counted once.'}
    return manifest, windows


def write_stage(captures, session_id, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new staging directory; existing evidence is never overwritten')
    manifest, windows = snapshot(captures, session_id)
    body = ''.join(canonical(row) + '\n' for row in windows).encode('utf-8')
    manifest['windows_sha256'] = hashlib.sha256(body).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'windows.jsonl').open('xb') as stream:
        stream.write(body)
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return manifest


def load_stage(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA or manifest.get('partition') != 'unassigned' or manifest.get('ready_for_fitting') is not False:
        raise ValueError('Expected unassigned structural stage, not a fitting/export manifest')
    raw = (directory / 'windows.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get('windows_sha256'):
        raise ValueError('Staged windows changed after snapshot')
    fresh, rows = snapshot(manifest['captures_root'], manifest['session_id'])
    fresh['windows_sha256'] = hashlib.sha256(''.join(canonical(row) + '\n' for row in rows).encode('utf-8')).hexdigest()
    if canonical(fresh) != canonical(manifest):
        raise ValueError('Staged source/annotation/image/target/code snapshot is stale; restage saved reviews')
    return manifest, rows


class StagedReviewedDataset:
    """Strict map dataset for inspection/smoke, deliberately without split labels."""
    def __init__(self, directory, *, as_torch=False):
        self.directory = Path(directory).resolve()
        self._manifest, rows = load_stage(self.directory)
        self._rows = copy.deepcopy(rows)
        self._stage_hashes = {name: sha(self.directory / name) for name in ('manifest.json', 'windows.jsonl')}
        self._lineage = {clip['clip']: clip for clip in self._manifest['capture_lineage']}
        self._as_torch = as_torch

    def __len__(self):
        return len(self._rows)

    @property
    def rows(self):
        return copy.deepcopy(self._rows)

    @property
    def manifest(self):
        return copy.deepcopy(self._manifest)

    def sample(self, index):
        row = self._rows[index]
        for name, digest in self._stage_hashes.items():
            if sha(self.directory / name) != digest:
                raise ValueError('Staged manifest/windows changed after load')
        root = Path(self._manifest['captures_root'])
        lineage = self._lineage[row['clip']]
        for name, key in (('capture.json', 'capture_sha256'), ('annotations.json', 'annotations_sha256'), ('frames.jsonl', 'index_sha256')):
            if sha(root / row['clip'] / name) != lineage[key]:
                raise ValueError('Source review metadata changed after load')
        if lineage['original_path'] is not None and sha(lineage['original_path']) != lineage['original_sha256']:
            raise ValueError('Original recording changed after load')
        for name, digest in self._manifest['code_sha256'].items():
            if sha(Path(__file__).parent / name) != digest:
                raise ValueError('Staged adapter/validator code changed after load')
        inputs, targets, geometries = [], [], []
        for frame in row['frames']:
            interval = frame['target'].get('interval_review')
            if interval is not None:
                previous = root / row['clip'] / 'annotation_history' / (interval['previous_annotation_revision'] + '.json')
                if sha(previous) != interval['previous_annotation_revision']:
                    raise ValueError('Immutable prior absence review changed after load')
            path = Path(frame['image'])
            if sha(path) != frame['sha256']:
                raise ValueError('Source review image changed after load')
            with Image.open(path) as image:
                dimensions = image.size
                if image.format != 'JPEG' or dimensions != (frame['source_width'], frame['source_height']):
                    raise ValueError('Source image type/dimensions changed after load')
                rgb = np.array(image.convert('RGB'))
            affine = source_to_output_affine(dimensions)
            warped = cv2.warpAffine(rgb, affine, INPUT_WH, flags=cv2.INTER_LINEAR)
            chw = warped.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
            inputs.append((chw - MEAN) / STD)
            geometry = transform_target(frame['target'], dimensions)
            targets.append(line_target(geometry))
            geometries.append({'source_frame': frame['source_frame'], 'elapsed_ms': frame['elapsed_ms'],
                               'source_image_sha256': frame['sha256'], 'source_dimensions': list(dimensions),
                               'source_target': copy.deepcopy(frame['target']), 'output_target': geometry,
                               'source_to_output_affine': affine.tolist()})
        x = np.ascontiguousarray(np.concatenate(inputs))
        y = np.ascontiguousarray(np.stack(targets))
        provenance = {'partition': 'unassigned', 'session_id': row['session_id'], 'clip': row['clip'],
                      'latest_source_frame': row['latest_source_frame'], 'window_kind': row['window_kind'],
                      'frames': geometries, 'target_method': TARGET_METHOD, 'ready_for_fitting': False}
        if self._as_torch:
            import torch
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, {0: y}, provenance

    def __getitem__(self, index):
        x, y, _ = self.sample(index)
        return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = write_stage(args.captures, args.session, args.output)
    print(json.dumps({k: manifest[k] for k in ('session_id', 'partition', 'capture_count', 'saved_frames',
                       'unique_review_images', 'counts', 'window_count', 'ready_for_fitting', 'reason_codes')}, indent=2))


if __name__ == '__main__':
    main()
