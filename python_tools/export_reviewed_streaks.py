"""Audit/export explicitly reviewed three-frame blur labels without editing captures.

JSONL output is a versioned training handoff, not an upstream BlurBall dataset
adapter or a trained model. Unknown/occluded frames never become negatives.
Usage: export_reviewed_streaks.py CAPTURES SPLITS [--output NEW_DIRECTORY]
Without --output, only print the audit. One whole session stays in one split.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image
from prepare_live_training import check_capture_session
from review_absence_provenance import checked_interval_absence


def point(value):
    if not isinstance(value, list) or len(value) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in value):
        raise ValueError('Invalid normalized streak endpoint')
    return value


def target(annotation, dimensions):
    flags = annotation.get('reviewed_classes')
    if (flags.get('ball') if isinstance(flags, dict) else annotation.get('reviewed')) is not True:
        return None
    geometry = annotation.get('geometry') or {}
    visibility, streak = geometry.get('visibility', 'unknown'), geometry.get('blur_streak')
    if visibility in ('unknown', 'occluded'):
        return None
    if visibility == 'absent':
        if annotation.get('ball') or streak is not None:
            raise ValueError('Absent-ball review conflicts with an observation')
        result = {'visibility': 'absent', 'position_px': None, 'half_length_px': None, 'axis_rad': None, 'method': 'human_confirmed_absence'}
        if annotation.get('ball_absence_review') is not None:
            result['interval_review'] = dict(annotation['ball_absence_review'])
        return result
    if visibility not in ('visible', 'partial'):
        raise ValueError('Invalid ball visibility')
    if streak is None or streak.get('reviewed') is not True:
        return None
    if streak.get('method') != 'manual_streak_endpoints' or not isinstance(streak.get('endpoints'), list) or len(streak['endpoints']) != 2:
        raise ValueError('Explicit manual streak endpoints required')
    width, height = dimensions
    if type(streak.get('source_width')) is not int or type(streak.get('source_height')) is not int or (streak['source_width'], streak['source_height']) != dimensions:
        raise ValueError('Streak dimensions do not match source JPEG')
    a, b = (point(p) for p in streak['endpoints'])
    dx, dy = (b[0]-a[0])*width, (b[1]-a[1])*height
    length = math.hypot(dx, dy)
    return {'visibility': visibility, 'endpoints_normalized': [a, b],
            'position_px': [(a[0]+b[0])*width/2, (a[1]+b[1])*height/2],
            'half_length_px': length/2, 'axis_rad': None if length == 0 else math.atan2(dy, dx) % math.pi,
            'method': 'manual_streak_endpoints'}


def audit(captures, split_file):
    splits = json.loads(split_file.read_text(encoding='utf-8'))
    if set(splits) != {'train', 'val', 'test'}:
        raise ValueError('Provide train, val and test complete-recording lists')
    root = captures.resolve()
    sessions, seen, image_splits, windows, excluded, counts = {}, set(), {}, [], Counter(), Counter()
    lineage = []
    for split, clips in splits.items():
        if not isinstance(clips, list) or any(not isinstance(c, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', c) for c in clips):
            raise ValueError('Invalid capture list')
        for clip in clips:
            directory = (root/clip).resolve()
            if clip in seen or directory.parent != root:
                raise ValueError('Recording path escapes root or overlaps splits')
            seen.add(clip)
            capture_path, annotations_path, index_path = (directory/name for name in ('capture.json', 'annotations.json', 'frames.jsonl'))
            capture = json.loads(capture_path.read_text(encoding='utf-8'))
            check_capture_session(capture, split, sessions)
            # This export requires explicit session identity even for old live
            # captures; do not silently treat each camera file as a new session.
            if not isinstance(capture.get('session_id'), str) or not capture['session_id'].strip():
                raise ValueError('Every streak-training capture needs a session ID')
            annotations = json.loads(annotations_path.read_text(encoding='utf-8'))
            index_text = index_path.read_text(encoding='utf-8')
            frames = [json.loads(line) for line in index_text.split('\n')[:-1] if line]
            lineage.append({'clip': clip, 'split': split, 'session_id': capture['session_id'],
                            'annotations_sha256': hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
                            'capture_sha256': hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                            'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest()})
            processed = []
            for row in frames:
                if not re.fullmatch(r'frame_[0-9]+\.jpg', row.get('file', '')):
                    raise ValueError('Invalid source frame filename')
                image = (directory/row['file']).resolve()
                if image.parent != directory:
                    raise ValueError('Source image escapes recording')
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in image_splits and image_splits[digest] != split:
                    raise ValueError('Identical source image occurs across dataset splits')
                image_splits[digest] = split
                with Image.open(image) as source:
                    if source.format != 'JPEG':
                        raise ValueError('Source frame is not a JPEG')
                    dimensions = source.size
                annotation = annotations.get(row['file'], {})
                checked_interval_absence(annotation, capture, row, lineage[-1]['index_sha256'], digest, directory/'annotation_history')
                label = target(annotation, dimensions)
                if label is None:
                    excluded['unreviewed_unknown_occluded_or_missing_streak'] += 1
                processed.append({**row, 'image': image.as_posix(), 'sha256': digest,
                                  'source_width': dimensions[0], 'source_height': dimensions[1], 'target': label})
            for i in range(2, len(processed)):
                group = processed[i-2:i+1]
                if any(f['target'] is None for f in group):
                    excluded['window_has_unreviewed_target'] += 1
                    continue
                ids = [f.get('source_frame') for f in group]
                times = [f.get('elapsed_ms') for f in group]
                if any(type(x) is not int for x in ids) or ids != list(range(ids[0], ids[0]+3)):
                    excluded['source_frame_gap'] += 1
                    continue
                if any(type(t) not in (int, float) or not math.isfinite(t) for t in times) or not all(b > a for a, b in zip(times, times[1:])):
                    excluded['invalid_source_timestamps'] += 1
                    continue
                if any(f.get('burst_id') != group[0].get('burst_id') for f in group) or any((f['source_width'], f['source_height']) != (group[0]['source_width'], group[0]['source_height']) for f in group):
                    excluded['burst_or_dimension_boundary'] += 1
                    continue
                windows.append({'split': split, 'clip': clip, 'session_id': capture['session_id'], 'frames': group,
                                'latest_source_frame': ids[-1], 'label_time_support': 'all_three_reviewed_frames',
                                'input_order': 'oldest_to_latest', 'prediction_time': 'latest_input_frame'})
                counts[split] += 1
    return {'schema_version': 1, 'source_splits': splits, 'split_manifest_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
            'accepted_counts': {split: counts[split] for split in splits}, 'excluded': dict(excluded),
            'ready_for_training': all(counts[split] > 0 for split in splits), 'capture_lineage': lineage, 'windows': windows,
            'scope': 'Explicitly reviewed causal blur-training handoff only; counts do not establish accuracy or sufficient sample size.',
            'label_convention': 'Blur-streak center and half-length in original source pixels; undirected axis, null for zero length. Unknowns excluded.'}


def export(captures, split_file, output):
    result = audit(captures, split_file)
    if not result['ready_for_training']:
        raise ValueError('No complete reviewed three-frame windows in every independent split; do not train from unknown labels')
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val', 'test'):
        with (output/(split+'.jsonl')).open('x', encoding='utf-8') as destination:
            for row in result['windows']:
                if row['split'] == split:
                    destination.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = {k: v for k, v in result.items() if k != 'windows'}
    (output/'audit.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path); parser.add_argument('splits', type=Path)
    parser.add_argument('--output', type=Path, help='New directory only; omit for read-only audit')
    args = parser.parse_args()
    result = export(args.captures, args.splits, args.output) if args.output else audit(args.captures, args.splits)
    print(json.dumps({k: v for k, v in result.items() if k != 'windows'}, indent=2))
