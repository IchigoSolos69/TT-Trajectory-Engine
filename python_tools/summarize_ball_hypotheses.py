"""Audit a completed causal journal and select unlabelled review bursts.

Counts accepted model hypotheses, not correct detections. A user declaration
that a recording contains no ball is not reviewed ground truth. This utility
does not decode images, import frames, assign labels, or change any model.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct


SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite journal data')
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _json(data):
    result = json.loads(data, object_pairs_hook=_pairs)
    _finite(result)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def _int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid ' + name)
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('Invalid ' + name)
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError('Invalid ' + name) from error
    if not math.isfinite(number):
        raise ValueError('Invalid ' + name)
    return number


def _hash(value, name):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError('Invalid ' + name)
    return value


def audit_journal(directory, model_path, source_path=None, runtime_path=None):
    """Require a complete, immutable source/model-matching causal replay."""
    directory = Path(directory)
    paths = {name: directory / name for name in
             ('manifest.json', 'completion.json', 'frames.jsonl', 'frames.idx')}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    manifest = _json(paths['manifest.json'].read_bytes())
    completion = _json(paths['completion.json'].read_bytes())
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Unsupported journal schema')
    total = _int(manifest.get('total_frames'), 'total frame count')
    if manifest.get('source_interval') != 'COMPLETE_RECORDING' or _int(manifest.get('source_declared_frames'), 'declared source frames') != total:
        raise ValueError('Only a complete source recording can be summarized')
    if (completion != {'status': 'COMPLETE', 'analyzed_frames': total, 'total_frames': total}
            or type(completion.get('analyzed_frames')) is not int or type(completion.get('total_frames')) is not int):
        raise ValueError('Completed manifest and frame counts disagree')
    if manifest.get('evaluation_policy') != 'CAUSAL_SOURCE_REPLAY_NOT_LIVE_LATENCY':
        raise ValueError('Expected causal source replay policy')
    fps = _number(manifest.get('source_fps'), 'source FPS')
    if fps <= 0:
        raise ValueError('Source FPS must be positive')
    source_hash = _hash(manifest.get('source_sha256'), 'source SHA256')
    model_hash = _hash(manifest.get('ball_model_sha256'), 'model SHA256')
    runtime_hash = _hash(manifest.get('runtime_sha256'), 'runtime SHA256')
    source_name = manifest.get('source')
    if not isinstance(source_name, str) or not source_name:
        raise ValueError('Missing source identity')
    source_path = Path(source_path) if source_path else Path(source_name)
    if sha256_file(source_path) != source_hash:
        raise ValueError('Original source hash mismatch')
    if sha256_file(model_path) != model_hash:
        raise ValueError('Actual model hash mismatch')
    if runtime_path and sha256_file(runtime_path) != runtime_hash:
        raise ValueError('Actual runtime hash mismatch')
    model = manifest.get('ball_model')
    if not isinstance(model, dict) or model.get('model_sha256') != model_hash:
        raise ValueError('Manifest model provenance mismatch')
    if type(model.get('lookahead_frames')) is not int or model['lookahead_frames'] != 0 or model.get('temporal_policy') != 'causal_channel_2':
        raise ValueError('Noncausal or unsupported model policy')
    if model.get('timestamp_basis') != 'nominal_source_fps_or_live_capture_clock':
        raise ValueError('Unsupported timestamp basis')
    warmup = _int(model.get('warmup_frames'), 'warmup frames', 0)
    if warmup >= total:
        raise ValueError('No eligible frames after warmup')
    index = paths['frames.idx'].read_bytes()
    if len(index) != total * 8:
        raise ValueError('Incomplete or excessive binary frame index')
    offsets = [item[0] for item in struct.iter_unpack('<Q', index)]
    rows = []
    offset = 0
    dimensions = None
    with paths['frames.jsonl'].open('rb') as stream:
        for line in stream:
            frame = len(rows) + 1
            if frame > total or offsets[frame - 1] != offset:
                raise ValueError('Journal index offset mismatch')
            if not line.endswith(b'\n') or not line.strip():
                raise ValueError('Uncommitted or empty journal row')
            row = _json(line)
            if type(row.get('frame_id')) is not int or row['frame_id'] != frame:
                raise ValueError('Missing, duplicate, or reordered source frame')
            timestamp = _number(row.get('source_time'), 'source timestamp')
            # Native serializer writes six decimal places. This checks nominal
            # FPS chronology, not independent decoder PTS/cut correctness.
            if abs(timestamp - (frame - 1) / fps) > 0.000002:
                raise ValueError('Source timestamp does not match nominal frame')
            size = (_int(row.get('source_width'), 'source width'),
                    _int(row.get('source_height'), 'source height'))
            if dimensions is None:
                dimensions = size
            elif dimensions != size:
                raise ValueError('Source dimensions changed inside recording')
            if row.get('ball_model') != model:
                raise ValueError('Frame model provenance mismatch')
            for key in ('source', 'source_sha256', 'ball_model_sha256'):
                if key in row and row[key] != manifest[key]:
                    raise ValueError('Frame source/model identity mismatch')
            if 'ball' not in row:
                raise ValueError('Missing explicit ball observation')
            ball = row['ball']
            if ball is not None:
                if not isinstance(ball, dict):
                    raise ValueError('Invalid ball observation')
                for key in ('x', 'y', 'confidence'):
                    value = _number(ball.get(key), 'ball ' + key)
                    if not 0 <= value <= 1:
                        raise ValueError('Ball normalized coordinate/score outside [0,1]')
                if frame <= warmup:
                    raise ValueError('Accepted hypothesis during model warmup')
            rows.append(row)
            offset += len(line)
    if len(rows) != total:
        raise ValueError('Journal missing completed source frames')
    if any(sha256_file(paths[name]) != digest for name, digest in hashes.items()):
        raise ValueError('Journal changed while auditing')
    evidence = {
        'journal_directory': str(directory.resolve()),
        'source': source_name, 'source_sha256': source_hash,
        'source_frame_count': total, 'source_fps': fps,
        'source_hash_verified': True, 'ball_model_sha256': model_hash,
        'model_hash_verified': True, 'runtime_sha256': runtime_hash,
        'runtime_hash_verified': bool(runtime_path),
        'journal_sha256': hashes['frames.jsonl'], 'index_sha256': hashes['frames.idx'],
        'manifest_sha256': hashes['manifest.json'], 'completion_sha256': hashes['completion.json'],
        'model_provenance': model,
    }
    return manifest, rows, evidence


def observation_runs(rows, fps):
    """Consecutive accepted observations; these are NOT identity tracks."""
    runs = []
    start = None
    for index, row in enumerate(rows + [{'ball': None}]):
        if row['ball'] is not None and start is None:
            start = index
        if row['ball'] is None and start is not None:
            end = index - 1
            runs.append({
                'start_source_frame': start + 1, 'end_source_frame': end + 1,
                'observations': end - start + 1,
                'start_timestamp_s': rows[start]['source_time'],
                'end_timestamp_s': rows[end]['source_time'],
                'timestamp_span_s': rows[end]['source_time'] - rows[start]['source_time'],
                'nominal_frame_coverage_s': (end - start + 1) / fps,
            })
            start = None
    return runs


def select_bursts(rows, runs, max_bursts=24, columns=4, spatial_rows=3, temporal_bins=12):
    """Greedy, deterministic temporal/grid diversity, without embeddings."""
    _int(max_bursts, 'maximum review bursts')
    if max_bursts > 1000:
        raise ValueError('Maximum review bursts exceeds bounded limit (1000)')
    for value, name in ((columns, 'grid columns'), (spatial_rows, 'grid rows'), (temporal_bins, 'temporal bins')):
        if _int(value, name) > 1000:
            raise ValueError('Excessive ' + name)
    total = len(rows)
    if total < 5:
        return []
    run_lengths = {frame: run['observations'] for run in runs
                   for frame in range(run['start_source_frame'], run['end_source_frame'] + 1)}
    candidates = {}
    for row in rows:
        ball = row['ball']
        if ball is None:
            continue
        frame = row['frame_id']
        start = min(max(1, frame - 2), total - 4)
        cell = (min(columns - 1, int(ball['x'] * columns)),
                min(spatial_rows - 1, int(ball['y'] * spatial_rows)))
        temporal = min(temporal_bins - 1, (frame - 1) * temporal_bins // total)
        accepted = sum(item['ball'] is not None for item in rows[start - 1:start + 4])
        candidate = {
            'start_source_frame': start, 'end_source_frame': start + 4,
            'anchor_source_frame': frame, 'anchor_timestamp_s': row['source_time'],
            'anchor_center_normalized': [ball['x'], ball['y']],
            'anchor_uncalibrated_model_score': ball['confidence'],
            'accepted_hypotheses_in_burst': accepted,
            'anchor_observation_run_length': run_lengths[frame],
            'spatial_bin': {'column': cell[0], 'row': cell[1]},
            'temporal_bin': temporal,
            'reason': 'ACCEPTED_BALL_HYPOTHESIS_IN_USER_DECLARED_NO_BALL_RECORDING',
        }
        # Several anchors can share a burst. Keep the highest evidence density,
        # then longer consecutive run, then earliest anchor; score is not truth.
        previous = candidates.get(start)
        rank = (accepted, run_lengths[frame], -frame)
        if previous is None or rank > (previous['accepted_hypotheses_in_burst'],
                                       previous['anchor_observation_run_length'], -previous['anchor_source_frame']):
            candidates[start] = candidate
    selected = []
    seen_cells, seen_temporal, seen_pairs = set(), set(), set()
    remaining = list(candidates.values())
    while remaining and len(selected) < max_bursts:
        eligible = [candidate for candidate in remaining if all(
            candidate['end_source_frame'] < prior['start_source_frame'] or
            candidate['start_source_frame'] > prior['end_source_frame'] for prior in selected)]
        if not eligible:
            break
        def rank(candidate):
            cell = (candidate['spatial_bin']['column'], candidate['spatial_bin']['row'])
            temporal = candidate['temporal_bin']
            return ((cell, temporal) not in seen_pairs, temporal not in seen_temporal,
                    cell not in seen_cells, candidate['accepted_hypotheses_in_burst'],
                    candidate['anchor_observation_run_length'], -candidate['anchor_source_frame'])
        chosen = max(eligible, key=rank)
        selected.append(chosen)
        cell = (chosen['spatial_bin']['column'], chosen['spatial_bin']['row'])
        seen_cells.add(cell); seen_temporal.add(chosen['temporal_bin'])
        seen_pairs.add((cell, chosen['temporal_bin']))
        remaining.remove(chosen)
    return sorted(selected, key=lambda item: item['start_source_frame'])


def summarize(directory, model_path, source_path=None, runtime_path=None, max_bursts=24,
              session_id=None):
    manifest, rows, evidence = audit_journal(directory, model_path, source_path, runtime_path)
    total, fps = len(rows), manifest['source_fps']
    warmup = manifest['ball_model']['warmup_frames']
    observed = [row for row in rows if row['ball'] is not None]
    runs = observation_runs(rows, fps)
    columns, grid_rows = 4, 3
    bins = Counter((min(columns - 1, int(row['ball']['x'] * columns)),
                    min(grid_rows - 1, int(row['ball']['y'] * grid_rows))) for row in observed)
    truth = {'declaration': 'USER_DECLARED_NO_BALL', 'review_status': 'UNREVIEWED',
             'ground_truth_available': False,
             'reason': 'User declaration has not been confirmed through reviewed source-frame annotations; model output never becomes ground truth.'}
    report = {
        'schema_version': 1, 'scope': 'Completed offline causal replay hypothesis diagnostic; not live latency or detection accuracy.',
        'identity': evidence, 'session_id': session_id, 'truth': truth,
        'coverage': {'source_frames': total, 'accounted_frames': total, 'contiguous_ids': True,
                     'index_offsets_verified': True, 'warmup_frames': warmup,
                     'eligible_frames': total - warmup, 'source_fps': fps,
                     'nominal_frame_duration_s': total / fps,
                     'timestamp_basis': manifest['ball_model']['timestamp_basis'],
                     'independent_decoder_pts_or_cut_audit': False},
        'accepted_hypotheses': {'count': len(observed),
                                'fraction_of_all_frames': len(observed) / total,
                                'fraction_of_eligible_frames': len(observed) / (total - warmup),
                                'nominal_observed_frame_coverage_s': len(observed) / fps},
        'accuracy': {'precision': None, 'recall': None, 'confirmed_false_positive_frames': None,
                     'confirmed_false_positive_rate': None, 'reason': truth['reason']},
        'observation_runs': {'definition': 'Consecutive source frames with accepted ball hypotheses; no identity continuity is implied.',
                             'count': len(runs), 'longest_run_frames': max((run['observations'] for run in runs), default=0),
                             'runs': runs},
        'spatial_histogram': {'basis': 'Normalized source-image hypothesis centers',
                              'columns': columns, 'rows': grid_rows,
                              'bins': [{'column': column, 'row': row, 'count': bins[(column, row)]}
                                       for row in range(grid_rows) for column in range(columns)]},
        'limitations': ['No annotations were created or changed.',
                        'Single same-venue gameplay session; not independent validation or test evidence.',
                        'Coarse temporal/spatial diversity is not visual embedding diversity.',
                        'Model scores are uncalibrated; a hypothesis is not a probability of correctness.'],
    }
    bursts = select_bursts(rows, runs, max_bursts, columns, grid_rows)
    selection = {
        'schema_version': 1, 'identity': evidence, 'session_id': session_id, 'truth': truth,
        'selection_method': 'deterministic_greedy_temporal_spatial_grid_v1',
        'selection_rule': 'Prefer unseen spatial/temporal pairs, then unseen temporal bins, then unseen spatial bins, then denser five-frame hypothesis windows, longer observation runs, earlier anchors.',
        'spatial_grid': [columns, grid_rows], 'temporal_bins': 12,
        'maximum_bursts': max_bursts, 'burst_frames': 5, 'selected_bursts': len(bursts),
        'selected_source_frames': len(bursts) * 5, 'nonoverlapping': True,
        'images_imported': False, 'automatic_labels_created': False,
        'use': 'Review selection only. Confirm visible/absent/occluded status before training; keep this session in one dataset split.',
        'bursts': bursts,
    }
    return report, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--session-id')
    parser.add_argument('--max-bursts', type=int, default=24)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve() == args.selection.resolve() or args.report.exists() or args.selection.exists():
        parser.error('Use two distinct new output paths; existing evidence is never overwritten')
    try:
        report, selection = summarize(args.journal, args.model, args.source, args.runtime,
                                      args.max_bursts, args.session_id)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Hypothesis audit failed: ' + str(error) + '\n')
    for path, value in ((args.report, report), (args.selection, selection)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'accepted_hypotheses': report['accepted_hypotheses'],
                      'observation_runs': report['observation_runs']['count'],
                      'selected_bursts': selection['selected_bursts'],
                      'truth_status': 'UNREVIEWED', 'accuracy_available': False}))


if __name__ == '__main__':
    main()
