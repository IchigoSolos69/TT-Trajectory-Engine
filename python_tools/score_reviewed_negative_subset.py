"""Score only saved, reviewed ball-absent images against an audited paired journal.

Unknown, positive and conflicting reviews are excluded, never relabeled. This is
a sampled-negative diagnostic, not full-video accuracy or a training/promotion
decision. No decoder, GPU inference or dataset split assignment is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from score_paired_ball_candidates import (HYPOTHESES, audit_journal, finite,
                                         integer, load_json, require,
                                         sha256_bytes, sha256_file)
from prepare_live_training import validate_annotation
from export_reviewed_streaks import target
from review_absence_provenance import checked_interval_absence


def watch(path, watched):
    path = Path(path).resolve()
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if path in watched:
        require(watched[path] == digest, f'File changed during review snapshot: {path}')
    watched[path] = digest
    return data, digest


def verify_snapshot(watched):
    for path, digest in watched.items():
        require(sha256_file(path) == digest,
                f'Source/model/journal/review changed during scoring; retry saved snapshot: {path}')


def watch_digest(path, watched):
    """Stream large original recordings rather than retaining their bytes."""
    path = Path(path).resolve()
    digest = sha256_file(path)
    if path in watched:
        require(watched[path] == digest, f'File changed during review snapshot: {path}')
    watched[path] = digest
    return digest


def reviewed_time(value):
    require(isinstance(value, str) and value.strip(), 'Saved ball review needs reviewed_at')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Saved review timestamp is malformed') from error
    require(parsed.tzinfo is not None, 'Saved review timestamp must include a timezone')


def checked_capture(folder, manifest, rows, watched):
    """Reuse annotation validators without requiring or inferring dataset splits."""
    folder = Path(folder).resolve()
    metadata = {}
    hashes = {}
    for name in ('capture.json', 'annotations.json', 'frames.jsonl'):
        data, hashes[name] = watch(folder / name, watched)
        if name == 'frames.jsonl':
            require(data and data.endswith(b'\n'), 'Complete newline-terminated capture index required')
            lines = data.splitlines()
            require(all(line.strip() for line in lines), 'Blank capture index row')
            metadata[name] = [load_json(line, f'{folder.name} capture row') for line in lines]
        else:
            metadata[name] = load_json(data, f'{folder.name}/{name}')
    capture, annotations, index = (metadata[name] for name in ('capture.json', 'annotations.json', 'frames.jsonl'))
    require(isinstance(capture, dict) and isinstance(annotations, dict), 'Capture/reviews must be objects')
    require(capture.get('source') == 'recorded_camera', 'Subset scorer requires original recorded-camera provenance')
    require(capture.get('original_sha256') == manifest['source_sha256'], 'Capture/journal original source identity differs')
    require(capture.get('source_frames') == manifest['source_declared_frames'] and
            capture.get('sampled_frames') == len(index), 'Capture/source index counts differ')
    require(capture.get('timestamp_basis') == 'opencv_ffmpeg_source_pts_ms', 'Source PTS basis is not established')
    require(isinstance(capture.get('session_id'), str) and capture['session_id'].strip(), 'Capture session identity missing')
    original = capture.get('original_path')
    require(isinstance(original, str) and original, 'Capture original recording path missing')
    original_hash = watch_digest(original, watched)
    require(original_hash == manifest['source_sha256'], 'Original recording bytes changed')
    selection = capture.get('selection')
    require(isinstance(selection, dict), 'Capture selection provenance missing')
    selection_method = selection.get('method')
    require(selection_method in ('uniform', 'review_selection'), 'Unknown capture selection method')
    selection_hash = None
    if selection_method == 'review_selection':
        require(isinstance(selection.get('selection_path'), str), 'Hard-mined selection path missing')
        raw, selection_hash = watch(selection['selection_path'], watched)
        require(selection_hash == selection.get('selection_sha256'), 'Hard-mined selection file changed')
        picked = load_json(raw, 'hard-mined selection')
        require(isinstance(picked, dict) and isinstance(picked.get('identity'), dict) and
                picked['identity'].get('source_sha256') == manifest['source_sha256'],
                'Hard-mined selection source identity differs')
        require(selection.get('identity') == picked['identity'] and selection.get('bursts') == picked.get('bursts'),
                'Capture selection provenance differs from retained selection file')
    names, ids, records = set(), set(), []
    previous_id, previous_pts = 0, -1.
    for item in index:
        require(isinstance(item, dict), 'Capture index row must be an object')
        name, fid, elapsed = item.get('file'), item.get('source_frame'), item.get('elapsed_ms')
        require(isinstance(name, str) and re.fullmatch(r'frame_[0-9]+\.jpg', name) and name not in names,
                'Capture image names must be unique safe frame filenames')
        require(integer(fid, 1) and fid not in ids and fid > previous_id and fid <= len(rows),
                'Capture source frame must be unique chronological and present in journal')
        require(finite(elapsed) and elapsed >= 0 and elapsed > previous_pts,
                'Capture source PTS must be finite and strictly chronological')
        paired = rows[fid - 1]
        require(abs(elapsed / 1000. - paired['source_pts_s']) <= 1e-6,
                'Saved review source PTS differs from actual journal PTS')
        names.add(name); ids.add(fid); previous_id, previous_pts = fid, elapsed
        image_path = (folder / name).resolve()
        require(image_path.parent == folder, 'Review image escapes capture folder')
        _, image_hash = watch(image_path, watched)
        with Image.open(image_path) as image:
            require(image.format == 'JPEG', 'Review image must be JPEG')
            rgb = image.convert('RGB')
            dimensions = rgb.size
            decoded_hash = hashlib.sha256(str(dimensions).encode() + rgb.tobytes()).hexdigest()
        require(list(dimensions) == paired['source_size'], 'Review image/journal source dimensions differ')
        annotation = annotations.get(name, {})
        require(isinstance(annotation, dict), 'Saved annotation must be an object')
        geometry = annotation.get('geometry') or {}
        require(isinstance(geometry, dict), 'Annotation geometry must be an object')
        flags = annotation.get('reviewed_classes')
        if flags is not None:
            require(isinstance(flags, dict) and all(type(flags.get(k)) is bool for k in ('ball', 'person', 'table')),
                    'Explicit per-class review flags must be boolean')
        is_saved = annotation.get('reviewed') is True and isinstance(flags, dict) and flags.get('ball') is True
        if is_saved:
            require(annotation.get('source_frame') == fid and type(annotation.get('source_frame')) is int,
                    'Saved review source frame differs from index')
            require(finite(annotation.get('elapsed_ms')) and abs(annotation['elapsed_ms'] - elapsed) <= 1e-6,
                    'Saved review timestamp differs from index')
            reviewed_time(annotation.get('reviewed_at'))
        provenance = checked_interval_absence(annotation, capture, item, hashes['frames.jsonl'],
                                               image_hash, folder / 'annotation_history')
        if provenance:
            watch(folder / 'annotation_history' / (provenance['previous_annotation_revision'] + '.json'), watched)
        label = validate_annotation(annotation, ('ball',))
        # The shared CNN validator guards absence/streak conflicts too; a visible
        # box without a streak remains positive evidence, not a CNN target here.
        cnn_label = target(annotation, dimensions)
        status, reason = 'excluded', 'NOT_EXPLICITLY_SAVED_BALL_REVIEW'
        canonical_label = None
        if is_saved:
            visibility = geometry.get('visibility', 'unknown')
            if label is not None and label['ball']:
                status, reason = 'positive', 'REVIEWED_POSITIVE_EXCLUDED'
                canonical_label = {'ball': label['ball']}
            elif (label is not None and label['ball'] == [] and visibility == 'absent' and
                  geometry.get('blur_streak') is None and cnn_label is not None and cnn_label['visibility'] == 'absent'):
                status, reason = 'absent', None
                canonical_label = {'ball': []}
            else:
                reason = 'UNKNOWN_OCCLUDED_OR_UNSUPPORTED_REVIEW_EXCLUDED'
        records.append({'capture_id': folder.name, 'file': name, 'source_sha256': manifest['source_sha256'],
                        'source_frame_id': fid, 'source_pts_s': paired['source_pts_s'],
                        'review_elapsed_ms': elapsed, 'review_image_sha256': image_hash,
                        'review_decoded_rgb_sha256': decoded_hash,
                        'journal_decoded_image_sha256': paired['decoded_image_sha256'],
                        'review_annotations_sha256': hashes['annotations.json'],
                        'reviewed_at': annotation.get('reviewed_at'), 'status': status, 'reason': reason,
                        'label': canonical_label, 'interval_absence_provenance': provenance,
                        'hypotheses': {key: paired[key] is not None for key in HYPOTHESES}})
    require(not (set(annotations) - names), 'Saved review references an unknown capture image')
    return records, {'capture_id': folder.name, 'capture_directory': str(folder),
                     'session_id': capture['session_id'], 'selection_method': selection_method,
                     'sampling': capture.get('sampling'), 'saved_images': len(index),
                     'file_sha256': hashes, 'selection_sha256': selection_hash,
                     'selection_bias': ('Evenly spaced bursts; consecutive images are correlated and not independent trials'
                                        if selection_method == 'uniform' else
                                        'Hard-mined using detector hypotheses; enriched failures, not a representative error rate')}


def negative_counts(records):
    result = {}
    for key in HYPOTHESES:
        fp = sum(record['hypotheses'][key] for record in records)
        count = len(records)
        result[key] = {'reviewed_absent_frames': count, 'confirmed_absent_frame_false_positives': fp,
                       'true_negatives': count - fp,
                       'sampled_negative_false_positive_rate': fp / count if count else None,
                       'sampled_negative_true_negative_rate': (count - fp) / count if count else None,
                       'precision': None, 'recall': None,
                       'ball_miss_rate': None,
                       'ball_miss_rate_reason': 'No positive reference denominator in this negative-only subset'}
    return result


def score_subset(journal_dir, captures_root, capture_ids, runtime_binary=None):
    rows, manifest, journal_audit = audit_journal(journal_dir, runtime_binary)
    require(manifest['source_interval'] == 'COMPLETE_RECORDING', 'Reviewed subset requires a complete original journal')
    require(all(value['available'] and value['sha256_match'] is True
                for value in journal_audit['source_and_model_files'].values()),
            'Fresh original source and model bytes are required for subset scoring')
    root = Path(captures_root).resolve()
    require(root.is_dir() and isinstance(capture_ids, (tuple, list)) and capture_ids,
            'Capture directory and explicit capture IDs required')
    require(len(set(capture_ids)) == len(capture_ids) and all(isinstance(name, str) and
            re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in capture_ids), 'Invalid/repeated capture ID')
    watched = {}
    for name, digest in journal_audit['file_sha256'].items():
        path = (Path(journal_dir) / name).resolve()
        require(sha256_file(path) == digest, 'Paired journal changed after strict audit')
        watched[path] = digest
    for name, value in journal_audit['source_and_model_files'].items():
        watched[Path(value['path']).resolve()] = manifest[name + '_sha256']
    if runtime_binary is not None:
        watched[Path(runtime_binary).resolve()] = manifest['runtime_sha256']
    records, lineage = [], []
    for name in capture_ids:
        folder = (root / name).resolve()
        require(folder.parent == root, 'Capture folder escapes capture root')
        items, provenance = checked_capture(folder, manifest, rows, watched)
        records.extend(items); lineage.append(provenance)
    require(len({capture['session_id'] for capture in lineage}) == 1,
            'The same original recording cannot have conflicting capture-session identities')
    groups = defaultdict(list)
    for record in records:
        groups[(record['source_sha256'], record['source_frame_id'])].append(record)
    conflicts = []
    for (_, fid), group in groups.items():
        require(len({record['review_decoded_rgb_sha256'] for record in group}) == 1,
                f'Duplicate source frame {fid} has different review image pixels')
        labels = {json.dumps(record['label'], sort_keys=True, separators=(',', ':'))
                  for record in group if record['label'] is not None}
        if len(labels) > 1:
            conflicts.append({'source_frame_id': fid, 'captures': [record['capture_id'] for record in group],
                              'reviewed_label_kinds': sorted({'positive' if record['label']['ball'] else 'absent'
                                                              for record in group if record['label'] is not None}),
                              'reason': 'CONFLICTING_DUPLICATE_REVIEWED_BALL_LABELS'})
            for record in group:
                record['reason'] = 'CONFLICTING_DUPLICATE_REVIEWED_BALL_LABELS'
                record['status'] = 'conflicting'
    eligible = [record for record in records if record['status'] == 'absent']
    unique = {record['source_frame_id']: record for record in eligible}
    captures = []
    for provenance in lineage:
        selected = [record for record in records if record['capture_id'] == provenance['capture_id']]
        negatives = [record for record in selected if record['status'] == 'absent']
        captures.append({**provenance, 'reviewed_absent_scored': len(negatives),
                         'review_status_counts': dict(Counter(record['status'] for record in selected)),
                         'exclusion_reasons': dict(Counter(record['reason'] for record in selected if record['reason'])),
                         'negative_subset_metrics': negative_counts(negatives)})
    verify_snapshot(watched)
    return {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'Explicitly saved reviewed absent-frame subset only; sampled-negative diagnostics',
            'source_sha256': manifest['source_sha256'], 'journal_directory': str(Path(journal_dir).resolve()),
            'source_frames_in_journal': len(rows), 'capture_ids': list(capture_ids),
            'journal_audit': journal_audit, 'selector_configuration': manifest['fusion'],
            'model_sha256': {name: manifest[name + '_sha256'] for name in ('yolo_model', 'blurball_model')},
            'capture_reports': captures, 'sampled_review_records': len(records),
            'unique_sampled_source_frames': len(groups), 'duplicate_sample_records': len(records) - len(groups),
            'reviewed_positive_annotation_rows_excluded': sum(record['label'] is not None and
                                                               bool(record['label']['ball']) for record in records),
            'reviewed_absent_annotation_rows_before_deduplication': sum(record['label'] == {'ball': []} for record in records),
            'reviewed_absent_unique_frames_scored': len(unique),
            'unsampled_source_frames_unknown': len(rows) - len(groups),
            'sampled_source_frames_excluded': len(groups) - len(unique),
            'deduplicated_union_metrics': negative_counts(list(unique.values())),
            'conflicting_source_frames': conflicts,
            'review_records': records,
            'snapshot_file_sha256': {str(path): digest for path, digest in sorted(watched.items())},
            'snapshot_reverified_before_report': True,
            'review_jpeg_matches_original_decoded_pixels_independently_verified': False,
            'image_origin_limitation': 'JPEG bytes/dimensions and saved source ID/PTS are audited; original pixels were not decoded again and JPEG pixels differ from raw journal BGR hashes',
            'selection_bias': 'Union mixes uniform and hard-mined bursts, overlaps are counted once; no extrapolation to the whole recording or independent sessions',
            'full_video_false_positive_count': None, 'full_video_accuracy': None,
            'global_precision': None, 'global_recall': None,
            'thresholds_fitted': False, 'promotion_approved': False, 'splits_assigned': False,
            'remaining_actions': ['Resolve duplicate-box disagreements through explicit human review; retain valid positive annotations',
                                  'Review actual visible-ball/streak examples for positive accuracy and temporal targets',
                                  'Record independent complete sessions for frozen validation and test sets']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal_directory', type=Path)
    parser.add_argument('captures_root', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('--capture', action='append', required=True, dest='captures')
    parser.add_argument('--runtime-binary', type=Path)
    args = parser.parse_args()
    try:
        report = score_subset(args.journal_directory, args.captures_root, args.captures, args.runtime_binary)
        with args.output_json.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ('reviewed_absent_unique_frames_scored',
                     'duplicate_sample_records', 'sampled_source_frames_excluded',
                     'deduplicated_union_metrics', 'promotion_approved')}, indent=2))


if __name__ == '__main__':
    main()
