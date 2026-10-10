"""Audit a completed native paired-candidate journal and optionally score explicit labels.

No detector is run, threshold is fitted, coordinate is blended, or model promoted.
Hashes establish file/identity consistency; they do not establish candidate truth.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct

try:
    from .score_blurball_comparison import metrics
except ImportError:
    from score_blurball_comparison import metrics


REVISION = 'conservative-source-candidates-v1'
HYPOTHESES = ('standalone_yolo', 'standalone_blurball_causal', 'experimental_selector')
IDENTITY_FIELDS = ('source_sha256', 'decoded_image_sha256', 'frame_id',
                   'source_pts_s', 'source_size')
CONFIG_POLICIES = {
    'revision': REVISION,
    'experimental': True,
    'thresholds_fitted': False,
    'coordinate_policy': 'CHOOSE_EXISTING_BLURBALL_CENTER_OR_YOLO_BOX_CENTER_NO_BLEND',
    'score_policy': 'RAW_MODEL_EVIDENCE_NEVER_AVERAGED_OR_PROBABILITY_CALIBRATED',
    'warmup_policy': 'ABSTAIN_UNTIL_THREE_REAL_CONSECUTIVE_SOURCE_FRAMES',
}
CONFIG_NUMBERS = ('yolo_proposal_score', 'yolo_agreement_score', 'yolo_solo_score',
                  'blurball_agreement_peak', 'blurball_solo_peak',
                  'agreement_radius_source_px', 'within_model_ambiguity_ratio')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def f32(value):
    """Round each arithmetic step as native Detection/cv::Point2f float32."""
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error):
        return math.copysign(math.inf, value)


def native_distance(first, second):
    # Point2f subtraction happens before cv::norm promotes products to double.
    dx, dy = (f32(first[i] - second[i]) for i in (0, 1))
    return math.sqrt(dx * dx + dy * dy)


def yolo_center(raw, size):
    return [f32(f32(raw[axis] + f32(raw[extent] * .5)) * f32(dimension))
            for axis, extent, dimension in zip(('x', 'y'), ('w', 'h'), size)]


def cpp_canonical_json(value):
    """Native sorted Object JSON with double max_digits10, not Python short floats.

    All config numbers pass through C++ number(double), including integral-looking
    radii. JSON strings use the same escaping as recording::quote for this schema.
    """
    if value is None:
        return 'null'
    if type(value) is bool:
        return 'true' if value else 'false'
    if type(value) in (int, float):
        require(finite(value), 'Canonical JSON number must be finite')
        return format(value, '.17g')
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return '[' + ','.join(cpp_canonical_json(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{' + ','.join(cpp_canonical_json(key) + ':' + cpp_canonical_json(value[key])
                              for key in sorted(value)) + '}'
    raise ValueError('Unsupported canonical JSON value')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON field {key}')
        result[key] = value
    return result


def _nonfinite_constant(value):
    raise ValueError(f'Nonfinite JSON constant {value}')


def load_json(data, context):
    try:
        result = json.loads(data, object_pairs_hook=_unique_object,
                            parse_constant=_nonfinite_constant)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f'{context}: {error}') from error
    require(isinstance(result, dict), f'{context}: expected JSON object')

    def finite_numbers(value):
        if isinstance(value, dict):
            return all(finite_numbers(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_numbers(v) for v in value)
        return type(value) is not float or math.isfinite(value)

    require(finite_numbers(result), f'{context}: nonfinite JSON number')
    return result


def expect_fields(obj, fields, context):
    require(isinstance(obj, dict), f'{context}: expected object')
    require(set(fields) <= set(obj), f'{context}: missing fields {sorted(set(fields) - set(obj))}')


def expect_value(obj, key, expected, context):
    require(isinstance(obj, dict) and key in obj and type(obj[key]) is type(expected) and obj[key] == expected,
            f'{context}: {key} does not match {expected!r}')


def point(value, finite_required=True):
    return (isinstance(value, list) and len(value) == 2 and
            all(finite(v) if finite_required else v is None or finite(v) for v in value))


def invalid_geometry(candidate):
    """Mirror invalidGeometry in ball_candidate_selection.hpp, in its order."""
    raw, center = candidate['raw_detection'], candidate['center_px']
    if raw['class_id'] != 0:
        return 'NON_BALL_CLASS'
    if not finite(raw['score']) or not 0 <= raw['score'] <= 1:
        return 'INVALID_RAW_SCORE'
    if not all(finite(raw[key]) for key in ('x', 'y', 'w', 'h')) or raw['w'] <= 0 or raw['h'] <= 0:
        return 'INVALID_BOX'
    if not point(center):
        return 'NONFINITE_CENTER'
    if not all(0 <= c < size for c, size in zip(center, candidate['source_size'])):
        return 'CENTER_OUTSIDE_SOURCE'
    if candidate['model'] == 'yolo':
        if center != yolo_center(raw, candidate['source_size']):
            return 'YOLO_CENTER_MISMATCH'
        if candidate['blur_evidence'] is not None:
            return 'UNEXPECTED_BLUR_EVIDENCE'
    else:
        blur = candidate['blur_evidence']
        if blur is None:
            return 'MISSING_BLURBALL_EVIDENCE'
        if (not finite(blur['component_mass']) or blur['component_mass'] <= 0 or
                not finite(blur['heatmap_peak']) or not 0 <= blur['heatmap_peak'] <= 1 or
                blur['heatmap_peak'] != raw['score']):
            return 'INVALID_BLURBALL_SCORE_OR_MASS'
        # pixelCenter is serialized once, as center_px. Its independent heatmap
        # derivation cannot be verified without rerunning inference.
        if (not finite(blur['extent_source_px']) or blur['extent_source_px'] < 0 or
                not finite(blur['axis_deg']) or not 0 <= blur['axis_deg'] < 180):
            return 'INVALID_BLUR_EVIDENCE'
        if not all(point(endpoint) for endpoint in blur['endpoints_px']):
            return 'NONFINITE_BLUR_ENDPOINT'
    return ''


def validate_candidate(candidate, row, model, model_hash, index):
    context = f'Frame {row["frame_id"]} {model} candidate {index}'
    expect_fields(candidate, (*IDENTITY_FIELDS, 'model', 'model_sha256', 'candidate_index',
                             'valid', 'invalid_reason', 'center_px', 'coordinate_basis',
                             'center_method', 'raw_score', 'score_kind', 'confidence',
                             'confidence_reason', 'raw_detection', 'blur_evidence'), context)
    for key in IDENTITY_FIELDS:
        require(type(candidate[key]) is type(row[key]) and candidate[key] == row[key],
                f'{context}: source-frame identity mismatch ({key})')
    expect_value(candidate, 'model', model, context)
    expect_value(candidate, 'model_sha256', model_hash, context)
    expect_value(candidate, 'candidate_index', index, context)
    expect_value(candidate, 'coordinate_basis', 'SOURCE_PIXELS', context)
    expect_value(candidate, 'center_method', 'YOLO_BOX_CENTER' if model == 'yolo'
                 else 'HEATMAP_WEIGHTED_CENTROID', context)
    expect_value(candidate, 'score_kind', 'DETECTOR_SCORE' if model == 'yolo' else 'HEATMAP_PEAK', context)
    expect_value(candidate, 'confidence', None, context)
    expect_value(candidate, 'confidence_reason', 'RAW_MODEL_SCORE_NOT_CORRECTNESS_CALIBRATED', context)
    require(type(candidate['valid']) is bool and point(candidate['center_px'], False),
            f'{context}: malformed valid flag or center')
    raw = candidate['raw_detection']
    expect_fields(raw, ('class_id', 'score', 'x', 'y', 'w', 'h'), context + ' raw_detection')
    require(type(raw['class_id']) is int, f'{context}: class_id must be integer')
    for key in ('score', 'x', 'y', 'w', 'h'):
        value = raw[key]
        require(value is None or finite(value), f'{context}: raw {key} must be finite or native null')
        require(value is None or value == f32(value), f'{context}: raw {key} is not native float32')
    require(candidate['raw_score'] == raw['score'] and type(candidate['raw_score']) is type(raw['score']),
            f'{context}: raw score was changed or averaged')
    for value in candidate['center_px']:
        require(value is None or value == f32(value), f'{context}: center is not native float32')
    blur = candidate['blur_evidence']
    if blur is not None:
        expect_fields(blur, ('heatmap_peak', 'component_mass', 'extent_source_px', 'axis_deg',
                             'axis_directed', 'endpoints_px'), context + ' blur_evidence')
        expect_value(blur, 'axis_directed', False, context)
        require(isinstance(blur['endpoints_px'], list) and len(blur['endpoints_px']) == 2 and
                all(point(endpoint, False) for endpoint in blur['endpoints_px']),
                f'{context}: malformed blur endpoints')
        values = [blur[key] for key in ('heatmap_peak', 'component_mass', 'extent_source_px', 'axis_deg')]
        values += [v for endpoint in blur['endpoints_px'] for v in endpoint]
        require(all(v is None or finite(v) and v == f32(v) for v in values),
                f'{context}: blur evidence is not native float32 or null')
    if model == 'blurball':
        require(blur is not None and point(candidate['center_px']) and finite(blur['component_mass']),
                f'{context}: standalone BlurBall evidence cannot be independently recomputed')
    reason = invalid_geometry(candidate)
    require(candidate['valid'] == (not reason) and candidate['invalid_reason'] == (reason or None),
            f'{context}: valid/invalid_reason disagrees with recomputed geometry ({reason})')


def recompute_selection(row, config):
    yolo = [c for c in row['yolo_candidates'] if c['valid'] and c['raw_score'] >= config['yolo_agreement_score']]
    blur = [c for c in row['blurball_candidates'] if c['valid'] and c['raw_score'] >= config['blurball_agreement_peak']]
    result = {'selected': None, 'reason': None, 'confidence': None,
              'confidence_reason': 'EXPERIMENTAL_UNCALIBRATED_SELECTOR',
              'yolo_support_index': None, 'blurball_support_index': None,
              'agreement_distance_source_px': None,
              'invalid_candidates': sum(not c['valid'] for c in row['yolo_candidates'] + row['blurball_candidates']),
              'eligible_yolo': len(yolo), 'eligible_blurball': len(blur)}

    def finish(reason, selected=None):
        result.update(reason=reason, selected=selected)
        return result

    if not row['temporal_window_ready']:
        return finish('BLURBALL_CAUSAL_WARMUP')
    rank = lambda c: c['blur_evidence']['component_mass'] if c['model'] == 'blurball' else c['raw_score']
    yolo_best, blur_best = (max(candidates, key=rank) if candidates else None for candidates in (yolo, blur))

    def ambiguous(candidates, winner):
        return winner is not None and any(c is not winner and rank(c) >= rank(winner) * config['within_model_ambiguity_ratio']
                                         and native_distance(c['center_px'], winner['center_px']) > config['agreement_radius_source_px']
                                         for c in candidates)

    if ambiguous(blur, blur_best):
        return finish('AMBIGUOUS_BLURBALL_CANDIDATES')
    if ambiguous(yolo, yolo_best):
        return finish('AMBIGUOUS_YOLO_CANDIDATES')
    result['yolo_support_index'] = yolo_best['candidate_index'] if yolo_best else None
    result['blurball_support_index'] = blur_best['candidate_index'] if blur_best else None
    if yolo_best and blur_best:
        distance = native_distance(blur_best['center_px'], yolo_best['center_px'])
        result['agreement_distance_source_px'] = distance
        if distance > config['agreement_radius_source_px']:
            return finish('DETECTOR_DISAGREEMENT')
        return finish('SPATIAL_AGREEMENT_BLURBALL_COORDINATE', blur_best)
    if blur_best:
        return finish('BLURBALL_ONLY_INSUFFICIENT_EVIDENCE') if blur_best['raw_score'] < config['blurball_solo_peak'] else finish('STRONG_BLURBALL_ONLY', blur_best)
    if yolo_best:
        return finish('YOLO_ONLY_INSUFFICIENT_EVIDENCE') if yolo_best['raw_score'] < config['yolo_solo_score'] else finish('CONSERVATIVE_YOLO_ONLY_FALLBACK', yolo_best)
    return finish('NO_ELIGIBLE_CANDIDATE')


def validate_manifest(manifest):
    context = 'manifest'
    expected = {'schema_version': 1, 'evaluation': 'PAIRED_CAUSAL_NATIVE_CANDIDATE_REPLAY',
                'timestamp_basis': 'ACTUAL_OPENCV_FFMPEG_DECODER_PTS_SECONDS',
                'invalid_pts_policy': 'FAIL_CLOSED_PARTIAL_JOURNAL',
                'gap_policy': 'RESET_CAUSAL_HISTORY_AND_STANDALONE_ANCHOR',
                'standalone_blurball_selection': 'HIGHEST_COMPONENT_MASS_STRICT_300_SOURCE_PIXEL_ANCHOR_GATE',
                'temporal_policy': 'CAUSAL_CHANNEL_2_T_MINUS_2_T_MINUS_1_T',
                'lookahead_frames': 0, 'warmup_frames': 2, 'source_images_shared': True,
                'opened_camera': False, 'labels_loaded': False, 'confidence_calibrated': False,
                'promotion_approved': False, 'experimental': True}
    for key, value in expected.items():
        expect_value(manifest, key, value, context)
    for key in ('source_sha256', 'yolo_model_sha256', 'blurball_model_sha256', 'runtime_sha256', 'fusion_sha256'):
        require(valid_hash(manifest.get(key)), f'manifest: invalid {key}')
    for key in ('source_path', 'yolo_model_path', 'blurball_model_path'):
        require(isinstance(manifest.get(key), str) and manifest[key], f'manifest: missing {key}')
    for key in ('source_declared_frames', 'requested_frames'):
        require(integer(manifest.get(key), 1), f'manifest: invalid {key}')
    require(manifest['requested_frames'] <= manifest['source_declared_frames'], 'manifest: requested frames exceed declared source')
    interval = manifest.get('source_interval')
    require(interval in ('COMPLETE_RECORDING', 'CHRONOLOGICAL_PREFIX_ONLY'), 'manifest: unknown source interval')
    require(interval != 'COMPLETE_RECORDING' or manifest['requested_frames'] == manifest['source_declared_frames'],
            'manifest: complete recording has incomplete requested coverage')
    for key in ('source_fps', 'maximum_temporal_gap_s'):
        require(finite(manifest.get(key)) and manifest[key] > 0, f'manifest: invalid {key}')
    require(manifest.get('standalone_yolo_threshold') == .25, 'manifest: unsupported standalone YOLO threshold')
    require(manifest.get('blurball_heatmap_threshold') == f32(.7), 'manifest: unsupported BlurBall heatmap threshold')
    require(finite(manifest.get('yolo_nms_threshold')) and 0 < manifest['yolo_nms_threshold'] <= 1,
            'manifest: invalid YOLO NMS threshold')
    preprocessing = manifest.get('preprocessing')
    for key, value in {'yolo': 'opencv-bgr-rgb-letterbox-640-chw-div255-v1',
                       'blurball': 'opencv-affine-rgb-imagenet-512x288-three-frames-v1',
                       'input_transform': 'RAW_SOURCE_NO_FILTER',
                       'decoded_image_hash': 'SHA256_UINT32_LE_WIDTH_HEIGHT_OPENCV_TYPE_THEN_PACKED_BGR_ROWS'}.items():
        expect_value(preprocessing, key, value, 'manifest preprocessing')
    config = manifest.get('fusion')
    expect_fields(config, (*CONFIG_POLICIES, *CONFIG_NUMBERS), 'fusion')
    require(set(config) == set(CONFIG_POLICIES) | set(CONFIG_NUMBERS), 'fusion: unknown configuration fields/revision')
    for key, value in CONFIG_POLICIES.items():
        expect_value(config, key, value, 'fusion')
    for key in CONFIG_NUMBERS:
        require(finite(config[key]) and config[key] > 0, f'fusion: invalid {key}')
    require(0 < config['yolo_proposal_score'] <= config['yolo_agreement_score'] <= config['yolo_solo_score'] <= 1 and
            config['yolo_proposal_score'] <= .25 and
            f32(.7) <= config['blurball_agreement_peak'] <= config['blurball_solo_peak'] <= 1 and
            config['within_model_ambiguity_ratio'] <= 1, 'fusion: inconsistent thresholds')
    for key in CONFIG_NUMBERS[:5]:
        require(config[key] == f32(config[key]), f'fusion: {key} is not native float32')
    require(manifest.get('yolo_proposal_threshold') == config['yolo_proposal_score'], 'manifest: proposal threshold differs from fusion')
    require(sha256_bytes(cpp_canonical_json(config).encode('utf-8')) == manifest['fusion_sha256'],
            'manifest: fusion SHA-256 does not match native canonical configuration')


def audit_runtime_binary(manifest, runtime_binary=None):
    """Separate recorded runtime identity from verification of supplied bytes."""
    provenance = {
        'runtime_binary_sha256_recorded_from_manifest': manifest['runtime_sha256'],
        'runtime_binary_current_file_reverified': False,
        'runtime_binary_supplied_path': None,
        'runtime_binary_sha256_current_file': None,
        'runtime_binary_limitation_reason':
            'No exact runtime binary was supplied. The manifest SHA-256 is recorded provenance; '
            'the historical executable bytes have not been reverified.',
    }
    if runtime_binary is not None:
        path = Path(runtime_binary)
        try:
            current_hash = sha256_file(path)
        except OSError as error:
            raise ValueError(f'Runtime binary is missing/unreadable: {path}') from error
        require(current_hash == manifest['runtime_sha256'],
                'Runtime binary SHA-256 mismatch: supplied executable does not match recorded runtime provenance')
        provenance.update(runtime_binary_current_file_reverified=True,
                          runtime_binary_supplied_path=str(path.resolve()),
                          runtime_binary_sha256_current_file=current_hash,
                          runtime_binary_limitation_reason=None)
    return provenance


def audit_journal(journal_dir, runtime_binary=None):
    journal_dir = Path(journal_dir)
    require(not (journal_dir / 'error.json').exists(), 'Journal contains an error/partial-run marker')
    metadata, hashes = {}, {}
    for name in ('manifest.json', 'diagnostic_summary.json', 'completion.json'):
        try:
            data = (journal_dir / name).read_bytes()
        except OSError as error:
            raise ValueError(f'Missing/unreadable required journal file {name}') from error
        metadata[name] = load_json(data, name)
        hashes[name] = sha256_bytes(data)
    manifest, summary, completion = (metadata[name] for name in ('manifest.json', 'diagnostic_summary.json', 'completion.json'))
    validate_manifest(manifest)
    runtime_provenance = audit_runtime_binary(manifest, runtime_binary)
    count = manifest['requested_frames']
    for obj, name in ((summary, 'diagnostic_summary'), (completion, 'completion')):
        expect_value(obj, 'status', 'COMPLETE', name)
        expect_value(obj, 'analyzed_frames', count, name)
    expect_value(completion, 'total_frames', count, 'completion')
    expect_value(summary, 'source_frames_decoded', count, 'diagnostic_summary')
    expect_value(summary, 'source_declared_frames', manifest['source_declared_frames'], 'diagnostic_summary')
    for key in ('source_sha256', 'yolo_model_sha256', 'blurball_model_sha256', 'fusion_sha256'):
        expect_value(summary, key, manifest[key], 'diagnostic_summary')
    for key, value in {'source_count_matches_metadata': count == manifest['source_declared_frames'],
                       'source_pts_strictly_increasing': True, 'lookahead_frames': 0,
                       'opened_camera': False, 'accuracy_measured': False, 'live_latency_measured': False,
                       'promotion_approved': False, 'labels_loaded': False,
                       'original_source_mutated': False, 'production_backend_modified': False}.items():
        expect_value(summary, key, value, 'diagnostic_summary')
    require(type(summary.get('decoder_eof_reached')) is bool, 'diagnostic_summary: missing decoder EOF flag')
    entire = summary['decoder_eof_reached'] and count == manifest['source_declared_frames']
    expect_value(summary, 'entire_recording_decoded', entire, 'diagnostic_summary')
    if manifest['source_interval'] == 'COMPLETE_RECORDING':
        require(entire, 'Complete recording journal must reach decoder EOF')
    else:
        expect_value(summary, 'decoder_eof_reached', False, 'diagnostic_summary prefix')

    verified_files = {}
    for name in ('source', 'yolo_model', 'blurball_model'):
        path = Path(manifest[name + '_path'])
        available = path.is_file()
        verified_files[name] = {'path': str(path), 'available': available,
                                'sha256_match': True if available else None}
        if available:
            require(sha256_file(path) == manifest[name + '_sha256'], f'{name}: file SHA-256 mismatch')

    try:
        index_data = (journal_dir / 'frames.idx').read_bytes()
        records_data = (journal_dir / 'frames.jsonl').read_bytes()
    except OSError as error:
        raise ValueError('Missing/unreadable journal JSONL/index') from error
    require(len(index_data) == count * 8, 'Journal index count/uint64 alignment disagrees with complete coverage')
    require(records_data.endswith(b'\n'), 'Journal JSONL has an incomplete final line')
    lines = records_data.splitlines(keepends=True)
    require(len(lines) == count and all(line.strip() for line in lines), 'Journal JSONL does not fully account for source frames')
    offsets = struct.unpack('<' + 'Q' * count, index_data)
    offset = 0
    rows, previous, window, anchor = [], None, [], None
    counters, reasons = Counter(), Counter()
    intervals = []
    totals = Counter()
    for fid, line in enumerate(lines, 1):
        require(offsets[fid - 1] == offset, f'Frame {fid}: corrupt index offset')
        offset += len(line)
        row = load_json(line, f'Frame {fid}')
        expect_fields(row, (*IDENTITY_FIELDS, 'fusion_sha256', 'nominal_source_time_s',
                            'same_decoded_pixels_as_previous', 'temporal_reset_reason',
                            'temporal_window_frame_ids', 'temporal_window_ready', 'yolo_candidates',
                            'blurball_candidates', 'standalone_yolo', 'standalone_blurball',
                            'experimental_selection', 'timings'), f'Frame {fid}')
        expect_value(row, 'frame_id', fid, f'Frame {fid}')
        expect_value(row, 'source_sha256', manifest['source_sha256'], f'Frame {fid}')
        expect_value(row, 'fusion_sha256', manifest['fusion_sha256'], f'Frame {fid}')
        require(valid_hash(row['decoded_image_sha256']), f'Frame {fid}: invalid decoded image SHA-256')
        require(finite(row['source_pts_s']) and row['source_pts_s'] >= 0,
                f'Frame {fid}: invalid actual source PTS')
        require(isinstance(row['source_size'], list) and len(row['source_size']) == 2 and
                all(integer(v, 1) for v in row['source_size']), f'Frame {fid}: invalid source dimensions')
        nominal = (fid - 1) / manifest['source_fps']
        require(finite(row['nominal_source_time_s']) and row['nominal_source_time_s'] == nominal,
                f'Frame {fid}: invalid nominal diagnostic time')
        reset = None
        if previous:
            interval = row['source_pts_s'] - previous['source_pts_s']
            require(interval > 0, f'Frame {fid}: actual PTS must be strictly increasing')
            intervals.append(interval)
            if interval > manifest['maximum_temporal_gap_s']:
                reset = 'SOURCE_PTS_GAP'
            if row['source_size'] != previous['source_size']:
                reset = 'SOURCE_DIMENSION_CHANGE'
        expect_value(row, 'temporal_reset_reason', reset, f'Frame {fid}')
        if reset:
            window, anchor = [], None
            counters['temporal_resets'] += 1
        duplicate = previous is not None and row['decoded_image_sha256'] == previous['decoded_image_sha256']
        expect_value(row, 'same_decoded_pixels_as_previous', duplicate, f'Frame {fid}')
        counters['consecutive_pixel_duplicates'] += duplicate
        window = (window + [fid])[-3:]
        require(row['temporal_window_frame_ids'] == window and
                all(type(v) is int for v in row['temporal_window_frame_ids']), f'Frame {fid}: invalid causal window frame IDs')
        ready = len(window) == 3
        expect_value(row, 'temporal_window_ready', ready, f'Frame {fid}')
        counters['causal_warmup_frames'] += not ready
        for model in ('yolo', 'blurball'):
            candidates = row[model + '_candidates']
            require(isinstance(candidates, list), f'Frame {fid}: candidate output must be list')
            for index, candidate in enumerate(candidates):
                validate_candidate(candidate, row, model, manifest[model + '_model_sha256'], index)
            counters[model + '_candidates'] += len(candidates)
            counters[model + '_invalid_candidates'] += sum(not c['valid'] for c in candidates)
        require(ready or not row['blurball_candidates'], f'Frame {fid}: BlurBall candidates during causal warmup')
        yolo_eligible = [c for c in row['yolo_candidates'] if c['valid'] and c['raw_score'] >= manifest['standalone_yolo_threshold']]
        baseline_yolo = max(yolo_eligible, key=lambda c: c['raw_score']) if yolo_eligible else None
        blur_eligible = [c for c in row['blurball_candidates']
                         if anchor is None or native_distance(c['center_px'], anchor) < 300]
        baseline_blur = max(blur_eligible, key=lambda c: c['blur_evidence']['component_mass']) if blur_eligible else None
        require(baseline_blur is None or baseline_blur['valid'], f'Frame {fid}: standalone BlurBall chose invalid evidence')
        require(row['standalone_yolo'] == baseline_yolo, f'Frame {fid}: standalone YOLO differs from highest-score existing candidate')
        require(row['standalone_blurball'] == baseline_blur, f'Frame {fid}: standalone BlurBall differs from mass/300px anchor policy')
        anchor = baseline_blur['center_px'] if baseline_blur else None
        expected = recompute_selection(row, manifest['fusion'])
        selected = row['experimental_selection']
        expect_fields(selected, expected, f'Frame {fid} experimental selection')
        require(set(selected) == set(expected), f'Frame {fid}: unknown experimental selection fields')
        for key, value in expected.items():
            if key == 'agreement_distance_source_px' and value is not None:
                require(finite(selected[key]) and math.isclose(selected[key], value, rel_tol=1e-14, abs_tol=1e-12),
                        f'Frame {fid}: experimental agreement distance differs from source proposals')
            else:
                require(type(selected[key]) is type(value) and selected[key] == value,
                        f'Frame {fid}: experimental {key} differs from recomputed selector; no blend permitted')
        reasons[selected['reason']] += 1
        row['experimental_selector'] = selected['selected']
        row['standalone_blurball_causal'] = row['standalone_blurball']
        for key in HYPOTHESES:
            counters[key + '_selected_frames'] += row[key] is not None
        timings = row['timings']
        expect_value(timings, 'blurball_inference_executed', ready, f'Frame {fid} timings')
        expect_value(timings, 'live_latency_measured', False, f'Frame {fid} timings')
        expect_value(timings, 'yolo_gpu_ms', None, f'Frame {fid} timings')
        for key in ('yolo_preprocess_wall_ms', 'yolo_inference_wall_ms', 'blurball_adapter_wall_ms',
                    'blurball_engine_wall_ms', 'frame_before_journal_wall_ms'):
            require(finite(timings.get(key)) and timings[key] >= 0, f'Frame {fid}: invalid {key}')
        gpu_ms = timings.get('blurball_gpu_ms')
        require(finite(gpu_ms) and gpu_ms >= 0 if ready else gpu_ms is None,
                f'Frame {fid}: BlurBall GPU timing does not match causal window')
        if not ready:
            expect_value(timings, 'blurball_engine_wall_ms', 0, f'Frame {fid} timings')
        totals['yolo_inference_total_wall_ms'] += timings['yolo_inference_wall_ms']
        totals['blurball_adapter_total_wall_ms'] += timings['blurball_adapter_wall_ms']
        totals['blurball_gpu_total_ms'] += gpu_ms or 0
        rows.append(row)
        previous = row
    for key in ('temporal_resets', 'consecutive_pixel_duplicates', 'causal_warmup_frames',
                'standalone_yolo_selected_frames'):
        expect_value(summary, key, counters[key], 'diagnostic_summary')
    expect_value(summary, 'standalone_blurball_selected_frames', counters['standalone_blurball_causal_selected_frames'], 'diagnostic_summary')
    expect_value(summary, 'experimental_selected_frames', counters['experimental_selector_selected_frames'], 'diagnostic_summary')
    for key, value in [('source_pts_first_s', rows[0]['source_pts_s']), ('source_pts_last_s', rows[-1]['source_pts_s'])]:
        require(summary.get(key) == value, f'diagnostic_summary: {key} mismatch')
    median_interval = sorted(intervals)[len(intervals) // 2] if intervals else None
    require(summary.get('median_source_interval_s') == median_interval, 'diagnostic_summary: median actual PTS interval mismatch')
    for key, value in totals.items():
        require(finite(summary.get(key)) and math.isclose(summary[key], value, rel_tol=1e-10, abs_tol=1e-7),
                f'diagnostic_summary: {key} does not match frame evidence')
    hashes['frames.jsonl'], hashes['frames.idx'] = sha256_bytes(records_data), sha256_bytes(index_data)
    return rows, manifest, {'status': 'PASS', 'frames_accounted_for': count,
                            **runtime_provenance,
                            'index_offsets_verified': True, 'file_sha256': hashes,
                            'source_and_model_files': verified_files,
                            'decoded_pixels_independently_recomputed': False,
                            'candidate_ground_truth_verified_by_hashes': False,
                            'selection_and_baseline_policies_recomputed': True,
                            'counters': dict(counters), 'selector_reason_histogram': dict(sorted(reasons.items()))}


def load_labels(labels_csv, frame_count, source_sha256=None):
    labels = {}
    try:
        with Path(labels_csv).open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames
            require(fields is not None and len(fields) == len(set(fields)) and
                    {'Frame', 'Visibility', 'X', 'Y'} <= set(fields), 'Malformed CSV label headers')
            for line_number, label in enumerate(reader, 2):
                require(None not in label and all(v is not None for v in label.values()),
                        f'Malformed CSV row {line_number}: field count mismatch')
                frame_text, visibility = label['Frame'].strip(), label['Visibility'].strip()
                require(re.fullmatch(r'[0-9]+', frame_text) is not None, f'Unknown/malformed reference frame at CSV row {line_number}')
                fid = int(frame_text)
                require(0 <= fid < frame_count, f'Unknown reference source frame {fid}')
                require(fid not in labels, f'Duplicate label source frame {fid}')
                if 'source_sha256' in label:
                    require(source_sha256 is not None and label['source_sha256'].strip() == source_sha256,
                            f'Mixed/unknown reference source identity for frame {fid}')
                require(visibility in ('0', '1'), 'Unknown/occluded visibility must not silently become an absent label')
                coordinates = [label['X'].strip(), label['Y'].strip()]
                if visibility == '1':
                    require(all(coordinates), f'Missing visible reference coordinates for frame {fid}')
                    try:
                        center = [float(c) for c in coordinates]
                    except ValueError as error:
                        raise ValueError(f'Malformed visible reference coordinates for frame {fid}') from error
                    require(all(math.isfinite(c) for c in center), 'Visible reference location must be finite')
                else:
                    # Explicit binary absence is the reference. Coordinates are unused.
                    for c in coordinates:
                        if c:
                            try:
                                float(c)
                            except ValueError as error:
                                raise ValueError(f'Malformed absent reference coordinates for frame {fid}') from error
                    center = None
                labels[fid] = {'visible': visibility == '1', 'center_px': center}
    except (OSError, csv.Error, UnicodeError) as error:
        raise ValueError(f'Malformed/unreadable label CSV: {error}') from error
    require(set(labels) == set(range(frame_count)), 'Labels and journal cover different source frames; missing reference frames')
    return labels


def matched_metrics(rows, labels, tolerance):
    result = {key: metrics(rows, labels, key, tolerance) for key in HYPOTHESES}
    for values in result.values():
        values['missed_visible_reference_frames'] = values['no_prediction_visible_frames'] + values['wrong_location_predictions']
        values['false_positives'] = values['absent_false_positives'] + values['wrong_location_predictions']
        values['false_negatives'] = values['missed_visible_reference_frames']
    return result


def regression_gates(values):
    baselines = HYPOTHESES[:2]
    strongest = max(baselines, key=lambda key: (values[key]['f1'], values[key]['recall'] or 0,
                                               values[key]['precision'] or 0))
    hybrid = values['experimental_selector']
    checks = {}
    for metric in ('precision', 'recall'):
        available = [(values[key][metric], key) for key in baselines if values[key][metric] is not None]
        target, baseline = max(available) if available else (None, None)
        actual = hybrid[metric]
        checks[metric + '_no_regression'] = {'baseline': baseline, 'baseline_value': target,
                                            'experimental_value': actual,
                                            'passed': actual >= target if target is not None and actual is not None else None}
    negatives = hybrid['absent_reference_frames']
    baseline = min(baselines, key=lambda key: values[key]['absent_false_positives']) if negatives else None
    checks['negative_false_positives_no_regression'] = {
        'baseline': baseline, 'absent_reference_frames': negatives,
        'baseline_value': values[baseline]['absent_false_positives'] if baseline else None,
        'experimental_value': hybrid['absent_false_positives'] if negatives else None,
        'passed': hybrid['absent_false_positives'] <= values[baseline]['absent_false_positives'] if negatives else None}
    outcomes = [check['passed'] for check in checks.values()]
    return {'strongest_baseline_by_f1': strongest,
            'policy': 'No regression against the strongest measured baseline for each metric; explicit absent labels required for negatives',
            'checks': checks,
            'passed': False if False in outcomes else True if all(v is True for v in outcomes) else None,
            'failed_checks': [key for key, check in checks.items() if check['passed'] is False],
            'unevaluated_checks': [key for key, check in checks.items() if check['passed'] is None]}


def score(journal_dir, labels_csv=None, tolerance=4., runtime_binary=None):
    require(finite(tolerance) and tolerance > 0, 'Tolerance must be finite and positive')
    rows, manifest, audit = audit_journal(journal_dir, runtime_binary)
    counts = {key: {'selected_frames': sum(row[key] is not None for row in rows),
                    'abstained_frames': sum(row[key] is None for row in rows)} for key in HYPOTHESES}
    counts['standalone_yolo']['candidate_count'] = audit['counters'].get('yolo_candidates', 0)
    counts['standalone_blurball_causal']['candidate_count'] = audit['counters'].get('blurball_candidates', 0)
    counts['experimental_selector']['selected_model_counts'] = dict(Counter(row['experimental_selector']['model']
        for row in rows if row['experimental_selector'] is not None))
    report = {'scope': 'Provisional evaluation of one source recording/prefix; checkpoint lineage is not established as blind validation',
              'journal_directory': str(Path(journal_dir).resolve()), 'source_interval': manifest['source_interval'],
              'entire_recording_decoded': manifest['source_interval'] == 'COMPLETE_RECORDING',
              'source_sha256': manifest['source_sha256'], 'yolo_model_sha256': manifest['yolo_model_sha256'],
              'blurball_model_sha256': manifest['blurball_model_sha256'], 'fusion_sha256': manifest['fusion_sha256'],
              **{key: value for key, value in audit.items() if key.startswith('runtime_binary_')},
              'selector_configuration': manifest['fusion'], 'audit': audit, 'hypothesis_counts': counts,
              'selector_reason_histogram': audit['selector_reason_histogram'], 'accuracy': None, 'gates': None,
              'matching': {'tolerance_source_px': tolerance, 'comparison': 'STRICT_DISTANCE_LESS_THAN_TOLERANCE',
                           'reference_coordinates': 'Unscaled CSV source pixels; zero-based Frame -> one-based journal frame_id',
                           'wrong_visible_localization': 'One false positive and one missed visible reference',
                           'recall_definition': 'Correctly localized detections / ALL visible reference frames'},
              'accuracy_measured': labels_csv is not None, 'settings_selected_before_scoring': True,
              'thresholds_fitted': False, 'confidence_calibrated': False, 'promotion_approved': False,
              'unlabeled_policy': 'Accuracy is null without explicit binary reference CSV; user-declared no-ball never becomes automatic ground truth',
              'remaining_gates': ['Independent recordings and venue sessions with known checkpoint lineage',
                                  'Broad ball/blur/occlusion and explicit negative reference comparisons',
                                  'Confidence/uncertainty calibration', 'Native live capture-to-display timing']}
    if labels_csv is not None:
        labels = load_labels(labels_csv, len(rows), manifest['source_sha256'])
        eligible_rows = [row for row in rows if row['temporal_window_ready']]
        report['reference'] = {'path': str(Path(labels_csv).resolve()), 'sha256': sha256_file(Path(labels_csv)),
                               'visibility_policy': 'Explicit binary 0/1; unknown and occluded visibility rejected'}
        report['accuracy'] = {'full_recording': matched_metrics(rows, labels, tolerance),
                              'after_causal_warmup': matched_metrics(eligible_rows, labels, tolerance),
                              'warmup_exclusion': 'Exclude every non-ready causal window, including after temporal resets'}
        report['gates'] = {scope: regression_gates(report['accuracy'][scope])
                           for scope in ('full_recording', 'after_causal_warmup')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journalDir', type=Path)
    parser.add_argument('outputJSON', type=Path)
    parser.add_argument('--labelsCSV', type=Path)
    parser.add_argument('--tolerance', type=float, default=4.)
    parser.add_argument('--runtime-binary', type=Path,
                        help='Exact archived native executable to SHA-256 verify against manifest runtime_sha256')
    args = parser.parse_args()
    try:
        result = score(args.journalDir, args.labelsCSV, args.tolerance, args.runtime_binary)
        # Preserve all existing reports; never overwrite/mix a prior evaluation.
        with args.outputJSON.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({'output': str(args.outputJSON), 'audit': result['audit']['status'],
                      'hypothesis_counts': result['hypothesis_counts'], 'gates': result['gates']},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
