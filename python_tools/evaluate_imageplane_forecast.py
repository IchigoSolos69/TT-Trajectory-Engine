"""Evaluate saved native image-plane fits against exact independent source labels.

This evaluator does not fit curves, interpolate labels, or use ground truth to
choose contacts, observations, model parameters or eligible fit support. A
recording detector with future-frame support makes the experiment noncausal.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

FIT_TIME_TOLERANCE = 2e-6  # Native std::to_string timestamp serialization.
SOURCE_TIME_TOLERANCE = 5.1e-5  # Telemetry source_time has four decimals.
HORIZONS = (1, 2)


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def point(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{name} must contain two coordinates')
    return [finite(v, name) for v in value]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field {key}')
        result[key] = value
    return result


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_labels(path, count):
    labels = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'Frame', 'Visibility', 'X', 'Y'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain Frame,Visibility,X,Y source labels')
        for row in reader:
            frame = int(row['Frame'])
            if frame < 0 or frame + 1 in labels:
                raise ValueError('Duplicate or negative ground-truth frame')
            visible = int(row['Visibility'])
            if visible not in (0, 1):
                raise ValueError('Unknown visibility cannot silently become an absent reference')
            labels[frame + 1] = {'visible': bool(visible), 'pixel':
                point([float(row['X']), float(row['Y'])], 'visible GT pixel') if visible else None}
    if set(labels) != set(range(1, count + 1)):
        raise ValueError('Labels must account for every exact zero-based source frame')
    return labels


def read_journal(path, count):
    rows = []
    offsets = []
    position = 0
    for raw in path.read_bytes().splitlines(keepends=True):
        offsets.append(position)
        position += len(raw)
        if not raw.endswith(b'\n') or not raw.strip():
            raise ValueError('Journal contains incomplete or empty records')
        rows.append(json.loads(raw, object_pairs_hook=unique_object))
    if len(rows) != count or [row.get('frame_id') for row in rows] != list(range(1, count + 1)):
        raise ValueError('Journal must contain every consecutive source frame, starting at 1')
    for row in rows:
        integer(row['frame_id'], 'frame_id', 1)
    index = path.with_name('frames.idx')
    if index.exists():
        data = index.read_bytes()
        if len(data) != count * 8 or list(struct.unpack('<' + 'Q' * count, data)) != offsets:
            raise ValueError('Journal index does not match complete source rows')
    completion = path.with_name('completion.json')
    if completion.exists():
        value = load_json(completion)
        if value.get('status') != 'COMPLETE' or value.get('analyzed_frames') != count or value.get('total_frames') != count:
            raise ValueError('Recording analysis is not complete')
    return rows


def observed_center(row):
    """Only explicit observations enter baseline histories, never future labels."""
    if row.get('ball') is None:
        return None
    blur = row.get('ball_blur')
    if not isinstance(blur, dict):
        raise ValueError('Observed BlurBall frame is missing its source-pixel center')
    return point(blur.get('center_px'), 'observed center')


def validate_flight(row, rows, observations, fps, model):
    frame = row['frame_id']
    canonical = (frame - 1) / fps
    source_time = finite(row.get('source_time'), 'source_time')
    if abs(source_time - canonical) > SOURCE_TIME_TOLERANCE:
        raise ValueError('Source timestamp differs from the declared nominal source FPS')
    if row.get('ball_model') != model:
        raise ValueError('Model provenance changes inside a single analysis journal')
    flight = row.get('ball_flight')
    if not isinstance(flight, dict):
        raise ValueError('Each source row must contain ball_flight metadata')
    if integer(flight.get('reference_frame'), 'reference_frame', 1) != frame:
        raise ValueError('Flight reference frame differs from its source row')
    if abs(finite(flight.get('reference_timestamp'), 'reference_timestamp') - canonical) > FIT_TIME_TOLERANCE:
        raise ValueError('Flight reference timestamp differs from its exact source frame')
    integer(flight.get('segment_id'), 'segment_id')
    if flight.get('method') != 'recent_observations_quadratic_v1' or flight.get('coordinate_basis') != 'source_pixels':
        raise ValueError('Unsupported flight fitting method or coordinate basis')
    horizon = finite(flight.get('maximum_horizon_s'), 'maximum_horizon_s')
    if not 0 < horizon <= .05 + 1e-9:
        raise ValueError('Forecast horizon exceeds the 50 ms bound')
    support = flight.get('support_frames')
    if not isinstance(support, list):
        raise ValueError('support_frames must be explicit')
    for support_frame in support:
        integer(support_frame, 'support frame', 1)
        if support_frame > frame:
            raise ValueError('Future observation leaked into the flight fit support')
        if observations[support_frame - 1] is None:
            raise ValueError('Fit support is not an explicit observed frame')
        if rows[support_frame - 1]['ball_flight']['segment_id'] != flight['segment_id']:
            raise ValueError('Fit support crosses a flight/contact segment boundary')
    if support != sorted(set(support)):
        raise ValueError('Fit support frames must be unique and chronological')
    forecasts = flight.get('points')
    if not isinstance(forecasts, list):
        raise ValueError('Forecast point list is missing')
    if flight.get('status') == 'UNAVAILABLE':
        if any(flight.get(key) is not None for key in ('fit_origin', 'fit_time_scale', 'fit_coefficients', 'fit_rmse_px')) or forecasts:
            raise ValueError('Unavailable fit cannot carry a forecast or coefficients')
        return flight
    if flight.get('status') != 'EXPERIMENTAL_FORECAST' or not 5 <= len(support) <= 9:
        raise ValueError('Ready quadratic fit requires five to nine observed support frames')
    origin = finite(flight.get('fit_origin'), 'fit_origin')
    scale = finite(flight.get('fit_time_scale'), 'fit_time_scale')
    last = (support[-1] - 1) / fps
    first = (support[0] - 1) / fps
    if abs(origin - last) > FIT_TIME_TOLERANCE or origin > canonical + FIT_TIME_TOLERANCE:
        raise ValueError('Fit origin is not its last past observation timestamp')
    if canonical > last + horizon + FIT_TIME_TOLERANCE:
        raise ValueError('Ready fit is past its last observation horizon')
    if scale < .04 - FIT_TIME_TOLERANCE or abs(scale - (last - first)) > FIT_TIME_TOLERANCE:
        raise ValueError('Fit time scale does not match observed support timestamps')
    if finite(flight.get('fit_rmse_px'), 'fit_rmse_px') < 0:
        raise ValueError('Fit residual cannot be negative')
    coefficients = flight.get('fit_coefficients')
    if not isinstance(coefficients, list) or len(coefficients) != 2:
        raise ValueError('Quadratic coefficients must be a 2x3 array')
    for axis in coefficients:
        if not isinstance(axis, list) or len(axis) != 3:
            raise ValueError('Quadratic coefficients must be a 2x3 array')
        for value in axis:
            finite(value, 'quadratic coefficient')
    previous = None
    for forecast in forecasts:
        timestamp = finite(forecast.get('timestamp'), 'forecast timestamp')
        if timestamp < canonical - FIT_TIME_TOLERANCE or timestamp > last + horizon + FIT_TIME_TOLERANCE:
            raise ValueError('Saved forecast exceeds its source time/observation horizon')
        if previous is not None and timestamp <= previous:
            raise ValueError('Saved forecast timestamps must increase')
        previous = timestamp
        point(forecast.get('pixel'), 'saved forecast pixel')
    return flight


def polynomial(flight, time):
    t = (time - flight['fit_origin']) / flight['fit_time_scale']
    return [axis[0] + axis[1] * t + axis[2] * t * t for axis in flight['fit_coefficients']]


def error_summary(values):
    ordered = sorted(values)
    def quantile(q):
        if not ordered:
            return None
        index = (len(ordered) - 1) * q
        low = math.floor(index)
        return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)
    return {'samples': len(values), 'mean_euclidean_error_px': sum(values) / len(values) if values else None,
            'rmse_euclidean_px': math.sqrt(sum(v * v for v in values) / len(values)) if values else None,
            'p50_error_px': quantile(.5), 'p95_error_px': quantile(.95)}


def evaluate(journal, labels_path, manifest_path=None):
    journal, labels_path = Path(journal), Path(labels_path)
    manifest_path = Path(manifest_path) if manifest_path else journal.with_name('manifest.json')
    manifest = load_json(manifest_path)
    if manifest.get('experimental_intelligence', False) is not False:
        raise ValueError('Raw-input baseline comparison requires experimental_intelligence:false; filtered fit inputs need separate provenance')
    fps = finite(manifest.get('source_fps'), 'source_fps')
    if fps <= 0:
        raise ValueError('Source FPS must be positive')
    count = integer(manifest.get('total_frames'), 'total_frames', 1)
    model = manifest.get('ball_model')
    if not isinstance(model, dict) or model.get('backend') != 'blurball':
        raise ValueError('An explicit BlurBall model provenance is required')
    lookahead = integer(model.get('lookahead_frames'), 'lookahead_frames')
    if model.get('temporal_policy') not in ('all_windows_two_frame_lookahead', 'causal_channel_2') or lookahead not in (0, 2):
        raise ValueError('Unknown temporal policy or lookahead')
    if (model['temporal_policy'] == 'causal_channel_2') != (lookahead == 0):
        raise ValueError('Lookahead contradicts the model temporal policy')
    rows = read_journal(journal, count)
    labels = read_labels(labels_path, count)
    observations = [observed_center(row) for row in rows]
    # Validate every source row first, including abstentions; never silently drop
    # malformed metadata, incomplete histories or future support.
    flights = [validate_flight(row, rows, observations, fps, model) for row in rows]
    samples = []
    bins = {}
    for offset in HORIZONS:
        accounting = Counter({key: 0 for key in ('target_outside_recording', 'target_absent_no_location_scored',
                              'visible_target_opportunities', 'fit_unavailable', 'last_observation_horizon_expired', 'paired_scored')})
        errors = {name: [] for name in ('quadratic', 'last_position', 'constant_velocity')}
        for row, flight in zip(rows, flights):
            frame, target = row['frame_id'], row['frame_id'] + offset
            if target > count:
                accounting['target_outside_recording'] += 1
                continue
            if not labels[target]['visible']:
                accounting['target_absent_no_location_scored'] += 1
                continue
            accounting['visible_target_opportunities'] += 1
            if flight['status'] != 'EXPERIMENTAL_FORECAST':
                accounting['fit_unavailable'] += 1
                continue
            support = flight['support_frames']
            last, previous = support[-1], support[-2]
            target_time, last_time = (target - 1) / fps, (last - 1) / fps
            if target_time > last_time + flight['maximum_horizon_s'] + 1e-9:
                accounting['last_observation_horizon_expired'] += 1
                continue
            last_pixel, previous_pixel = observations[last - 1], observations[previous - 1]
            interval = (last - previous) / fps
            predicted = {'quadratic': polynomial(flight, target_time), 'last_position': last_pixel,
                         'constant_velocity': [last_pixel[axis] + (last_pixel[axis] - previous_pixel[axis]) / interval * (target_time - last_time) for axis in (0, 1)]}
            if any(not all(math.isfinite(v) for v in pixel) for pixel in predicted.values()):
                raise ValueError('Nonfinite forecast must not be excluded from the paired comparison')
            paired_errors = {name: math.dist(pixel, labels[target]['pixel']) for name, pixel in predicted.items()}
            for name, error in paired_errors.items():
                errors[name].append(error)
            accounting['paired_scored'] += 1
            samples.append({'reference_frame': frame, 'target_frame': target, 'source_frame_horizon': offset,
                            'target_timestamp': target_time, 'last_observed_frame': last,
                            'elapsed_from_last_observation_s': target_time - last_time,
                            'segment_id': flight['segment_id'], 'support_frames': support,
                            'predictions_px': predicted, 'reference_px': labels[target]['pixel'], 'errors_px': paired_errors})
        accounting['source_reference_rows'] = count
        if accounting['target_outside_recording'] + accounting['target_absent_no_location_scored'] + accounting['visible_target_opportunities'] != count:
            raise ValueError('Internal target accounting is incomplete')
        if accounting['paired_scored'] + accounting['fit_unavailable'] + accounting['last_observation_horizon_expired'] != accounting['visible_target_opportunities']:
            raise ValueError('Internal forecast/abstention accounting is incomplete')
        coverage = accounting['paired_scored'] / accounting['visible_target_opportunities'] if accounting['visible_target_opportunities'] else None
        bins[str(offset)] = {'nominal_horizon_s': offset / fps, 'coverage_visible_targets': coverage,
                            'accounting': dict(accounting), 'paired_errors': {name: error_summary(values) for name, values in errors.items()}}
    all_errors = {name: [sample['errors_px'][name] for sample in samples] for name in ('quadratic', 'last_position', 'constant_velocity')}
    return {'schema_version': 1, 'scope': 'Exact source-pixel next-one/two-frame forecast on one recording; not physical 3D, calibrated uncertainty or live accuracy',
            'journal': str(journal.resolve()), 'label_path': str(labels_path.resolve()), 'manifest_path': str(manifest_path.resolve()),
            'hashes': {'journal_sha256': digest(journal), 'labels_sha256': digest(labels_path), 'manifest_sha256': digest(manifest_path),
                       'source_sha256': manifest.get('source_sha256'), 'ball_model_sha256': manifest.get('ball_model_sha256'), 'runtime_sha256': manifest.get('runtime_sha256')},
            'source_frames': count, 'source_fps': fps, 'source_observed_frames': sum(p is not None for p in observations),
            'model_provenance': model, 'detector_lookahead_frames': lookahead, 'causal_future_prediction_benchmark': lookahead == 0,
            'future_support_warning': 'The detector can use two future source frames, including the evaluated targets; fit supports are past-only but this is NOT a causal forecast benchmark.' if lookahead else None,
            'reference_policy': 'Zero-based official CSV matched to exact one-based native source frames; no interpolation; absent references are not assigned a location',
            'baseline_policy': 'Paired same eligible references/targets; last-position and constant-velocity use the last one/two raw observed fit-support centers; no GT gating',
            'fit_input_policy': 'Default detector-only journal; filtered experimental-tracker inputs are rejected until they have explicit fitting-observation provenance',
            'contacts': {'ground_truth_contact_labels_available': False, 'contact_subset_accuracy': None, 'note': 'Only native contact/motion segment boundaries influence existing fits; no GT-based resets or contact exclusions'},
            'fit_states': dict(Counter(flight['status'] for flight in flights)), 'fit_reasons': dict(Counter(flight.get('reason', 'UNSPECIFIED') for flight in flights)),
            'future_support_rejections': 0, 'metadata_policy': 'Malformed metadata or incomplete source coverage aborts evaluation; not omitted as abstentions',
            'horizons_source_frames': bins, 'paired_all_horizons': {name: error_summary(values) for name, values in all_errors.items()},
            'uncertainty_coverage': None, 'promotion_approved': False, 'samples': samples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    result = evaluate(args.journal, args.labels, args.manifest)
    if args.output.exists():
        parser.error('Refusing to overwrite an earlier forecast evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('source_frames', 'source_observed_frames', 'detector_lookahead_frames', 'causal_future_prediction_benchmark', 'fit_states', 'horizons_source_frames', 'paired_all_horizons', 'promotion_approved')}, indent=2, allow_nan=False))
