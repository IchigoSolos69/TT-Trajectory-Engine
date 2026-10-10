"""Score source-indexed ball proposals against explicit, binary visibility CSV labels.

Incorrectly localized visible predictions count as both FP and missed locations.
The upstream BlurBall recall convention is reported separately to avoid inflating
recall by removing incorrectly localized positives from its denominator.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def metrics(rows, labels, key, tolerance):
    tp = fp_absent = fp_location = missed = tn = 0
    errors = []
    for row in rows:
        label = labels[row['frame_id'] - 1]
        visible = label['visible']
        candidate = row[key]
        if visible:
            if candidate is None:
                missed += 1
            else:
                distance = math.dist(candidate['center_px'], label['center_px'])
                errors.append(distance)
                if distance < tolerance:
                    tp += 1
                else:
                    fp_location += 1
        elif candidate is None:
            tn += 1
        else:
            fp_absent += 1
    positives = tp + fp_location + missed
    predictions = tp + fp_location + fp_absent
    ordered = sorted(errors)
    def quantile(q):
        if not ordered:
            return None
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        return ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)
    precision = tp / predictions if predictions else None
    recall = tp / positives if positives else None
    return {'frames': len(rows), 'visible_reference_frames': positives,
            'absent_reference_frames': fp_absent + tn, 'tp': tp,
            'wrong_location_predictions': fp_location, 'absent_false_positives': fp_absent,
            'no_prediction_visible_frames': missed, 'true_negatives': tn,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision and recall else 0.,
            'upstream_recall_convention': tp / (tp + missed) if tp + missed else None,
            'localization_all_visible_predictions': {
                'count': len(errors), 'median_source_px': quantile(.5), 'p95_source_px': quantile(.95),
                'rmse_source_px': math.sqrt(sum(d*d for d in errors) / len(errors)) if errors else None}}


def score(predictions, labels_path, tolerance=4.):
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    if [r['frame_id'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Predictions must account for consecutive source frames, starting at 1')
    labels = {}
    with labels_path.open(newline='', encoding='utf-8-sig') as stream:
        for label in csv.DictReader(stream):
            fid = int(label['Frame'])
            visible = int(label['Visibility'])
            if visible not in (0, 1):
                raise ValueError('Unknown/occluded visibility must not silently become an absent label')
            if fid in labels:
                raise ValueError(f'Duplicate label source frame {fid}')
            center = [float(label['X']), float(label['Y'])]
            if visible and not all(math.isfinite(c) for c in center):
                raise ValueError('Visible reference location must be finite')
            labels[fid] = {'visible': bool(visible), 'center_px': center}
    if set(labels) != set(range(len(rows))):
        raise ValueError('Labels and predictions cover different source frames')
    report = {'scope': 'One official match22 clip, not the entire dataset or live-camera accuracy',
              'label_sha256': hashlib.sha256(labels_path.read_bytes()).hexdigest(),
              'prediction_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
              'matching': f'One proposal per source frame; strict distance < {tolerance} SOURCE pixels',
              'reference_coordinates': 'Unscaled official CSV source pixels; zero-based CSV -> one-based source IDs',
              'recall_definition': 'Correctly localized detections / ALL visible reference frames',
              'settings_selected_before_scoring': True,
              'full_recording': {key: metrics(rows, labels, key, tolerance)
                                 for key in ['yolo', 'blurball_offline', 'blurball_causal']},
              'after_two_frame_causal_warmup': {
                  key: metrics(rows[2:], labels, key, tolerance)
                  for key in ['yolo', 'blurball_offline', 'blurball_causal']},
              'deployment_approved': False,
              'remaining_gates': ['Broad match/live negatives and blur/occlusion comparisons',
                                  'Confidence and uncertainty calibration',
                                  'Native integration and capture-to-display timing',
                                  'Independent venue sessions']}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tolerance', type=float, default=4.)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('Tolerance must be finite and positive')
    result = score(args.predictions, args.labels, args.tolerance)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
