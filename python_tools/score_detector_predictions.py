"""Score actual C++ top-1 predictions, excluding unknown annotation frames.

Use game_1 chronological validation to select thresholds; test_1 is development.
No aggregate performance here establishes accuracy on a live webcam.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from improve_detector import annotation_status


def score(rows, threshold, tolerance=15):
    tp = fp = fn = tn = 0
    errors, confidences = [], []
    reliability = [{'count': 0, 'confidence_sum': 0., 'correct': 0} for _ in range(10)]
    for row in rows:
        detected = row['confidence'] > threshold
        positive = row['status'] == 'positive'
        error = math.hypot(row['x'] - row['gt_x'], row['y'] - row['gt_y']) if positive else math.inf
        correct = positive and detected and error <= tolerance
        if correct:
            tp += 1
            errors.append(error)
            confidences.append(row['confidence'])
        else:
            fp += int(detected)
            fn += int(positive)
            tn += int(not detected and not positive)
        if detected:
            bucket = reliability[min(9, int(row['confidence'] * 10))]
            bucket['count'] += 1
            bucket['confidence_sum'] += row['confidence']
            bucket['correct'] += int(correct)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    errors.sort()
    return {'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0,
            'localized_frame_accuracy': (tp + tn) / len(rows) if rows else 0,
            'median_correct_error_px': errors[len(errors)//2] if errors else None,
            'mean_correct_confidence': sum(confidences)/len(confidences) if confidences else None,
            'reliability_bins': reliability}


def evaluate(predictions, markup, first_frame=1):
    labels = json.loads(markup.read_text())
    known = {int(fid): value for fid, value in labels.items()
             if int(fid) >= first_frame and annotation_status(value) != 'unknown'}
    rows = []
    seen = set()
    unknown = 0
    with predictions.open(newline='') as source:
        for raw in csv.DictReader(source):
            fid = int(raw['frame'])
            if fid in seen:
                raise ValueError(f'Duplicate predicted frame: {fid}')
            seen.add(fid)
            if fid < first_frame:
                continue
            if fid not in known:
                unknown += 1
                continue
            value = known[fid]
            x, y, confidence = float(raw['x']), float(raw['y']), float(raw['confidence'])
            if not all(math.isfinite(v) for v in (x, y, confidence)) or not 0 <= confidence <= 1:
                raise ValueError(f'Invalid prediction: {raw}')
            rows.append({'x': x, 'y': y, 'confidence': confidence,
                         'gt_x': value['x'], 'gt_y': value['y'], 'status': annotation_status(value)})
    missing = sorted(set(known) - seen)
    if missing:
        raise ValueError(f'Missing predictions for {len(missing)} annotated frames; first: {missing[:10]}')
    thresholds = [.05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .60, .70, .80, .90]
    scores = [score(rows, threshold) for threshold in thresholds]
    return {'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
            'markup_sha256': hashlib.sha256(markup.read_bytes()).hexdigest(),
            'evaluation': 'top-1 localization within 15 pixels, labeled frames only',
            'unknown_frames_excluded': unknown, 'positive_frames': sum(r['status'] == 'positive' for r in rows),
            'negative_frames': sum(r['status'] == 'negative' for r in rows),
            'warning': 'Sparse negatives; precision is conditional on labeled frames. Live accuracy unmeasured.',
            'scores': scores, 'best_f1_threshold': max(scores, key=lambda s: s['f1'])['threshold']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('markup', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--first-frame', type=int, default=1)
    args = parser.parse_args()
    report = evaluate(args.predictions, args.markup, args.first_frame)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'scores'}, indent=2))
    print('threshold TP FP FN TN precision recall F1')
    for s in report['scores']:
        print(f"{s['threshold']:.2f} {s['tp']} {s['fp']} {s['fn']} {s['tn']} {s['precision']:.4f} {s['recall']:.4f} {s['f1']:.4f}")
