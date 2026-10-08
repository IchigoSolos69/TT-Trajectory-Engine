"""Score a causal native detector/recovery replay. Ground truth is scoring-only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from improve_detector import annotation_status


def metrics(rows):
    tp = fp = fn = tn = 0
    for positive, detected, error in rows:
        correct = positive and detected and error <= 15
        tp += int(correct)
        fp += int(detected and not correct)
        fn += int(positive and not correct)
        tn += int(not positive and not detected)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def evaluate(predictions, markup, engine):
    labels = json.loads(markup.read_text())
    baseline, hybrid, recovered = [], [], []
    seen = set()
    unknown = calls = accepted = 0
    times = []
    details = []
    previous_frame = None
    with predictions.open(newline='') as source:
        for raw in csv.DictReader(source):
            fid = int(raw['frame'])
            if fid in seen or (previous_frame is not None and fid != previous_frame+1):
                raise ValueError('Replay must contain exactly one row per consecutive source frame')
            seen.add(fid)
            previous_frame = fid
            numbers = {k: float(v) for k, v in raw.items()}
            if not all(math.isfinite(v) for v in numbers.values()):
                raise ValueError('Nonfinite replay output')
            if raw['recovery_accepted'] not in ('0', '1') or not 0 <= numbers['missed_frames'] <= 4:
                raise ValueError('Invalid recovery state')
            invoked = numbers['missed_frames'] > 0
            calls += int(invoked)
            accepted += int(raw['recovery_accepted'] == '1')
            if invoked:
                times.append(numbers['gpu_ms'])
            label = labels.get(str(fid))
            status = annotation_status(label)
            if status == 'unknown':
                unknown += 1
                continue
            positive = status == 'positive'
            detected = numbers['baseline_x'] >= 0 and numbers['baseline_y'] >= 0
            use = raw['recovery_accepted'] == '1'
            if use and (detected or not invoked or numbers['recovery_score'] < .95):
                raise ValueError('Recovery violated causal dispatch policy')
            base_error = math.hypot(numbers['baseline_x']-label['x'], numbers['baseline_y']-label['y']) if positive else math.inf
            error = math.hypot(numbers['recovery_x']-label['x'], numbers['recovery_y']-label['y']) if positive else math.inf
            baseline.append((positive, detected, base_error))
            hybrid.append((positive, detected or use, error if use else base_error))
            if invoked:
                recovered.append((positive, use, error))
                details.append(dict(frame=fid, positive=positive, accepted=use,
                                    error_px=error if positive else None, score=numbers['recovery_score']))
    if not seen:
        raise ValueError('Empty replay')
    first, last = min(seen), max(seen)
    missing = [fid for fid, label in labels.items() if first <= int(fid) <= last and
               annotation_status(label) != 'unknown' and int(fid) not in seen]
    if missing:
        raise ValueError('Missing labeled frames')
    times.sort()
    baseline_metrics, hybrid_metrics = metrics(baseline), metrics(hybrid)
    recovery_metrics = metrics(recovered)
    baseline_absent_fp = sum(not positive and detected for positive, detected, _ in baseline)
    hybrid_absent_fp = sum(not positive and detected for positive, detected, _ in hybrid)
    gates = {
        'accepted_recovery_precision_95': recovery_metrics['precision'] is not None and recovery_metrics['precision'] >= .95,
        'precision_non_regression': (hybrid_metrics['precision'] is not None and baseline_metrics['precision'] is not None and hybrid_metrics['precision'] >= baseline_metrics['precision']),
        'recall_non_regression': (hybrid_metrics['recall'] is not None and baseline_metrics['recall'] is not None and hybrid_metrics['recall'] >= baseline_metrics['recall']),
        'absent_frame_false_positive_non_regression': hybrid_absent_fp <= baseline_absent_fp,
        'independent_sequence_verified': False,
        'live_camera_verified': False,
    }
    return {'scope': 'C++ TensorRT recovery with causal detector-driven ROIs; full source-frame replay. Sparse annotations, no live camera accuracy claim.',
            'first_frame': first, 'last_frame': last, 'source_frames': len(seen),
            'known_positive': sum(r[0] for r in baseline), 'known_negative': sum(not r[0] for r in baseline),
            'unknown_excluded': unknown, 'localization_tolerance_px': 15,
            'detector_threshold': .25, 'recovery_threshold': .95,
            'max_missed_frames': 4, 'recovery_extends_anchor_lifetime': False,
            'baseline': baseline_metrics, 'hybrid': hybrid_metrics,
            'baseline_absent_false_positives': baseline_absent_fp,
            'hybrid_absent_false_positives': hybrid_absent_fp,
            'promotion_gates': gates,
            'invoked_known_subset': recovery_metrics, 'recovery_calls': calls,
            'accepted_hypotheses_all_frames': accepted,
            'gpu_inference_median_ms': times[len(times)//2] if times else None,
            'timing_scope': 'Single offline replay, recovery GPU calls only; excludes decoding/crops/detector/dashboard. Neither a multi-pass throughput benchmark nor live-camera FPS.',
            'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
            'markup_sha256': hashlib.sha256(markup.read_bytes()).hexdigest(),
            'engine_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
            'promotion': 'NOT_APPROVED: requires independent sequences, negatives and live-camera evaluation',
            'known_recovery_frames': details}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('markup', type=Path)
    parser.add_argument('engine', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.predictions, args.markup, args.engine)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'known_recovery_frames'}, indent=2))
