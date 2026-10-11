"""Re-score both archived runs and compare original per-frame model evidence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python_tools'))
from score_paired_ball_candidates import score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    artifact = Path(__file__).resolve().parent
    old_dir = ROOT / 'artifacts/paired_ball_official_22_003_primary_v2'
    new_dir = artifact / 'paired640_primary_v2'
    old_metrics = json.loads((old_dir / 'metrics.json').read_text(encoding='utf-8'))
    labels = Path(old_metrics['reference']['path'])
    old_binary = Path(old_metrics['runtime_binary_supplied_path'])
    new_binary = artifact / 'eval_paired_ball_candidates_shape.exe'
    old = score(old_dir, labels, runtime_binary=old_binary)
    new = score(new_dir, labels, runtime_binary=new_binary)
    write_new(new_dir / 'metrics.json', new)
    read_rows = lambda folder: [json.loads(line) for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    previous, current = read_rows(old_dir), read_rows(new_dir)
    stable_fields = ['frame_id', 'source_sha256', 'decoded_image_sha256', 'source_pts_s',
        'nominal_source_time_s', 'source_size', 'same_decoded_pixels_as_previous',
        'temporal_window_frame_ids', 'temporal_window_ready', 'temporal_reset_reason',
        'yolo_candidates', 'blurball_candidates', 'standalone_yolo', 'standalone_blurball',
        'experimental_selection', 'fusion_sha256']
    differences = {key: [index + 1 for index, (a, b) in enumerate(zip(previous, current)) if a[key] != b[key]]
                   for key in stable_fields}
    identity_keys = ['source_sha256', 'yolo_model_sha256', 'blurball_model_sha256', 'fusion_sha256']
    checks = {'full_272_coverage': len(previous) == len(current) == 272,
              'same_source_models_selector': all(old[key] == new[key] for key in identity_keys),
              'all_original_frame_evidence_identical': not any(differences.values()),
              'aggregate_accuracy_identical': old['accuracy'] == new['accuracy'],
              'hypothesis_counts_identical': old['hypothesis_counts'] == new['hypothesis_counts'],
              'selector_reasons_identical': old['selector_reason_histogram'] == new['selector_reason_histogram'],
              'regression_gates_identical': old['gates'] == new['gates'],
              'both_archived_runtime_bytes_verified': old['runtime_binary_current_file_reverified'] and new['runtime_binary_current_file_reverified'],
              'reference_matches_historical_hash': sha(labels) == old_metrics['reference']['sha256'],
              'actual640_tensor_shape_recorded': new['tensor_contracts']['yolo']['input_shape'] == [1, 3, 640, 640]
                  and new['tensor_contracts']['yolo']['output_shape'] == [1, 5, 8400]}
    report = {'scope': 'Metadata/native-shape migration regression on complete published development recording; not blind accuracy or live timing',
              'checks': checks, 'passed': all(checks.values()), 'changed_frame_ids_by_field': differences,
              'previous_journal': str(old_dir), 'new_journal': str(new_dir),
              'old_binary_sha256': sha(old_binary), 'new_binary_sha256': sha(new_binary),
              'tensor_contracts': new['tensor_contracts'], 'promotion_approved': False,
              '960_status': 'Shape support is implemented; root raw960 CUDA p95 1.05 px case fails 1 px parity gate; 960 is not released'}
    write_new(artifact / 'paired_shape_regression.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit('Shape-migration regression failed; inspect preserved per-field evidence')


if __name__ == '__main__':
    main()
