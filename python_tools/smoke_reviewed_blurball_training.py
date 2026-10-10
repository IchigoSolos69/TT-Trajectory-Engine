"""Check two real reviewed CNN optimizer steps without saving a candidate.

Consumes a strict unassigned-session staging snapshot, not fabricated splits.
No candidate checkpoint, accuracy result or promotion is produced. The first
fully visible window and first fully absent window are selected deterministically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from evaluate_blurball_candidate import sha
from smoke_blurball_training import experiment
from stage_reviewed_training import StagedReviewedDataset


def select_examples(rows):
    result = {}
    for index, row in enumerate(rows):
        states = [frame['target']['visibility'] for frame in row['frames']]
        if states == ['visible'] * 3 and 'reviewed_visible' not in result:
            result['reviewed_visible'] = index
        if states == ['absent'] * 3 and 'reviewed_absent' not in result:
            result['reviewed_absent'] = index
    if set(result) != {'reviewed_visible', 'reviewed_absent'}:
        raise ValueError('Real smoke needs a fully reviewed visible triplet and a verified absent triplet')
    return [(result[kind], kind) for kind in ('reviewed_visible', 'reviewed_absent')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('staged', type=Path)
    parser.add_argument('--repo', type=Path, default=Path('artifacts/blurball_research/source'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/blurball_research/blurball_best'))
    parser.add_argument('--loss', required=True, choices=('mse', 'quality_focal'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Real reviewed data handoff / two in-memory optimizer steps only',
              'passed': False, 'loss': args.loss, 'candidate_fine_tuned': False,
              'accuracy_measured': False, 'checkpoint_saved': False,
              'promotion_approved': False, 'splits_assigned': False,
              'live_latency_measured': False, 'full_upstream_runner_verified': False}
    started = time.perf_counter()
    try:
        report['stage'] = 'audit_real_dataset'
        dataset = StagedReviewedDataset(args.staged, as_torch=True)
        examples = select_examples(dataset.rows)
        report['staged_manifest_sha256'] = sha(args.staged / 'manifest.json')
        report['staged_windows_sha256'] = sha(args.staged / 'windows.jsonl')
        report['real_smoke_source_sha256'] = sha(Path(__file__))
        report['staging_source_sha256'] = sha(Path(__file__).with_name('stage_reviewed_training.py'))
        report['available_windows'] = len(dataset)
        report['selected_examples'] = [
            {'dataset_index': index, 'kind': kind,
             'session_id': dataset.rows[index]['session_id'],
             'clip': dataset.rows[index]['clip'],
             'source_frames': [frame['source_frame'] for frame in dataset.rows[index]['frames']]}
            for index, kind in examples]
        experiment(args, report, dataset=dataset, examples=examples,
                   dataset_scope='Original reviewed images; current session unassigned; no validation/test data')
        # A successful backward pass does not waive mutable-label provenance.
        # Revalidate the complete staging snapshot after both optimizer steps.
        StagedReviewedDataset(args.staged, as_torch=False)
        report['snapshot_reverified_after_optimizer'] = True
    except Exception:
        report['passed'] = False
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['offline_smoke_wall_s'] = time.perf_counter() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'passed': report['passed'], 'stage': report.get('stage'),
                          'report': str(args.output / 'report.json')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
