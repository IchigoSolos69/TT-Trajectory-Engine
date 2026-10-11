"""Two real reviewed in-memory optimizer steps; no saved candidate or accuracy.

CPU is the default to avoid competing with native GPU sessions. --device cuda
requires available CUDA and never silently falls back. Existing staging validates
all source/review/target/code hashes before loading and after optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import torch

import smoke_reviewed_blurball_training as reviewed_smoke
from stage_reviewed_training import StagedReviewedDataset
from temporal_ball_student import TemporalBallStudent, loss, reviewed_sample, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('staged', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'yolo11n.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'scope': 'Two real reviewed in-memory optimizer steps / causal student prototype',
              'accuracy_measured': False, 'candidate_fine_tuned': False, 'checkpoint_saved': False,
              'promotion_approved': False, 'live_latency_measured': False, 'splits_assigned': False,
              'teacher_used': False, 'device': args.device}
    started = time.perf_counter()
    try:
        torch.manual_seed(3407)
        torch.set_num_threads(4)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        report['stage'] = 'audit_staging_snapshot'
        dataset = StagedReviewedDataset(args.staged, as_torch=True)
        selection_helper_path = Path(reviewed_smoke.__file__).resolve()
        selection_helper_sha256 = sha(selection_helper_path)
        examples = reviewed_smoke.select_examples(dataset.rows)
        report['staged_manifest_sha256'] = sha(args.staged / 'manifest.json')
        report['staged_windows_sha256'] = sha(args.staged / 'windows.jsonl')
        report['prototype_source_sha256'] = sha(Path(__file__).with_name('temporal_ball_student.py'))
        report['smoke_source_sha256'] = sha(Path(__file__))
        report['selection_helper_source'] = selection_helper_path.as_posix()
        report['selection_helper_source_sha256'] = selection_helper_sha256
        report['torch'] = torch.__version__
        import ultralytics
        report['ultralytics'] = ultralytics.__version__
        report['available_windows'] = len(dataset)
        model = TemporalBallStudent(args.checkpoint, freeze_encoder=True).to(args.device).train()
        report['model'] = model.provenance
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        report['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report['head_geometry'] = {'fusion_stride': 4, 'heatmap': [1, 1, 288, 512], 'visibility': [1, 1]}
        encoder_before = {name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=0)
        report['optimizer'] = {'name': 'AdamW', 'learning_rate': 1e-4, 'weight_decay': 0}
        report['steps'] = []
        for index, kind in examples:
            report['stage'] = 'load_reviewed_example'
            x, targets, provenance = dataset.sample(index)
            rgb, supervision, context = reviewed_sample(x, targets, provenance)
            rgb = rgb.to(args.device)
            supervision = {k: v.to(args.device) for k, v in supervision.items()}
            before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            optimizer.zero_grad(set_to_none=True)
            report['stage'] = 'forward_backward'
            out = model(rgb)
            parts = loss(out, supervision)
            if not torch.isfinite(parts['total']):
                raise RuntimeError('Nonfinite scalar reviewed loss')
            parts['total'].backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise RuntimeError(f'Frozen encoder received gradient: {name}')
                    continue
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f'Missing/nonfinite prototype gradient: {name}')
                gradients[name] = float(parameter.grad.detach().norm())
                if gradients[name] == 0:
                    raise RuntimeError(f'Prototype parameter disconnected from supervision: {name}')
            report['stage'] = 'optimizer_step'
            optimizer.step()
            changed = []
            for name, parameter in model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise RuntimeError(f'Nonfinite parameter after optimizer: {name}')
                if name in before and not torch.equal(parameter, before[name]):
                    changed.append(name)
            if len(changed) != len(before):
                raise RuntimeError('All trainable prototype heads/fusion must change in this smoke')
            report['steps'].append({'dataset_index': index, 'kind': kind, 'context': context,
                'input_shape': list(rgb.shape), 'target_nonzero_pixels': int(supervision['heatmap'].count_nonzero()),
                'visibility_target': float(supervision['visibility'].item()),
                'losses': {k: float(v.detach()) for k, v in parts.items()},
                'finite_gradient_tensors': len(gradients), 'gradient_norms': gradients,
                'changed_parameter_tensors': len(changed), 'changed_parameters': changed})
        report['stage'] = 'reverify_weights_and_review_snapshot'
        for name, value in model.encoder.state_dict().items():
            if not torch.equal(value.detach().cpu(), encoder_before[name]):
                raise RuntimeError(f'Frozen encoder parameters/statistics changed: {name}')
        if sha(args.checkpoint) != model.provenance['checkpoint_sha256']:
            raise RuntimeError('Original local checkpoint changed')
        if sha(Path(__file__).with_name('temporal_ball_student.py')) != report['prototype_source_sha256'] or sha(Path(__file__)) != report['smoke_source_sha256']:
            raise RuntimeError('Prototype/smoke source changed during verification')
        if sha(selection_helper_path) != selection_helper_sha256:
            raise RuntimeError('Reviewed example selection helper changed during verification')
        StagedReviewedDataset(args.staged, as_torch=False)
        report['snapshot_reverified_after_optimizer'] = True
        report['encoder_weights_and_statistics_unchanged'] = True
        report['original_checkpoint_unchanged'] = True
        report['prototype_and_smoke_source_unchanged'] = True
        report['selection_helper_source_unchanged'] = True
        report['passed'] = True
        report['stage'] = 'complete'
    except Exception:
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['offline_smoke_wall_s'] = time.perf_counter() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'passed': report['passed'], 'stage': report.get('stage'), 'report': str(args.output / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
