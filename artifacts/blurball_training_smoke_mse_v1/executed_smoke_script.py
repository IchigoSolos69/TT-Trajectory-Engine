"""Real network/loss/backward smoke on isolated artificial adapter fixtures.

No real recordings or annotations are used, and no checkpoint is saved. This is
training-environment evidence, never an accuracy test or a fitted candidate.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from blurball_reviewed_dataset import ReviewedBlurBallDataset, synthetic_preview
from evaluate_blurball_candidate import build_network, import_file, sha


def experiment(args, report):
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; this smoke requires the inspected GPU network')
    fixture = args.output / 'synthetic_fixture'
    synthetic_preview(fixture, loss_choice=args.loss)
    dataset = ReviewedBlurBallDataset(fixture / 'export', fixture / 'synthetic_captures',
                                     fixture / 'splits.json', 'train', loss_choice=args.loss,
                                     as_torch=True)
    report['dataset_scope'] = 'SYNTHETIC ONLY; sharp, tiny, streak and absent adapter examples'
    report['gpu'] = torch.cuda.get_device_name()
    report['torch'] = torch.__version__
    report['checkpoint_sha256'] = sha(args.checkpoint)
    report['network_source_sha256'] = sha(args.repo / 'src/models/blurball.py')
    report['loss_source_sha256'] = sha(args.repo / 'src/losses/quality_focal_loss.py') if args.loss == 'quality_focal' else None
    report['stage'] = 'load_network'
    model = build_network(args.repo, args.checkpoint).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    report['optimizer'] = {'name': 'Adam', 'learning_rate': 1e-5, 'weight_decay': 0}
    report['parameter_count'] = sum(p.numel() for p in model.parameters())
    if args.loss == 'quality_focal':
        module = import_file('blurball_training_smoke_qfl', args.repo / 'src/losses/quality_focal_loss.py')
        criterion = module.QualityFocalLoss(beta=2, auto_weight=False, scales=[0]).cuda()
        report['loss_adapter'] = ('Direct dictionary input to the inspected QualityFocalLoss; '
                                  'upstream HeatmapLoss passes tensors to its dictionary API and is not used')
    else:
        criterion = torch.nn.MSELoss()
        report['loss_adapter'] = 'Sigmoid/clamp then MSE, matching inspected train_blur.yaml default'
    report['steps'] = []
    # Deliberately include both a fully artificial visible ball and blank target.
    for index in (0, 3):
        x, targets, provenance = dataset.sample(index)
        x = x.unsqueeze(0).cuda()
        targets = {0: targets[0].unsqueeze(0).cuda()}
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        report['stage'] = 'forward'
        outputs = model(x)
        if set(outputs) != {0} or tuple(outputs[0].shape) != (1, 3, 288, 512):
            raise RuntimeError('Network output contract differs from three-frame adapter')
        if not torch.isfinite(outputs[0]).all():
            raise RuntimeError('Nonfinite network logits')
        probabilities = torch.sigmoid(outputs[0]).clamp(1e-4, 1 - 1e-4)
        loss = criterion({0: probabilities}, targets) if args.loss == 'quality_focal' else criterion(probabilities, targets[0])
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError('Loss must be a finite scalar')
        report['stage'] = 'backward'
        loss.backward()
        missing, gradient_sq, tensors = [], 0., 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Nonfinite gradient: ' + name)
            gradient_sq += parameter.grad.detach().double().square().sum().item()
            tensors += 1
        if missing or gradient_sq <= 0:
            raise RuntimeError('Missing/zero training gradients: ' + ','.join(missing))
        report['stage'] = 'optimizer_step'
        optimizer.step()
        changed, max_delta = 0, 0.
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError('Nonfinite parameter after optimizer: ' + name)
            delta = (parameter.detach() - before[name]).abs().max().item()
            changed += delta > 0
            max_delta = max(max_delta, delta)
        if not changed:
            raise RuntimeError('Optimizer did not change any trainable parameter')
        report['steps'].append({'synthetic_kind': 'sharp' if index == 0 else 'absent',
                                'input_shape': list(x.shape), 'target_shape': list(targets[0].shape),
                                'target_positive_pixels': int(targets[0].count_nonzero().item()),
                                'loss': loss.item(), 'finite_gradient_tensors': tensors,
                                'gradient_l2': gradient_sq ** .5, 'changed_parameter_tensors': changed,
                                'max_parameter_delta': max_delta, 'target_method': provenance['target_method']})
    report['checkpoint_unchanged'] = sha(args.checkpoint) == report['checkpoint_sha256']
    if not report['checkpoint_unchanged']:
        raise RuntimeError('Original checkpoint changed during smoke')
    report['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['stage'] = 'complete'
    report['passed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('artifacts/blurball_research/source'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/blurball_research/blurball_best'))
    parser.add_argument('--loss', required=True, choices=('mse', 'quality_focal'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Artificial fixture training-environment smoke only', 'loss': args.loss,
              'passed': False, 'real_training_run': False, 'accuracy_measured': False,
              'checkpoint_saved': False, 'promotion_approved': False,
              'live_latency_measured': False, 'full_upstream_runner_verified': False}
    started = time.perf_counter()
    try:
        experiment(args, report)
    except Exception:
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['offline_smoke_wall_s'] = time.perf_counter() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
