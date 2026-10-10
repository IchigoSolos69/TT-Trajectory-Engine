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


def validate_examples(dataset, examples):
    if not isinstance(examples, (list, tuple)) or len(examples) != 2:
        raise ValueError('Smoke requires exactly one positive and one absent example')
    seen, categories = set(), []
    for entry in examples:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError('Invalid smoke example')
        index, kind = entry
        if type(index) is not int or not 0 <= index < len(dataset) or index in seen:
            raise ValueError('Smoke example indices must be unique integers within the dataset')
        seen.add(index)
        if kind in ('sharp', 'reviewed_visible'):
            categories.append('positive')
        elif kind in ('absent', 'reviewed_absent'):
            categories.append('absent')
        else:
            raise ValueError('Unknown smoke example kind')
    if sorted(categories) != ['absent', 'positive']:
        raise ValueError('Smoke requires both positive and absent supervision')


def experiment(args, report, *, dataset=None, examples=None, dataset_scope=None):
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; this smoke requires the inspected GPU network')
    if dataset is None:
        fixture = args.output / 'synthetic_fixture'
        synthetic_preview(fixture, loss_choice=args.loss)
        dataset = ReviewedBlurBallDataset(fixture / 'export', fixture / 'synthetic_captures',
                                         fixture / 'splits.json', 'train', loss_choice=args.loss,
                                         as_torch=True)
        examples = [(0, 'sharp'), (3, 'absent')]
        dataset_scope = 'SYNTHETIC ONLY; sharp, tiny, streak and absent adapter examples'
    elif not examples or not dataset_scope:
        raise ValueError('External smoke data requires explicit example selection and scope')
    validate_examples(dataset, examples)
    report['dataset_scope'] = dataset_scope
    report['gpu'] = torch.cuda.get_device_name()
    report['torch'] = torch.__version__
    report['checkpoint_sha256'] = sha(args.checkpoint)
    report['network_source_sha256'] = sha(args.repo / 'src/models/blurball.py')
    report['model_config_sha256'] = sha(args.repo / 'src/configs/model/blurball.yaml')
    report['upstream_training_config_sha256'] = sha(args.repo / 'src/configs/train_blur.yaml')
    report['adapter_source_sha256'] = sha(Path(__file__).with_name('blurball_reviewed_dataset.py'))
    report['smoke_source_sha256'] = sha(Path(__file__))
    report['loss_source_sha256'] = sha(args.repo / 'src/losses/quality_focal_loss.py') if args.loss == 'quality_focal' else None
    report['stage'] = 'load_network'
    model = build_network(args.repo, args.checkpoint).train()
    if model._out_scales != [0]:
        raise RuntimeError('Only the inspected output-scale-zero network is supported')
    # The inspected forward returns only y_list[0]. Final-stage fusion outputs
    # 1..3 are allocated/computed but never consumed by that output or its loss.
    # Their exact parameters are disconnected; other branch feature parameters
    # still feed output zero and MUST receive gradients. Do not blanket-ignore
    # None gradients or remove feature branches from the network.
    final_stage_index = len(model.stage4) - 1
    inactive = set()
    for branch, fusion in enumerate(model.stage4[-1].fuse_layers):
        if branch not in model._out_scales:
            inactive.update(f'stage4.{final_stage_index}.fuse_layers.{branch}.{name}'
                            for name, _ in fusion.named_parameters())
    report['expected_inactive_parameters'] = sorted(inactive)
    report['inactive_reason'] = 'Final-stage auxiliary fusion outputs are not returned by inspected out_scales=[0] forward'
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
    # Both paths must exercise a positive window and verified blank targets.
    for index, example_kind in examples:
        x, targets, provenance = dataset.sample(index)
        x = x.unsqueeze(0).cuda()
        targets = {0: targets[0].unsqueeze(0).cuda()}
        if tuple(x.shape) != (1, 9, 288, 512) or not torch.isfinite(x).all():
            raise RuntimeError('Invalid/nonfinite normalized RGB fixture input')
        if tuple(targets[0].shape) != (1, 3, 288, 512) or not torch.isfinite(targets[0]).all() or ((targets[0] < 0) | (targets[0] > 1)).any():
            raise RuntimeError('Invalid/nonfinite fixture heatmap target')
        positive_pixels = [int(channel.count_nonzero().item()) for channel in targets[0][0]]
        if example_kind in ('sharp', 'reviewed_visible') and not all(positive_pixels):
            raise RuntimeError('Every reviewed positive frame must retain a nonempty target')
        if example_kind in ('absent', 'reviewed_absent') and any(positive_pixels):
            raise RuntimeError('Reviewed absent frames must have exactly blank targets')
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
        if set(missing) != inactive or gradient_sq <= 0:
            raise RuntimeError('Unexpected missing/zero output-connected gradients: ' + ','.join(missing))
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
        report['steps'].append({'example_kind': example_kind, 'dataset_index': index,
                                'source_provenance': provenance,
                                'input_shape': list(x.shape), 'target_shape': list(targets[0].shape),
                                'target_positive_pixels': int(targets[0].count_nonzero().item()),
                                'positive_pixels_per_frame': positive_pixels,
                                'loss': loss.item(), 'finite_gradient_tensors': tensors,
                                'expected_inactive_gradient_tensors': len(missing),
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
