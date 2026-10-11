"""Research-only causal ball student using a local YOLO11n spatial encoder.

Input: RGB float [B,3,3,H,W] in [0,1], oldest to latest, H/W divisible by 32.
Output: current-frame heatmap logits [B,1,H,W] and visibility logits [B,1].
Actual pretrained Ultralytics blocks 0..4 provide stride-4/8 features. New
ordered temporal fusion and dense heads are randomly initialized. This is not
a compatible YOLO detector checkpoint, a fitted model, or a deployment path.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

GEOMETRY = 'reviewed_source_max_dimension_affine_512x288_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_rgb(x):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 5:
        raise ValueError('Floating RGB tensor [B,3,3,H,W] required')
    if x.shape[0] < 1 or tuple(x.shape[1:3]) != (3, 3) or min(x.shape[3:]) < 32 or any(v % 32 for v in x.shape[3:]):
        raise ValueError('Three RGB frames and spatial dimensions divisible by 32 required')
    if not torch.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('RGB values must be finite and in [0,1]')


class TemporalBallStudent(nn.Module):
    def __init__(self, checkpoint, *, freeze_encoder=True):
        super().__init__()
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file() or checkpoint.suffix.lower() != '.pt':
            raise ValueError('Existing local YOLO11n .pt checkpoint required; downloads are forbidden')
        from ultralytics import YOLO
        checkpoint_sha256 = sha(checkpoint)
        original = YOLO(str(checkpoint)).model
        if sha(checkpoint) != checkpoint_sha256:
            raise RuntimeError('Local YOLO checkpoint changed during model loading')
        blocks = list(original.model.children())[:5]
        expected = ['Conv', 'Conv', 'C3k2', 'Conv', 'C3k2']
        if (original.yaml.get('scale') != 'n' or Path(original.yaml.get('yaml_file', '')).name != 'yolo11n.yaml' or [type(m).__name__ for m in blocks] != expected or
                any(m.f != -1 for m in blocks)):
            raise ValueError('Only inspected sequential YOLO11n blocks 0..4 are supported')
        if blocks[2].cv2.conv.out_channels != 64 or blocks[4].cv2.conv.out_channels != 128:
            raise ValueError('YOLO11n feature channel contract differs')
        self.encoder = nn.ModuleList(blocks)
        self.freeze_encoder = bool(freeze_encoder)
        self.encoder.requires_grad_(not self.freeze_encoder)
        self.fuse8 = nn.Sequential(nn.Conv2d(3 * 128, 64, 1), nn.SiLU(),
                                   nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.fuse4 = nn.Sequential(nn.Conv2d(3 * 64 + 64, 32, 1), nn.SiLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.heatmap = nn.Conv2d(32, 1, 1)
        self.visibility = nn.Linear(32, 1)
        self.provenance = {'encoder': 'actual Ultralytics YOLO11n blocks 0..4',
                           'checkpoint': checkpoint.as_posix(), 'checkpoint_sha256': checkpoint_sha256,
                           'temporal_offsets': [-2, -1, 0], 'latest_frame_only': True,
                           'encoder_frozen': self.freeze_encoder,
                           'new_fusion_and_heads': 'random initialization',
                           'deployment_compatible': False}
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            # Prevent frozen pretrained BatchNorm statistics changing on tiny
            # reviewed batches or being coupled across frames during training.
            self.encoder.eval()
        return self

    def forward(self, images):
        validate_rgb(images)
        batch, _, _, height, width = images.shape
        features = images.reshape(batch * 3, 3, height, width)
        shallow = None
        for index, block in enumerate(self.encoder):
            features = block(features)
            if index == 2:
                shallow = features
        if tuple(shallow.shape[1:]) != (64, height // 4, width // 4) or tuple(features.shape[1:]) != (128, height // 8, width // 8):
            raise RuntimeError('Encoder feature geometry differs from inspected YOLO11n')
        ordered8 = features.reshape(batch, 3 * 128, height // 8, width // 8)
        ordered4 = shallow.reshape(batch, 3 * 64, height // 4, width // 4)
        fused8 = F.interpolate(self.fuse8(ordered8), size=ordered4.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse4(torch.cat((ordered4, fused8), dim=1))
        return {'heatmap_logits': F.interpolate(self.heatmap(fused), size=(height, width), mode='bilinear', align_corners=False),
                'visibility_logits': self.visibility(fused.mean((-2, -1)))}


class CausalWindow:
    """Stream adapter; timestamp is source seconds, never wall-clock time.

    Never assembles windows across sources, excessive time gaps or seeks. Pixel
    identity alone is not treated as a cut: genuinely static frames can repeat.
    The reviewed dataset separately excludes duplicate decoded review images.
    """
    def __init__(self, *, max_gap_s=.1):
        if type(max_gap_s) not in (float, int) or not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('Finite positive temporal gap bound required')
        self.max_gap_s = max_gap_s
        self.frames = deque(maxlen=3)
        self.last = None

    def reset(self):
        self.frames.clear()
        self.last = None

    def push(self, source_id, frame_id, timestamp, rgb):
        if not isinstance(source_id, str) or not source_id.strip() or type(frame_id) is not int or frame_id <= 0:
            raise ValueError('Nonempty source and positive integer frame ID required')
        if type(timestamp) not in (float, int) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Finite nonnegative source timestamp required')
        if not isinstance(rgb, torch.Tensor) or rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError('RGB frame [3,H,W] required')
        validate_rgb(rgb[None, None].expand(1, 3, -1, -1, -1))
        current = (source_id, frame_id, timestamp, tuple(rgb.shape), rgb.device, rgb.dtype)
        if self.last is not None:
            old_source, old_id, old_time, old_shape, old_device, old_dtype = self.last
            if (source_id != old_source or frame_id != old_id + 1 or timestamp <= old_time or timestamp - old_time > self.max_gap_s or
                    tuple(rgb.shape) != old_shape or rgb.device != old_device or rgb.dtype != old_dtype):
                self.reset()
        self.frames.append(rgb.detach().clone())
        self.last = current
        if len(self.frames) < 3:
            return None
        return torch.stack(tuple(self.frames)).unsqueeze(0), {'source_id': source_id,
                'source_frame_ids': [frame_id - 2, frame_id - 1, frame_id], 'timestamp': timestamp,
                'temporal_offsets': [-2, -1, 0]}


def reviewed_sample(x, targets, provenance, *, max_gap_ms=100.):
    """Adapt strict staging normalization without modifying its reviewed targets."""
    if provenance.get('partition') != 'unassigned' or provenance.get('target_method') != 'continuous_binary_capsule_v1':
        raise ValueError('Strict unassigned reviewed staging sample required')
    frames = provenance.get('frames', [])
    if len(frames) != 3 or any(f.get('output_target', {}).get('visibility') not in ('visible', 'absent') for f in frames):
        raise ValueError('All three targets must be explicitly visible or absent')
    ids = [f.get('source_frame') for f in frames]
    times = [f.get('elapsed_ms') for f in frames]
    if any(type(i) is not int or i < 1 for i in ids) or ids != list(range(ids[0], ids[0] + 3)):
        raise ValueError('Consecutive causal source frames required')
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in times) or not times[0] < times[1] < times[2]:
        raise ValueError('Strictly increasing source timestamps required')
    if type(max_gap_ms) not in (float, int) or not math.isfinite(max_gap_ms) or max_gap_ms <= 0 or any(b - a > max_gap_ms for a, b in zip(times, times[1:])):
        raise ValueError('Source timestamps exceed the explicit temporal continuity bound')
    if not isinstance(x, torch.Tensor) or tuple(x.shape) != (9, 288, 512) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Staging input must be finite normalized [9,288,512]')
    if not isinstance(targets, dict) or set(targets) != {0} or not isinstance(targets[0], torch.Tensor) or tuple(targets[0].shape) != (3, 288, 512):
        raise ValueError('Staging target must be {0: [3,288,512]}')
    y = targets[0]
    if not y.is_floating_point() or not torch.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('Finite staging target probabilities required')
    for channel, frame in zip(y, frames):
        if bool(channel.count_nonzero()) != (frame['output_target']['visibility'] == 'visible'):
            raise ValueError('Reviewed presence conflicts with heatmap target')
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    rgb = x.reshape(3, 3, 288, 512) * std + mean
    if ((rgb < -1e-6) | (rgb > 1 + 1e-6)).any():
        raise ValueError('Normalization does not recover valid RGB')
    rgb = rgb.clamp(0, 1).unsqueeze(0)
    supervision = {'heatmap': y[-1:].unsqueeze(0).clone(),
                   'visibility': y.new_tensor([[float(frames[-1]['output_target']['visibility'] == 'visible')]])}
    context = {'source_ids': [provenance['clip']], 'source_frame_ids': [ids], 'geometry': GEOMETRY}
    return rgb, supervision, context


def _probability_tensor(value, shape, name, device):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or tuple(value.shape) != tuple(shape) or
            value.device != device or not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any()):
        raise ValueError(f'{name} must be finite probability tensor with the exact output shape/device')


def loss(outputs, supervision, *, teacher=None, teacher_weight=0., context=None, purpose='training'):
    """Balanced reviewed QFL + visibility BCE; optional aligned training teacher.

    Teacher tensors are auxiliary training signals, never reference labels or
    validation/test ground truth. Only current-frame, identical-geometry causal
    teachers are accepted. A lookahead teacher cannot distill this live student.
    """
    if not isinstance(outputs, dict) or set(outputs) != {'heatmap_logits', 'visibility_logits'}:
        raise ValueError('Exact student output contract required')
    heat, visible = outputs['heatmap_logits'], outputs['visibility_logits']
    if (not isinstance(heat, torch.Tensor) or not heat.is_floating_point() or heat.ndim != 4 or heat.shape[0] < 1 or heat.shape[1] != 1 or min(heat.shape[-2:]) < 1 or
            not isinstance(visible, torch.Tensor) or not visible.is_floating_point() or tuple(visible.shape) != (heat.shape[0], 1) or
            visible.device != heat.device or not torch.isfinite(heat).all() or not torch.isfinite(visible).all()):
        raise ValueError('Finite current-frame output logits required')
    if not isinstance(supervision, dict) or set(supervision) != {'heatmap', 'visibility'}:
        raise ValueError('Reviewed heatmap and explicit visibility supervision required')
    target, presence = supervision['heatmap'], supervision['visibility']
    _probability_tensor(target, heat.shape, 'Reviewed heatmap', heat.device)
    _probability_tensor(presence, visible.shape, 'Reviewed visibility', heat.device)
    if target.requires_grad or presence.requires_grad:
        raise ValueError('Reviewed ground-truth tensors must be detached')
    if ((presence != 0) & (presence != 1)).any() or not torch.equal(target.flatten(1).gt(0).any(1), presence[:, 0].bool()):
        raise ValueError('Binary reviewed visibility must agree with target support')
    focal = F.binary_cross_entropy_with_logits(heat, target, reduction='none') * (target - heat.sigmoid()).abs().pow(2)
    support = target > 0
    positive = (focal * support).flatten(1).sum(1) / support.flatten(1).sum(1).clamp_min(1)
    background = (focal * ~support).flatten(1).sum(1) / (~support).flatten(1).sum(1).clamp_min(1)
    heat_loss = (positive + background).mean()
    visibility_loss = F.binary_cross_entropy_with_logits(visible, presence)
    if type(teacher_weight) not in (float, int) or not math.isfinite(teacher_weight) or teacher_weight < 0:
        raise ValueError('Finite nonnegative auxiliary teacher weight required')
    auxiliary = heat.sum() * 0
    if teacher is None:
        if teacher_weight != 0:
            raise ValueError('Nonzero distillation weight requires teacher provenance/tensors')
    else:
        if purpose != 'training' or teacher_weight <= 0 or context is None:
            raise ValueError('Teacher is allowed only as explicitly weighted aligned training signal')
        required = {'heatmap', 'visibility', 'model_sha256', 'source_ids', 'source_frame_ids', 'geometry', 'temporal_offsets'}
        if not isinstance(teacher, dict) or set(teacher) != required or not isinstance(teacher['model_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', teacher['model_sha256']):
            raise ValueError('Teacher model hash and exact metadata/tensor contract required')
        ids, sources = context.get('source_frame_ids'), context.get('source_ids')
        if (not isinstance(ids, list) or len(ids) != heat.shape[0] or any(not isinstance(row, list) or len(row) != 3 or
                any(type(i) is not int or i <= 0 for i in row) or row != list(range(row[0], row[0] + 3)) for row in ids) or
                not isinstance(sources, list) or len(sources) != heat.shape[0] or any(not isinstance(s, str) or not s.strip() for s in sources)):
            raise ValueError('One source identity and consecutive source-frame row required per batch item')
        if teacher['temporal_offsets'] != [-2, -1, 0] or any(teacher[key] != context.get(key) for key in ('source_ids', 'source_frame_ids', 'geometry')):
            raise ValueError('Teacher must match source frame IDs, current frame, causal history and geometry')
        _probability_tensor(teacher['heatmap'], heat.shape, 'Teacher heatmap', heat.device)
        _probability_tensor(teacher['visibility'], visible.shape, 'Teacher visibility', heat.device)
        if teacher['heatmap'].requires_grad or teacher['visibility'].requires_grad:
            raise ValueError('Teacher signals must be detached')
        auxiliary = F.binary_cross_entropy_with_logits(heat, teacher['heatmap']) + F.binary_cross_entropy_with_logits(visible, teacher['visibility'])
    total = heat_loss + visibility_loss + teacher_weight * auxiliary
    return {'total': total, 'heatmap': heat_loss, 'visibility': visibility_loss, 'distillation': auxiliary}
