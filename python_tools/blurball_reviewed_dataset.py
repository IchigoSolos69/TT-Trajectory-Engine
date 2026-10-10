"""Audited reviewed-streak dataset for BlurBall; no training or relabeling.

Map-style dataset returns (RGB float32 [9,288,512], {0: targets[3,288,512]}).
Use as_torch=True for PyTorch's DataLoader. sample(index) also returns provenance.
The binary capsule target is an explicit versioned correction to the upstream
discrete-line target, not a claim of exact target/paper reproduction.

audit EXPORT CAPTURES SPLITS --loss quality_focal|mse is read-only.
synthetic-preview NEW_DIR --loss ... creates isolated synthetic test evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from export_reviewed_streaks import audit as audit_captures, export, target

ROOT = Path(__file__).resolve().parents[1]
INPUT_WH = (512, 288)
MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
TARGET_METHOD = 'continuous_binary_capsule_v1'
SOURCE_COMMIT = '2f0f5496f7ba4b5b1a36790749935121b2ce972d'
SPLITS = ('train', 'val', 'test')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_jsonl(path):
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise ValueError(f'JSONL must be nonempty and newline-terminated: {path}')
    lines = raw.decode('utf-8').splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f'Blank JSONL row: {path}')
    return [json.loads(line) for line in lines]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_to_output_affine(dimensions, wh=INPUT_WH):
    """Exact upstream zero-rotation max-dimension affine, not aspect-fit resize."""
    if len(dimensions) != 2 or any(type(v) is not int or v <= 0 for v in dimensions):
        raise ValueError('Positive integer source dimensions required')
    width, height = dimensions
    outw, outh = wh
    # Build the same float32 three-point construction as upstream image.py.
    center = np.array([width / 2., height / 2.], dtype=np.float32)
    size = max(width, height)
    src = np.array([center, center + [0., -size / 2.], center + [-size / 2., -size / 2.]], dtype=np.float32)
    dst = np.array([[outw / 2., outh / 2.], [outw / 2., outh / 2. - outw / 2.], [0., outh / 2. - outw / 2.]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def transform_target(label, dimensions, wh=INPUT_WH):
    visibility = label.get('visibility')
    if visibility == 'absent':
        if any(label.get(k) is not None for k in ('position_px', 'half_length_px', 'axis_rad')):
            raise ValueError('Absent target conflicts with geometry')
        return {'visibility': 'absent', 'center_px': None, 'endpoints_px': None,
                'half_length_px': None, 'axis_deg': None, 'axis_directed': False}
    if visibility != 'visible' or label.get('method') != 'manual_streak_endpoints':
        raise ValueError('Only reviewed fully visible streaks or explicit absence are eligible')
    points = np.array(label['endpoints_normalized'], dtype=np.float64)
    if points.shape != (2, 2) or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError('Invalid normalized endpoints')
    source = points * np.array(dimensions)
    matrix = source_to_output_affine(dimensions, wh)
    output = np.c_[source, np.ones(2)] @ matrix.T
    center = output.mean(axis=0)
    # The normalized GUI domain includes the image edge at 1; a center on or
    # beyond that boundary is not a valid output pixel. Never turn it blank.
    if not (0 <= center[0] < wh[0] and 0 <= center[1] < wh[1]):
        raise ValueError('Visible target center is outside retained output crop')
    difference = output[1] - output[0]
    half_length = float(np.linalg.norm(difference) / 2)
    angle = None if half_length == 0 else math.degrees(math.atan2(difference[1], difference[0])) % 180
    return {'visibility': visibility, 'center_px': center.tolist(), 'endpoints_px': output.tolist(),
            'half_length_px': half_length, 'axis_deg': angle, 'axis_directed': False}


def line_target(transformed, wh=INPUT_WH):
    """Distance-to-segment binary capsule; label extent never rounded/inflated.

    Positive streak radius 2.5 output pixels; sharp point radius 3, matching the
    inspected upstream sharp-point special case. Unlike its range(int(l))
    sampling, fractional positive streaks remain visible targets.
    """
    width, height = wh
    if transformed['visibility'] == 'absent':
        return np.zeros((height, width), dtype=np.float32)
    a, b = np.array(transformed['endpoints_px'], dtype=np.float64)
    vector = b - a
    length2 = float(vector @ vector)
    ys, xs = np.mgrid[:height, :width]
    if length2 == 0:
        distance2 = (xs - a[0]) ** 2 + (ys - a[1]) ** 2
        radius = 3.
    else:
        fraction = np.clip(((xs - a[0]) * vector[0] + (ys - a[1]) * vector[1]) / length2, 0, 1)
        distance2 = (xs - (a[0] + fraction * vector[0])) ** 2 + (ys - (a[1] + fraction * vector[1])) ** 2
        radius = 2.5
    result = (distance2 <= radius ** 2).astype(np.float32)
    if not result.any():
        raise ValueError('Visible target must not produce a blank absence heatmap')
    return result


def loss_config(choice):
    if choice == 'quality_focal':
        return {'name': 'heatmap', 'sub_name': 'quality_focal', 'auto_weight': False, 'scales': [0], 'beta': 2.}
    if choice == 'mse':
        return {'name': 'heatmap', 'sub_name': 'mse'}
    raise ValueError('Choose loss explicitly: quality_focal (paper) or mse (upstream train_blur default)')


def validate_source_records(captures, split_file):
    """Strict preflight missing from the older handoff exporter; read-only."""
    splits = json.loads(Path(split_file).read_text(encoding='utf-8'))
    if not isinstance(splits, dict) or set(splits) != set(SPLITS) or any(not isinstance(splits[s], list) or not splits[s] for s in SPLITS):
        raise ValueError('Independent nonempty train/val/test capture lists required')
    root = Path(captures).resolve()
    decoded_splits, source_splits, sessions, seen_clips = {}, {}, {}, set()
    for split in SPLITS:
        for clip in splits[split]:
            if not isinstance(clip, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', clip) or clip in seen_clips:
                raise ValueError('Invalid or repeated capture ID')
            seen_clips.add(clip)
            folder = (root / clip).resolve()
            if folder.parent != root:
                raise ValueError('Capture path escapes root')
            capture = json.loads((folder / 'capture.json').read_text(encoding='utf-8'))
            if not isinstance(capture, dict):
                raise ValueError('Capture metadata dictionary required')
            if capture.get('split') is not None and capture['split'] != split:
                raise ValueError('Declared capture split conflicts with frozen split manifest')
            session = capture.get('session_id')
            if not isinstance(session, str) or not session.strip():
                raise ValueError('Explicit complete-session identity required')
            if session in sessions and sessions[session] != split:
                raise ValueError('Capture session overlaps dataset splits')
            sessions[session] = split
            original_hash = capture.get('original_sha256')
            if original_hash is not None:
                if not isinstance(original_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', original_hash):
                    raise ValueError('Invalid original source hash')
                if original_hash in source_splits and source_splits[original_hash] != split:
                    raise ValueError('Original recording overlaps dataset splits')
                source_splits[original_hash] = split
                original = capture.get('original_path')
                if not isinstance(original, str) or not Path(original).is_file() or sha(original) != original_hash:
                    raise ValueError('Original recording hash/path does not match capture provenance')
            annotations = json.loads((folder / 'annotations.json').read_text(encoding='utf-8'))
            if not isinstance(annotations, dict):
                raise ValueError('Annotation dictionary required')
            rows = read_jsonl(folder / 'frames.jsonl')
            declared_count = capture.get('sampled_frames')
            if declared_count is not None and (type(declared_count) is not int or declared_count != len(rows)):
                raise ValueError('Declared sampled frame count does not match complete index')
            filenames, ids = set(), set()
            previous_id, previous_time = None, None
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError('Frame index row dictionary required')
                name, fid, elapsed = row.get('file'), row.get('source_frame'), row.get('elapsed_ms')
                if not isinstance(name, str) or not re.fullmatch(r'frame_[0-9]+\.jpg', name) or name in filenames:
                    raise ValueError('Invalid or duplicate frame filename')
                if type(fid) is not int or fid <= 0 or fid in ids:
                    raise ValueError('Positive unique one-based source frame IDs required')
                if not finite_number(elapsed) or elapsed < 0:
                    raise ValueError('Finite nonnegative source timestamp required')
                if previous_id is not None and (fid <= previous_id or elapsed <= previous_time):
                    raise ValueError('Source index IDs/timestamps must be strictly chronological')
                previous_id, previous_time = fid, elapsed
                burst = row.get('burst_id')
                if not ((type(burst) is int and burst > 0) or (isinstance(burst, str) and burst.strip())):
                    raise ValueError('Explicit burst identity required')
                filenames.add(name); ids.add(fid)
                annotation = annotations.get(name, {})
                if not isinstance(annotation, dict):
                    raise ValueError('Invalid annotation row')
                flags = annotation.get('reviewed_classes')
                if flags is not None and (not isinstance(flags, dict) or any(type(flags.get(k)) is not bool for k in ('ball', 'person', 'table'))):
                    raise ValueError('Explicit boolean class-review flags required')
                if 'source_frame' in annotation and (type(annotation['source_frame']) is not int or annotation['source_frame'] != fid):
                    raise ValueError('Reviewed annotation source frame differs from index')
                if 'elapsed_ms' in annotation and (not finite_number(annotation['elapsed_ms']) or abs(annotation['elapsed_ms'] - elapsed) > 1e-6):
                    raise ValueError('Reviewed annotation timestamp differs from index')
                geometry = annotation.get('geometry') or {}
                if not isinstance(geometry, dict):
                    raise ValueError('Annotation geometry dictionary required')
                if geometry.get('visibility') == 'absent' and annotation.get('ball') != []:
                    raise ValueError('Absent-ball annotation requires an explicit empty ball list')
                image_path = (folder / name).resolve()
                if image_path.parent != folder:
                    raise ValueError('Frame path escapes recording')
                with Image.open(image_path) as image:
                    if image.format != 'JPEG':
                        raise ValueError('Expected original JPEG frame')
                    rgb = image.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    if digest in decoded_splits and decoded_splits[digest] != split:
                        raise ValueError('Identical decoded RGB image occurs across dataset splits')
                    decoded_splits[digest] = split
    return {'sessions': sessions, 'decoded_unique_images': len(decoded_splits)}


def audit_export(export_dir, captures, split_file, *, loss_choice):
    """Rebuild the handoff from original labels and compare every row and hash."""
    selected_loss = loss_config(loss_choice)
    source_check = validate_source_records(captures, split_file)
    fresh = audit_captures(Path(captures), Path(split_file))
    if not fresh['ready_for_training']:
        raise ValueError('No complete reviewed windows in every independent split; unknowns cannot train')
    directory = Path(export_dir).resolve()
    summary = json.loads((directory / 'audit.json').read_text(encoding='utf-8'))
    expected = {k: v for k, v in fresh.items() if k != 'windows'}
    if canonical(summary) != canonical(expected):
        raise ValueError('Export audit lineage/settings no longer match original captures or split manifest')
    accepted, exclusions, file_hashes = {}, Counter(), {'audit.json': sha(directory / 'audit.json')}
    for split in SPLITS:
        rows = read_jsonl(directory / (split + '.jsonl'))
        expected_rows = [r for r in fresh['windows'] if r['split'] == split]
        if canonical(rows) != canonical(expected_rows):
            raise ValueError('Export window/target/source provenance does not match audited original labels')
        eligible = []
        for row in rows:
            if any(frame['target']['visibility'] == 'partial' for frame in row['frames']):
                exclusions['partial_streak_window_excluded'] += 1
                continue
            for frame in row['frames']:
                transform_target(frame['target'], (frame['source_width'], frame['source_height']))
            eligible.append(row)
        if not eligible:
            raise ValueError(f'No fully reviewed visible/absent windows remain for independent {split}')
        accepted[split] = eligible
        file_hashes[split + '.jsonl'] = sha(directory / (split + '.jsonl'))
    manifest = {'schema_version': 1, 'adapter': 'reviewed_blurball_dataset_v1',
                'source_commit': SOURCE_COMMIT, 'model_input_shape': [9, 288, 512],
                'target_shape': [3, 288, 512], 'input_order': 'oldest_to_latest_RGB_CHW',
                'normalization': {'mean': MEAN[:, 0, 0].tolist(), 'std': STD[:, 0, 0].tolist()},
                'affine': 'upstream_uniform_max_dimension_zero_rotation',
                'target_method': TARGET_METHOD, 'streak_radius_output_px': 2.5,
                'sharp_radius_output_px': 3., 'loss': selected_loss,
                'loss_selection_note': 'quality_focal is paper-described; train_blur default is mse; neither runs here',
                'target_difference': 'Continuous segment, no integer-length truncation; preserves exact reviewed geometry. Not upstream discrete-target bitwise parity.',
                'augmentation': 'none', 'partial_policy': 'exclude_unresolved_truncated_streak_targets',
                'accepted_counts': {s: len(accepted[s]) for s in SPLITS},
                'adapter_excluded': dict(exclusions), 'source_checks': source_check,
                'capture_lineage': fresh['capture_lineage'],
                'split_manifest_sha256': fresh['split_manifest_sha256'], 'export_hashes': file_hashes,
                'scope': 'Structural training handoff only; synthetic evidence is not real supervision, sufficient sample size, accuracy or model promotion.'}
    return manifest, accepted


class ReviewedBlurBallDataset:
    """PyTorch-compatible map dataset with no stochastic augmentation or fitting."""
    def __init__(self, export_dir, captures, split_file, split, *, loss_choice, as_torch=False):
        if split not in SPLITS:
            raise ValueError('Invalid split')
        self.manifest, rows = audit_export(export_dir, captures, split_file, loss_choice=loss_choice)
        self._rows = copy.deepcopy(rows[split])
        self._export = Path(export_dir).resolve()
        self._captures = Path(captures).resolve()
        self._split_file = Path(split_file).resolve()
        self._as_torch = as_torch
        self._lineage = {row['clip']: row for row in self.manifest['capture_lineage']}

    def __len__(self):
        return len(self._rows)

    def _validate_current(self, row):
        if sha(self._split_file) != self.manifest['split_manifest_sha256']:
            raise ValueError('Dataset split manifest changed after audit')
        for name, expected in self.manifest['export_hashes'].items():
            if sha(self._export / name) != expected:
                raise ValueError('Export changed after audit')
        lineage = self._lineage[row['clip']]
        for name, key in [('annotations.json', 'annotations_sha256'), ('capture.json', 'capture_sha256'), ('frames.jsonl', 'index_sha256')]:
            if sha(self._captures / row['clip'] / name) != lineage[key]:
                raise ValueError('Source annotation/capture/index changed after audit')

    def sample(self, index):
        row = self._rows[index]
        self._validate_current(row)
        inputs, targets, geometries = [], [], []
        for frame in row['frames']:
            path = Path(frame['image'])
            if sha(path) != frame['sha256']:
                raise ValueError('Source image changed after audit')
            with Image.open(path) as image:
                dimensions = image.size
                if image.format != 'JPEG' or dimensions != (frame['source_width'], frame['source_height']):
                    raise ValueError('Source image type/dimensions changed after audit')
                rgb = np.array(image.convert('RGB'))
            affine = source_to_output_affine(dimensions)
            resized = cv2.warpAffine(rgb, affine, INPUT_WH, flags=cv2.INTER_LINEAR)
            chw = resized.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
            inputs.append((chw - MEAN) / STD)
            geometry = transform_target(frame['target'], dimensions)
            targets.append(line_target(geometry))
            geometries.append({'source_target': copy.deepcopy(frame['target']), 'output_target': geometry,
                               'source_frame': frame['source_frame'], 'elapsed_ms': frame['elapsed_ms'],
                               'source_image_sha256': frame['sha256'], 'source_dimensions': list(dimensions),
                               'source_to_output_affine': affine.tolist()})
        x, y = np.ascontiguousarray(np.concatenate(inputs)), np.ascontiguousarray(np.stack(targets))
        provenance = {'split': row['split'], 'clip': row['clip'], 'session_id': row['session_id'],
                      'latest_source_frame': row['latest_source_frame'], 'frames': geometries,
                      'target_method': TARGET_METHOD, 'loss': copy.deepcopy(self.manifest['loss'])}
        if self._as_torch:
            import torch
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, {0: y}, provenance

    def __getitem__(self, index):
        x, y, _ = self.sample(index)
        return x, y


def synthetic_preview(output, *, loss_choice):
    """Generate isolated artificial examples; never read/edit the real captures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    captures = output / 'synthetic_captures'; captures.mkdir()
    splits = {s: ['synthetic_' + s] for s in SPLITS}
    split_file = output / 'splits.json'; split_file.write_text(json.dumps(splits, indent=2) + '\n')
    kinds = ('sharp', 'tiny_positive', 'diagonal', 'absent')
    for split_index, split in enumerate(SPLITS):
        folder = captures / ('synthetic_' + split); folder.mkdir()
        (folder / 'capture.json').write_text(json.dumps({'source': 'recorded_camera', 'session_id': 'SYNTHETIC_ONLY_' + split,
                                                       'purpose': 'Isolated adapter test, not real training data'}) + '\n')
        rows, annotations = [], {}
        for kind_index, kind in enumerate(kinds):
            for time in range(3):
                fid = kind_index * 3 + time + 1
                filename = f'frame_{fid}.jpg'
                image = Image.new('RGB', (1920, 1080), (25 + split_index * 35, 30 + time * 20, 40 + kind_index * 15))
                draw = ImageDraw.Draw(image); draw.text((50, 40), f'SYNTHETIC ONLY {split} {kind} t={time}', fill='white')
                center = (.4 + time * .02, .4)
                offset = (0., 0.) if kind == 'sharp' else ((.0004, 0.) if kind == 'tiny_positive' else (.04, .025))
                endpoints = [[center[0]-offset[0], center[1]-offset[1]], [center[0]+offset[0], center[1]+offset[1]]]
                if kind != 'absent':
                    xy = [(x * 1920, y * 1080) for x, y in endpoints]
                    draw.line(xy, fill='white', width=5); draw.ellipse((xy[0][0]-2,xy[0][1]-2,xy[0][0]+2,xy[0][1]+2), fill='white')
                image.save(folder / filename, quality=95)
                elapsed = (fid - 1) * 1000 / 60
                rows.append({'file': filename, 'source_frame': fid, 'elapsed_ms': elapsed, 'burst_id': kind_index + 1})
                geometry = {'visibility': 'absent' if kind == 'absent' else 'visible', 'blur_streak': None if kind == 'absent' else
                            {'endpoints': endpoints, 'source_width': 1920, 'source_height': 1080, 'reviewed': True, 'method': 'manual_streak_endpoints'}}
                annotations[filename] = {'reviewed': True, 'reviewed_classes': {'ball': True, 'person': False, 'table': False},
                                         'ball': [], 'person': [], 'table': [], 'source_frame': fid, 'elapsed_ms': elapsed, 'geometry': geometry}
        (folder / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (folder / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n')
    export_dir = output / 'export'; export(captures, split_file, export_dir)
    dataset = ReviewedBlurBallDataset(export_dir, captures, split_file, 'train', loss_choice=loss_choice)
    report = copy.deepcopy(dataset.manifest)
    report['scope'] = 'SYNTHETIC ONLY: target/loader preview, no real labels, training or weights.'
    report['examples'] = []
    for index in range(len(dataset)):
        x, targets, provenance = dataset.sample(index)
        canvas = Image.new('RGB', (1024, 3 * 312), (10, 15, 24))
        draw = ImageDraw.Draw(canvas)
        for channel in range(3):
            recovered = ((x[channel*3:channel*3+3] * STD + MEAN) * 255).round().clip(0, 255).astype(np.uint8).transpose(1,2,0)
            rgb = Image.fromarray(recovered)
            heat = Image.fromarray((targets[0][channel] * 255).astype(np.uint8)).convert('RGB')
            canvas.paste(rgb, (0, channel * 312 + 24)); canvas.paste(heat, (512, channel * 312 + 24))
            geometry = provenance['frames'][channel]['output_target']
            draw.text((8, channel * 312 + 6), f'SYNTHETIC t{channel}: RGB input | {TARGET_METHOD}; l={geometry["half_length_px"]}', fill='white')
        name = f'preview_{kinds[index]}.png'; canvas.save(output / name)
        report['examples'].append({'preview': name, 'input_shape': list(x.shape), 'target_shape': list(targets[0].shape), 'provenance': provenance})
    (output / 'adapter_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('audit'); check.add_argument('export', type=Path); check.add_argument('captures', type=Path); check.add_argument('splits', type=Path)
    preview = commands.add_parser('synthetic-preview'); preview.add_argument('output', type=Path)
    for command in (check, preview):
        command.add_argument('--loss', required=True, choices=('quality_focal', 'mse'))
    args = parser.parse_args()
    if args.command == 'audit':
        result, _ = audit_export(args.export, args.captures, args.splits, loss_choice=args.loss)
    else:
        result = synthetic_preview(args.output, loss_choice=args.loss)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
