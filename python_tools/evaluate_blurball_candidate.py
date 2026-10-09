"""Isolated BlurBall/YOLO comparison; proposals are never ground truth or telemetry.

Uses the official network and checkpoint, dataset RGB/affine normalization, and
step-1 component selection. Reports offline lookahead and causal results separately.
Requires torch, PyYAML, OpenCV and the existing TensorRT Python runtime for YOLO.
No upstream runner, global dependency changes, frame deletion or model promotion.
"""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from compare_lighting_filters import Detector


class Config(dict):
    """The upstream network uses both dictionary and attribute configuration access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def config_tree(value):
    if isinstance(value, dict):
        return Config({k: config_tree(v) for k, v in value.items()})
    if isinstance(value, list):
        return [config_tree(v) for v in value]
    return value


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_network(repo, checkpoint):
    cfg = config_tree(yaml.safe_load((repo / 'src/configs/model/blurball.yaml').read_text()))
    if (cfg.frames_in, cfg.frames_out, cfg.inp_width, cfg.inp_height) != (3, 3, 512, 288):
        raise ValueError('This evaluator requires the inspected three-frame BlurBall config')
    module = import_file('upstream_blurball_network', repo / 'src/models/blurball.py')
    model = module.BlurBall(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'], strict=True)
    return model.cuda().eval()


def preprocess(frame, transform):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.warpAffine(rgb, transform, (512, 288), flags=cv2.INTER_LINEAR)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]
    return (chw - mean) / std


def components(heatmap, inverse, threshold, width, height):
    """Upstream threshold/mass/centroid semantics, with source-scaled blur endpoints.

    PCA axis is undirected (180-degree ambiguity). Mass and heatmap peak are not
    calibrated probabilities. Length refers to the thresholded model heatmap.
    """
    count, labels = cv2.connectedComponents((heatmap > threshold).astype(np.uint8))
    results = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        weights = heatmap[ys, xs]
        mass = float(weights.sum())
        center = np.array([(xs * weights).sum() / mass, (ys * weights).sum() / mass])
        source = inverse @ np.r_[center, 1.]
        if not (0 <= source[0] < width and 0 <= source[1] < height):
            continue
        coords = np.column_stack([xs, ys]).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(coords, mean=None)
        axis = eigenvectors[0] if len(eigenvectors) else np.array([1., 0.])
        projected = (coords - mean) @ axis
        half_length = float((projected.max() - projected.min()) / 2)
        endpoints = [inverse @ np.r_[center + sign * half_length * axis, 1.]
                     for sign in (-1, 1)]
        vector = endpoints[1] - endpoints[0]
        results.append({'center_px': source.tolist(), 'component_mass': mass,
                        'heatmap_peak': float(weights.max()),
                        'blur_extent_source_px': float(np.linalg.norm(vector)),
                        'blur_axis_deg': float(np.degrees(np.arctan2(vector[1], vector[0])) % 180),
                        'blur_endpoints_px': [p.tolist() for p in endpoints],
                        'confidence': None, 'confidence_reason': 'Uncalibrated heatmap evidence'})
    return results


def select(candidates, previous, max_disp):
    if previous is not None:
        candidates = [c for c in candidates if np.linalg.norm(
            np.array(c['center_px']) - previous['center_px']) < max_disp]
    # Stable max retains earliest candidate on equal mass, as the upstream tracker.
    return max(candidates, key=lambda c: c['component_mass'], default=None)


def draw(frame, row):
    image = frame.copy()
    for key, color, text in [('yolo', (0, 180, 255), 'YOLO'),
                             ('blurball_offline', (255, 255, 0), 'BlurBall offline'),
                             ('blurball_causal', (255, 80, 255), 'BlurBall causal')]:
        candidate = row[key]
        if candidate is None:
            continue
        x, y = map(round, candidate['center_px'])
        cv2.circle(image, (x, y), 7, color, 2)
        cv2.putText(image, text, (x + 10, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        if 'blur_endpoints_px' in candidate:
            a, b = [tuple(map(round, p)) for p in candidate['blur_endpoints_px']]
            cv2.line(image, a, b, color, 2)
    cv2.putText(image, f"frame {row['frame_id']} | UNVALIDATED MODEL PROPOSALS", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    return image


@torch.inference_mode()
def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    model = build_network(args.repo, args.checkpoint)
    affine_module = import_file('upstream_blurball_affine', args.repo / 'src/utils/image.py')
    baseline = Detector(args.yolo_engine)
    capture = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open source video: {args.video}')
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not np.isfinite(fps):
        raise RuntimeError('Source FPS unavailable; do not invent timing')
    rows, all_candidates = [], []
    buffer = deque(maxlen=3)
    previous_hash = None
    duplicates = []
    gpu_ms = []
    previous_causal = None
    width = height = None
    started = time.perf_counter()
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if width is None:
                height, width = frame.shape[:2]
                center = np.array([width / 2., height / 2.], np.float32)
                transform = affine_module.get_affine_transform(center, max(width, height), 0, [512, 288])
                inverse = affine_module.get_affine_transform(center, max(width, height), 0, [512, 288], inv=1)
                example = torch.from_numpy(np.concatenate([preprocess(frame, transform)] * 3))[None].cuda()
                for _ in range(5):
                    model(example)
                torch.cuda.synchronize()
            if frame.shape[:2] != (height, width):
                raise RuntimeError('Variable image dimensions unsupported')
            frame_id = len(rows) + 1
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            digest = hashlib.sha256(frame).hexdigest()
            if digest == previous_hash:
                duplicates.append(frame_id)
            previous_hash = digest
            proposal = baseline.infer(frame)
            rows.append({'frame_id': frame_id, 'source_pts_s': pts,
                         'yolo': proposal if proposal['accepted_at_0_25'] else None,
                         'yolo_top_proposal': proposal,
                         'blurball_causal': None, 'blurball_offline': None,
                         'causal_reason': 'NEEDS_THREE_SOURCE_FRAMES' if frame_id < 3 else 'NO_COMPONENT'})
            all_candidates.append([])
            buffer.append(preprocess(frame, transform))
            if len(buffer) == 3:
                tensor = torch.from_numpy(np.concatenate(buffer))[None].cuda()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                output = model(tensor)[0]
                end.record()
                end.synchronize()
                gpu_ms.append(begin.elapsed_time(end))
                if tuple(output.shape) != (1, 3, 288, 512) or not torch.isfinite(output).all():
                    raise RuntimeError('Unexpected/nonfinite BlurBall output')
                heatmaps = output.sigmoid().float().cpu().numpy()[0]
                for channel in range(3):
                    candidates = components(heatmaps[channel], inverse, args.threshold, width, height)
                    for candidate in candidates:
                        candidate['window_start_frame'] = frame_id - 2
                        candidate['output_channel'] = channel
                        candidate['lookahead_frames'] = 2 - channel
                    all_candidates[frame_id - 3 + channel].extend(candidates)
                    if channel == 2:
                        previous_causal = select(candidates, previous_causal, args.max_disp)
                        rows[-1]['blurball_causal'] = previous_causal
                        rows[-1]['causal_reason'] = 'MODEL_COMPONENT' if previous_causal else 'NO_COMPONENT'
            if frame_id % 50 == 0:
                print(f'{args.video.name}: decoded {frame_id}/{reported}', flush=True)
    finally:
        capture.release()
    if len(rows) != reported:
        raise RuntimeError(f'Decodable frame mismatch: decoded={len(rows)}, reported={reported}')
    previous_offline = None
    for row, candidates in zip(rows, all_candidates):
        previous_offline = select(candidates, previous_offline, args.max_disp)
        row['blurball_offline'] = previous_offline
        row['offline_candidate_count'] = len(candidates)
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    # Second sequential decode only for comparison imagery; never inference on screenshots.
    capture = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    target_ids = set(np.linspace(1, len(rows), min(12, len(rows)), dtype=int))
    panels = []
    try:
        for row in rows:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Comparison decode ended early')
            rendered = draw(frame, row)
            if writer is None:
                writer = cv2.VideoWriter(str(args.output / 'comparison.avi'),
                                        cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('Comparison video encoder unavailable')
            writer.write(rendered)
            if row['frame_id'] in target_ids:
                panels.append(cv2.resize(rendered, (640, 360)))
        while len(panels) % 3:
            panels.append(np.zeros((360, 640, 3), np.uint8))
        if panels and not cv2.imwrite(str(args.output / 'comparison_sheet.jpg'), np.vstack(
                [np.hstack(panels[i:i+3]) for i in range(0, len(panels), 3)])):
            raise RuntimeError('Cannot save comparison sheet')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    report = {'source_path': str(args.video.resolve()), 'source_sha256': sha(args.video),
              'source_fps': fps, 'source_size': [width, height], 'decoded_frames': len(rows),
              'exact_consecutive_duplicate_frames': duplicates,
              'duplicates_removed': False, 'analytical_frames_skipped': 0,
              'blurball_checkpoint_sha256': sha(args.checkpoint),
              'blurball_network_source_sha256': sha(args.repo / 'src/models/blurball.py'),
              'yolo_engine_sha256': sha(args.yolo_engine),
              'settings': {'heatmap_threshold': args.threshold, 'yolo_threshold': .25,
                           'tracker_max_displacement_source_px': args.max_disp,
                           'preprocess': 'Dataset RGB affine warp + ImageNet normalization'},
              'proposal_frames': {key: sum(row[key] is not None for row in rows)
                                  for key in ['yolo', 'blurball_offline', 'blurball_causal']},
              'causal_warmup_frames': min(len(rows), 2),
              'offline_lookahead_limit_frames': 2,
              'offline_gpu_forward_ms': {'passes': len(gpu_ms), 'median': float(np.median(gpu_ms)) if gpu_ms else None,
                                         'p95': float(np.percentile(gpu_ms, 95)) if gpu_ms else None},
              'offline_wall_s': time.perf_counter() - started,
              'live_capture_to_display_latency_ms': None,
              'accuracy': None, 'accuracy_reason': 'No independently reviewed ground truth for these source clips',
              'promoted': False, 'trajectory_basis': 'Image pixel proposals only; no physical 3D or spin inference'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', type=Path, default=Path('artifacts/blurball_research/source'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/blurball_research/blurball_best'))
    parser.add_argument('--yolo-engine', type=Path, default=Path('models/yolov11_ttgames.engine'))
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--max-disp', type=float, default=300.)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.max_disp <= 0:
        parser.error('Threshold must be in (0,1) and displacement must be positive')
    evaluate(args)
