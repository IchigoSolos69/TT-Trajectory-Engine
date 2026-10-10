"""Isolated fixed-shape BlurBall ONNX/TensorRT FP32 export and parity experiment.

Never modifies deployed models. Uses consecutive original video frames and reports
offline GPU forward timings separately from unavailable live end-to-end latency.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import onnx
import tensorrt as trt
import torch

from evaluate_blurball_candidate import build_network, components, import_file, preprocess, sha


class HeatmapOutput(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, tensor):
        return self.network(tensor)[0]


def source_windows(video, repo):
    """Decode sequentially and preserve three strictly consecutive source frames."""
    affine_module = import_file('blurball_export_affine', repo / 'src/utils/image.py')
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot decode original source {video}')
    targets = {3, 4, 5, 40, 41, 42, 80, 81, 82}
    buffer = deque(maxlen=3)
    records = []
    index = 0
    try:
        while index < max(targets):
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            height, width = frame.shape[:2]
            if index == 1:
                center = np.array([width / 2., height / 2.], np.float32)
                transform = affine_module.get_affine_transform(center, max(width, height), 0, [512, 288])
                inverse = affine_module.get_affine_transform(center, max(width, height), 0, [512, 288], inv=1)
                shape = frame.shape[:2]
            if frame.shape[:2] != shape:
                raise RuntimeError('Source dimensions changed within a temporal window')
            buffer.append(preprocess(frame, transform))
            if index in targets:
                if len(buffer) != 3:
                    raise RuntimeError('Incomplete real source-frame window')
                records.append({'source': str(video.resolve()), 'source_frame_ids': [index - 2, index - 1, index],
                                'source_pts_s': capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.,
                                'source_size': [width, height], 'inverse': inverse,
                                'tensor': torch.from_numpy(np.concatenate(buffer))[None].cuda()})
    finally:
        capture.release()
    if len(records) < 3:
        raise RuntimeError(f'Need at least three real consecutive windows for {video}')
    return records


def build_engine(onnx_path, engine_path):
    """TensorRT 11 uses strongly typed ONNX tensors; do not use removed FP16 flags."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'TensorRT ONNX parser rejected the model:\n{errors}')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
    config.clear_flag(trt.BuilderFlag.TF32)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT did not produce a serialized engine; inspect complete builder log')
    engine_path.write_bytes(bytes(serialized))
    return {'build_wall_s': time.perf_counter() - started, 'workspace_limit_bytes': 512 * 1024 * 1024,
            'precision': 'Strongly typed FP32 ONNX tensors; TensorRT TF32 disabled',
            'parser_errors': [str(parser.get_error(i)) for i in range(parser.num_errors)]}


class EngineRunner:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError('Exported TensorRT engine could not deserialize')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Exported TensorRT execution context unavailable')
        self.stream = torch.cuda.Stream()
        self.tensors = {}
        self.contract = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dimension < 1 for dimension in shape):
                raise RuntimeError(f'Unexpected dynamic TensorRT shape {name}: {shape}')
            dtype = self.engine.get_tensor_dtype(name)
            if dtype != trt.float32:
                raise RuntimeError(f'Expected FP32 TensorRT tensor {name}, received {dtype}')
            self.tensors[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
            if not self.context.set_tensor_address(name, self.tensors[name].data_ptr()):
                raise RuntimeError(f'Could not bind TensorRT tensor {name}')
            mode = self.engine.get_tensor_mode(name)
            self.contract.append({'name': name, 'shape': shape, 'dtype': str(dtype), 'mode': str(mode)})
            if mode == trt.TensorIOMode.INPUT:
                self.input = name
            else:
                self.output = name
        if tuple(self.tensors[self.input].shape) != (1, 9, 288, 512):
            raise RuntimeError('Exported engine input contract mismatch')
        if tuple(self.tensors[self.output].shape) != (1, 3, 288, 512):
            raise RuntimeError('Exported engine output contract mismatch')

    def run(self, tensor):
        # The input was produced on the current Torch stream. Honor that producer.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.tensors[self.input].copy_(tensor)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                raise RuntimeError('Exported TensorRT execution failed')
            end.record(self.stream)
        end.synchronize()
        return self.tensors[self.output].clone(), begin.elapsed_time(end)


def compare_components(reference, candidate, window):
    width, height = window['source_size']
    deltas, counts, unmatched = [], [], 0
    for channel in range(3):
        expected = components(reference[channel], window['inverse'], .7, width, height)
        actual = components(candidate[channel], window['inverse'], .7, width, height)
        counts.append({'channel': channel, 'pytorch': len(expected), 'tensorrt': len(actual)})
        unmatched += abs(len(expected) - len(actual))
        available = list(actual)
        # One-to-one nearest matching avoids a component-mass ordering tie artifact.
        for item in expected:
            if not available:
                break
            distances = [np.linalg.norm(np.array(item['center_px']) - other['center_px']) for other in available]
            match = int(np.argmin(distances))
            deltas.append(float(distances[match]))
            available.pop(match)
    return {'counts': counts, 'unmatched_components': unmatched,
            'max_centroid_delta_source_px': max(deltas, default=0.), 'matched_components': len(deltas)}


@torch.inference_mode()
def experiment(args, report):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = HeatmapOutput(build_network(args.repo, args.checkpoint)).cuda().eval()
    onnx_path, engine_path = args.output / 'blurball_fp32.onnx', args.output / 'blurball_fp32.engine'
    example = torch.zeros((1, 9, 288, 512), dtype=torch.float32, device='cuda')
    report['stage'] = 'onnx_export'
    print('Exporting the inspected fixed-shape network to ONNX...', flush=True)
    torch.onnx.export(model, example, str(onnx_path), input_names=['frames'], output_names=['heatmap_logits'],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(onnx_path))
    report['onnx_sha256'] = sha(onnx_path)
    report['stage'] = 'tensorrt_build'
    print('Building an isolated strongly typed FP32 TensorRT engine...', flush=True)
    report['engine_build'] = build_engine(onnx_path, engine_path)
    report['engine_sha256'] = sha(engine_path)
    runner = EngineRunner(engine_path)
    report['engine_io'] = runner.contract
    report['stage'] = 'real_window_parity'
    windows = [window for video in args.videos for window in source_windows(video, args.repo)]
    for _ in range(3):
        model(windows[0]['tensor'])
        runner.run(windows[0]['tensor'])
    torch.cuda.synchronize()
    comparisons = []
    for window in windows:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        reference = model(window['tensor'])
        end.record()
        end.synchronize()
        reference_ms = begin.elapsed_time(end)
        actual, actual_ms = runner.run(window['tensor'])
        if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
            raise RuntimeError('Nonfinite parity output')
        difference = (reference - actual).abs()
        reference_hm = reference.sigmoid().cpu().numpy()[0]
        actual_hm = actual.sigmoid().cpu().numpy()[0]
        comparison = {'source': window['source'], 'source_frame_ids': window['source_frame_ids'],
                      'max_logit_error': difference.max().item(), 'mean_logit_error': difference.mean().item(),
                      'max_heatmap_probability_error': float(np.abs(reference_hm - actual_hm).max()),
                      'pytorch_gpu_forward_ms': reference_ms, 'tensorrt_gpu_forward_ms': actual_ms,
                      **compare_components(reference_hm, actual_hm, window)}
        comparisons.append(comparison)
        print(f"{Path(window['source']).name} {window['source_frame_ids']}: logit error "
              f"{comparison['max_logit_error']:.6g}, centroid delta "
              f"{comparison['max_centroid_delta_source_px']:.6g}px", flush=True)
    report['windows'] = comparisons
    report['parity_passed'] = all(row['max_logit_error'] < .05 and
                                 row['max_heatmap_probability_error'] < .005 and
                                 row['max_centroid_delta_source_px'] <= .5 and
                                 row['unmatched_components'] == 0 for row in comparisons)
    report['parity_criteria'] = {'max_logit_error_below': .05, 'max_heatmap_probability_error_below': .005,
                                'max_centroid_delta_source_px_at_most': .5, 'unmatched_components': 0}
    report['real_window_count'] = len(comparisons)
    report['offline_gpu_forward_ms'] = {name: {'passes': len(comparisons),
        'median': float(np.median([row[f'{name}_gpu_forward_ms'] for row in comparisons])),
        'p95': float(np.percentile([row[f'{name}_gpu_forward_ms'] for row in comparisons], 95))}
        for name in ['pytorch', 'tensorrt']}
    report['stage'] = 'complete'
    if not report['parity_passed']:
        raise RuntimeError('Numerical/component parity failed the explicit experiment criteria')


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'experiment': 'Isolated BlurBall fixed-shape ONNX/TensorRT FP32 parity', 'promoted': False,
              'repo': str(args.repo.resolve()), 'checkpoint_sha256': sha(args.checkpoint),
              'network_source_sha256': sha(args.repo / 'src/models/blurball.py'),
              'sources': [{'path': str(video.resolve()), 'sha256': sha(video)} for video in args.videos],
              'versions': {'torch': torch.__version__, 'tensorrt': trt.__version__, 'onnx': onnx.__version__},
              'gpu': torch.cuda.get_device_name(0), 'live_capture_to_display_latency_ms': None,
              'scope': 'Offline export feasibility and numerical parity; not native C++ integration, accuracy or live performance.',
              'parity_passed': False}
    started = time.perf_counter()
    try:
        experiment(args, report)
    except Exception as exc:
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['wall_s'] = time.perf_counter() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/blurball_research/export'))
    parser.add_argument('--repo', type=Path, default=Path('artifacts/blurball_research/source'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/blurball_research/blurball_best'))
    parser.add_argument('--videos', type=Path, nargs='+', default=[Path('C:/Users/Adi/Downloads/001.mp4'),
                                                               Path('C:/Users/Adi/Downloads/demo5.mp4')])
    main(parser.parse_args())
