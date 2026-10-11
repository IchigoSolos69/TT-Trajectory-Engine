"""Export inspected static YOLO P3/P2 detectors for the shape-aware C++ interface.

Failures propagate with full tracebacks. No base-model substitution or dummy files.
TensorRT 11 uses ONNX types for precision; FP16 internal tensors retain FP32 I/O.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'artifacts/python_deps').is_dir():
    sys.path.insert(0, str(ROOT / 'artifacts/python_deps'))


def validate_onnx_contract(graph, *, classes=None, imgsz=None, strides=None):
    import onnx
    inputs, outputs = list(graph.graph.input), list(graph.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('Expected exactly one static detector input and output')
    shapes = []
    for value in inputs + outputs:
        dims = value.type.tensor_type.shape.dim
        if any(d.dim_param or d.dim_value <= 0 for d in dims):
            raise ValueError('Dynamic or nonpositive dimensions are unsupported')
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError('C++ deployment requires FP32 input/output tensors')
        shapes.append([d.dim_value for d in dims])
    first, last = shapes
    if len(first) != 4 or first[:2] != [1, 3] or any(v % 32 for v in first[2:]):
        raise ValueError(f'Unsupported RGB NCHW input shape: {first}')
    if len(last) != 3 or last[0] != 1 or not 5 <= last[1] <= 84:
        raise ValueError(f'Unsupported YOLO channel-major box output shape: {last}')
    if classes is not None and last[1] != 4 + classes:
        raise ValueError('Output channel count differs from configured class names')
    if imgsz is not None and first != [1, 3, imgsz, imgsz]:
        raise ValueError('Export input differs from requested resolution')
    if strides is not None:
        if list(strides) not in ([8, 16, 32], [4, 8, 16, 32]):
            raise ValueError(f'Only inspected P3/P2 detection strides supported: {strides}')
        anchors = sum((first[2] // stride) * (first[3] // stride) for stride in strides)
        if last[2] != anchors:
            raise ValueError(f'Expected {anchors} anchors from model strides; got {last[2]}')
    return {'input': first, 'output': last, 'classes': last[1] - 4,
            'anchors': last[2], 'dtype': 'float32', 'strides': list(strides) if strides is not None else None}


def export_to_onnx(model_path, output_onnx, imgsz=640, fp16=False, kind='ball'):
    import onnx
    from ultralytics import YOLO
    model_path, output_onnx = Path(model_path), Path(output_onnx)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if type(imgsz) is not int or imgsz not in (640, 960):
        raise ValueError('Inspected deployment experiments support imgsz=640 or 960')
    if kind not in ('ball', 'scene'):
        raise ValueError('Detector kind must be ball or scene')
    if output_onnx.exists():
        raise FileExistsError(f'Refusing to overwrite existing export: {output_onnx}')
    manifest_path = output_onnx.with_suffix('.export.json')
    if manifest_path.exists():
        raise FileExistsError(f'Refusing existing export manifest: {manifest_path}')
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    original_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    # Ultralytics writes ONNX beside the checkpoint. A private copy prevents
    # exporting from touching original checkpoints or sibling exports/caches.
    with tempfile.TemporaryDirectory(prefix='yolo_export_', dir=output_onnx.parent) as temporary:
        private_model = Path(temporary) / model_path.name
        shutil.copyfile(model_path, private_model)
        if hashlib.sha256(private_model.read_bytes()).hexdigest() != original_sha:
            raise RuntimeError('Private checkpoint copy differs from original checkpoint')
        model = YOLO(str(private_model))
        expected = {0: 'ball'} if kind == 'ball' else {0: 'person', 1: 'table'}
        if model.names != expected:
            raise ValueError(f'Expected {expected}; got {model.names}')
        strides = [int(v) for v in model.model.stride.tolist()]
        exported = Path(model.export(format='onnx', imgsz=imgsz, batch=1, dynamic=False,
                                     simplify=False, opset=17, device='cpu'))
        graph = onnx.load(str(exported))
        if hashlib.sha256(private_model.read_bytes()).hexdigest() != original_sha:
            raise RuntimeError('Private checkpoint changed during export')
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != original_sha:
        raise RuntimeError('Original checkpoint changed during isolated export')
    if fp16:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel
        graph = convert_float_to_float16(graph, keep_io_types=True)
        OnnxModel(graph).topological_sort()
    onnx.checker.check_model(graph)
    contract = validate_onnx_contract(graph, classes=len(expected), imgsz=imgsz, strides=strides)
    onnx.save(graph, str(output_onnx))
    export_manifest = {**contract, 'checkpoint': str(model_path.resolve()), 'checkpoint_sha256': original_sha,
                       'onnx_sha256': hashlib.sha256(output_onnx.read_bytes()).hexdigest(),
                       'internal_fp16': fp16, 'kind': kind, 'promoted': False}
    manifest_path.write_text(json.dumps(export_manifest, indent=2) + '\n')
    print(f'Validated ONNX: {output_onnx}, contract={contract}, internal_fp16={fp16}', flush=True)
    return True


def compile_tensorrt_engine(onnx_path, engine_path, fp16=True, trtexec=None):
    # fp16 must already be encoded in the ONNX graph for TensorRT 11.
    engine_path = Path(engine_path)
    if engine_path.exists():
        raise FileExistsError(f'Refusing to overwrite engine: {engine_path}')
    manifest_path = engine_path.with_suffix('.manifest.json')
    if manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite engine manifest: {manifest_path}')
    import onnx
    graph = onnx.load(str(onnx_path))
    contract = validate_onnx_contract(graph)
    has_fp16 = any(value.data_type == onnx.TensorProto.FLOAT16 for value in graph.graph.initializer)
    if has_fp16 != bool(fp16):
        raise ValueError('Requested internal precision differs from typed ONNX initializers')
    trtexec = trtexec or shutil.which('trtexec') or 'C:/TensorRT/bin/trtexec.exe'
    subprocess.run([str(trtexec), f'--onnx={onnx_path}', f'--saveEngine={engine_path}',
                    '--memPoolSize=workspace:1024', '--skipInference'], check=True)
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError('TensorRT did not produce a nonempty engine')
    manifest = {'onnx': str(Path(onnx_path).resolve()), 'engine': str(engine_path.resolve()),
                'fp16_internal': fp16, **contract, 'promoted': False,
                'onnx_sha256': hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest(),
                'engine_sha256': hashlib.sha256(engine_path.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--trtexec', type=Path)
    parser.add_argument('--kind', choices=['ball', 'scene'], default='ball')
    parser.add_argument('--imgsz', choices=[640, 960], type=int, default=640)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = 'yolov11_ttgames' if args.kind == 'ball' else 'scene'
    onnx_file = args.output_dir / f'{basename}.onnx'
    engine_file = args.output_dir / f'{basename}.engine'
    export_to_onnx(args.model, onnx_file, imgsz=args.imgsz, fp16=args.fp16, kind=args.kind)
    compile_tensorrt_engine(onnx_file, engine_file, args.fp16, args.trtexec)


if __name__ == '__main__':
    main()
