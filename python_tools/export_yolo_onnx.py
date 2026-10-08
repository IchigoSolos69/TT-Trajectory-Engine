"""Export a real single-class ball detector for the fixed C++ TensorRT interface.

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

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'artifacts/python_deps').is_dir():
    sys.path.insert(0, str(ROOT / 'artifacts/python_deps'))


def export_to_onnx(model_path, output_onnx, imgsz=640, fp16=False, kind='ball'):
    import onnx
    from ultralytics import YOLO
    model_path, output_onnx = Path(model_path), Path(output_onnx)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if imgsz != 640:
        raise ValueError('The current C++ deployment requires imgsz=640')
    if output_onnx.exists():
        raise FileExistsError(f'Refusing to overwrite existing export: {output_onnx}')
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    expected = {0: 'ball'} if kind == 'ball' else {0: 'person', 1: 'table'}
    if model.names != expected:
        raise ValueError(f'Expected {expected}; got {model.names}')
    exported = Path(model.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                                 simplify=False, opset=17, device='cpu'))
    graph = onnx.load(str(exported))
    if fp16:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel
        graph = convert_float_to_float16(graph, keep_io_types=True)
        OnnxModel(graph).topological_sort()
    onnx.checker.check_model(graph)
    shapes = [[d.dim_value for d in value.type.tensor_type.shape.dim]
              for value in list(graph.graph.input) + list(graph.graph.output)]
    if shapes != [[1, 3, 640, 640], [1, 4 + len(expected), 8400]]:
        raise ValueError(f'Unexpected deployment shapes: {shapes}')
    for value in list(graph.graph.input) + list(graph.graph.output):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError('C++ deployment requires FP32 input/output tensors')
    onnx.save(graph, str(output_onnx))
    print(f'Validated ONNX: {output_onnx}, shapes={shapes}, internal_fp16={fp16}', flush=True)
    return True


def compile_tensorrt_engine(onnx_path, engine_path, fp16=True, trtexec=None):
    # fp16 must already be encoded in the ONNX graph for TensorRT 11.
    engine_path = Path(engine_path)
    if engine_path.exists():
        raise FileExistsError(f'Refusing to overwrite engine: {engine_path}')
    trtexec = trtexec or shutil.which('trtexec') or 'C:/TensorRT/bin/trtexec.exe'
    subprocess.run([str(trtexec), f'--onnx={onnx_path}', f'--saveEngine={engine_path}',
                    '--memPoolSize=workspace:1024', '--skipInference'], check=True)
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError('TensorRT did not produce a nonempty engine')
    import onnx
    graph = onnx.load(str(onnx_path))
    shape = [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim]
    manifest = {'onnx': str(Path(onnx_path).resolve()), 'engine': str(engine_path.resolve()),
                'fp16_internal': fp16, 'input': [1, 3, 640, 640], 'output': shape,
                'onnx_sha256': hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest(),
                'engine_sha256': hashlib.sha256(engine_path.read_bytes()).hexdigest()}
    engine_path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--trtexec', type=Path)
    parser.add_argument('--kind', choices=['ball', 'scene'], default='ball')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = 'yolov11_ttgames' if args.kind == 'ball' else 'scene'
    onnx_file = args.output_dir / f'{basename}.onnx'
    engine_file = args.output_dir / f'{basename}.engine'
    export_to_onnx(args.model, onnx_file, fp16=args.fp16, kind=args.kind)
    compile_tensorrt_engine(onnx_file, engine_file, args.fp16, args.trtexec)


if __name__ == '__main__':
    main()
