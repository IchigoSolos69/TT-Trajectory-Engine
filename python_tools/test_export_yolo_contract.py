"""CPU contract/isolated-export guards, separate from real TensorRT parity."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import onnx
from onnx import helper, TensorProto
from export_yolo_onnx import validate_onnx_contract, export_to_onnx, compile_tensorrt_engine


def graph(size=640, anchors=8400, classes=1, *, dtype=TensorProto.FLOAT):
    first=helper.make_tensor_value_info('images',dtype,[1,3,size,size])
    last=helper.make_tensor_value_info('output',dtype,[1,classes+4,anchors])
    return helper.make_model(helper.make_graph([], 'contract_fixture', [first], [last]))


class Contracts(unittest.TestCase):
    def test_standard_640(self):
        self.assertEqual(validate_onnx_contract(graph(),classes=1,imgsz=640,strides=[8,16,32])['anchors'],8400)

    def test_standard_960(self):
        self.assertEqual(validate_onnx_contract(graph(960,18900),classes=1,imgsz=960,strides=[8,16,32])['input'],[1,3,960,960])

    def test_p2_both_sizes(self):
        for size,count in [(640,34000),(960,76500)]:
            self.assertEqual(validate_onnx_contract(graph(size,count),strides=[4,8,16,32])['anchors'],count)

    def test_scene_two_classes(self):
        self.assertEqual(validate_onnx_contract(graph(classes=2),classes=2)['classes'],2)

    def test_wrong_anchor_count(self):
        with self.assertRaises(ValueError):validate_onnx_contract(graph(960,8400),strides=[8,16,32])

    def test_wrong_classes(self):
        with self.assertRaises(ValueError):validate_onnx_contract(graph(),classes=2)

    def test_dynamic_dimensions(self):
        value=graph();value.graph.input[0].type.tensor_type.shape.dim[2].dim_param='height'
        with self.assertRaises(ValueError):validate_onnx_contract(value)

    def test_fp16_io_rejected(self):
        with self.assertRaises(ValueError):validate_onnx_contract(graph(dtype=TensorProto.FLOAT16))

    def test_extra_output_rejected(self):
        value=graph();value.graph.output.append(helper.make_tensor_value_info('other',TensorProto.FLOAT,[1]))
        with self.assertRaises(ValueError):validate_onnx_contract(value)

    def test_channel_last_rejected(self):
        value=graph();value.graph.output[0].type.tensor_type.shape.dim[1].dim_value=8400
        value.graph.output[0].type.tensor_type.shape.dim[2].dim_value=5
        with self.assertRaises(ValueError):validate_onnx_contract(value)

    def test_isolated_export_uses_requested_size_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'input.pt';checkpoint.write_bytes(b'original checkpoint');sibling=root/'input.onnx';sibling.write_bytes(b'original sibling')
            observed={}
            class FakeYOLO:
                def __init__(self,path):
                    self.path=Path(path);self.names={0:'ball'};self.model=types.SimpleNamespace(stride=types.SimpleNamespace(tolist=lambda:[8,16,32]))
                    observed['private_checkpoint']=str(self.path)
                def export(self,**kwargs):
                    observed.update(kwargs);count=sum((kwargs['imgsz']//stride)**2 for stride in [8,16,32]);target=self.path.with_suffix('.onnx')
                    onnx.save(graph(kwargs['imgsz'],count),target);return str(target)
            with patch.dict('sys.modules',{'ultralytics':types.SimpleNamespace(YOLO=FakeYOLO)}),patch('onnx.checker.check_model'):
                target=root/'out'/'candidate.onnx';self.assertTrue(export_to_onnx(checkpoint,target,imgsz=960))
            self.assertNotEqual(observed['private_checkpoint'],str(checkpoint));self.assertEqual(observed['imgsz'],960)
            self.assertEqual(checkpoint.read_bytes(),b'original checkpoint');self.assertEqual(sibling.read_bytes(),b'original sibling')
            manifest=json.loads(target.with_suffix('.export.json').read_text());self.assertEqual(manifest['output'],[1,5,18900]);self.assertFalse(manifest['promoted'])
            self.assertEqual(manifest['checkpoint_sha256'],hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_existing_manifest_refused_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'input.pt';checkpoint.write_bytes(b'x');target=root/'candidate.onnx';target.with_suffix('.export.json').write_text('{}')
            with patch.dict('sys.modules',{'ultralytics':types.SimpleNamespace(YOLO=lambda _:self.fail('Must refuse before model loading'))}):
                with self.assertRaises(FileExistsError):export_to_onnx(checkpoint,target)
            self.assertFalse(target.exists())

    def test_engine_precision_checked_before_subprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'model.onnx';onnx.save(graph(),source)
            with patch('export_yolo_onnx.subprocess.run',side_effect=AssertionError('Must not run engine build')):
                with self.assertRaises(ValueError):compile_tensorrt_engine(source,root/'model.engine',fp16=True)
            self.assertFalse((root/'model.engine').exists())


if __name__=='__main__':unittest.main(verbosity=2)
