"""Synthetic target/alignment tests; no real labels, GPU execution or training."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from blurball_reviewed_dataset import (ReviewedBlurBallDataset, audit_export, line_target,
                                      loss_config, source_to_output_affine, transform_target)
from export_reviewed_streaks import export, target
from test_reviewed_streaks import StreakTests, label

ROOT = Path(__file__).resolve().parents[1]


def source_label(endpoints, dimensions=(1920, 1080)):
    annotation = label()
    annotation['geometry']['blur_streak'].update(endpoints=endpoints, source_width=dimensions[0], source_height=dimensions[1])
    return target(annotation, dimensions)


class AdapterTests(unittest.TestCase):
    def fixture(self, root):
        captures, splits = StreakTests().fixture(root)
        handoff = root / 'export'
        export(captures, splits, handoff)
        return captures, splits, handoff

    def test_affine_matches_authoritative_upstream_all_aspects(self):
        spec = importlib.util.spec_from_file_location('inspected_upstream_image', ROOT / 'artifacts/blurball_research/source/src/utils/image.py')
        upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
        for dimensions in [(1920,1080),(640,480),(1080,1920),(193,109)]:
            w,h=dimensions
            expected=upstream.get_affine_transform(np.array([w/2,h/2],np.float32), max(w,h), 0, [512,288])
            np.testing.assert_array_equal(source_to_output_affine(dimensions),expected)
        np.testing.assert_allclose(source_to_output_affine((640,480)),[[.8,0,0],[0,.8,-48]],atol=1e-12)

    def test_endpoint_units_scale_once_and_inverse_alignment(self):
        geometry=source_label([[.25,.25],[.75,.75]])
        before=copy.deepcopy(geometry)
        transformed=transform_target(geometry,(1920,1080))
        np.testing.assert_allclose(transformed['endpoints_px'],[[128,72],[384,216]],atol=1e-10)
        np.testing.assert_allclose(transformed['center_px'],[256,144])
        self.assertAlmostEqual(transformed['half_length_px'],math.hypot(256,144)/2)
        self.assertAlmostEqual(transformed['axis_deg'],math.degrees(math.atan2(144,256)))
        matrix=source_to_output_affine((1920,1080)); inverse=cv2.invertAffineTransform(matrix)
        reconstructed=np.c_[transformed['endpoints_px'],np.ones(2)] @ inverse.T
        np.testing.assert_allclose(reconstructed,np.array(geometry['endpoints_normalized'])*[1920,1080],atol=1e-10)
        self.assertEqual(before,geometry,'Target transformation must not mutate source extent/provenance')

    def test_capsule_reversal_and_fractional_positive_never_blank(self):
        for endpoints in [[[.3,.4],[.5,.4]],[[.4,.3],[.4,.5]],[[.2,.3],[.3,.4]],[[.3,.4],[.3001,.4001]]]:
            a=transform_target(source_label(endpoints),(1920,1080))
            b=transform_target(source_label(endpoints[::-1]),(1920,1080))
            np.testing.assert_array_equal(line_target(a),line_target(b))
            self.assertAlmostEqual(a['half_length_px'],b['half_length_px'])
            self.assertAlmostEqual(a['axis_deg'],b['axis_deg'])
        tiny=transform_target(source_label([[.4,.4],[.4001,.4]]),(1920,1080))
        self.assertLess(tiny['half_length_px'],1)
        pixels=line_target(tiny)
        self.assertGreater(pixels.sum(),0)
        self.assertEqual(pixels.dtype,np.float32)
        self.assertEqual(set(np.unique(pixels)),{0,1})

    def test_sharp_upstream_parity_and_intentional_fractional_fix(self):
        spec=importlib.util.spec_from_file_location('inspected_upstream_heatmap',ROOT/'artifacts/blurball_research/source/src/utils/heatmap.py')
        upstream=importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
        sharp=transform_target(source_label([[.4,.4],[.4,.4]]),(1920,1080))
        self.assertEqual(sharp['half_length_px'],0); self.assertIsNone(sharp['axis_deg'])
        reference=upstream.gen_line_binary_map((512,288),sharp['center_px'],0,0,2.5)
        np.testing.assert_array_equal(line_target(sharp),reference)
        tiny=transform_target(source_label([[.4,.4],[.4001,.4]]),(1920,1080))
        upstream_tiny=upstream.gen_line_binary_map((512,288),tiny['center_px'],tiny['axis_deg'],tiny['half_length_px'],2.5)
        self.assertEqual(upstream_tiny.sum(),0,'Observed upstream regression must remain documented')
        self.assertGreater(line_target(tiny).sum(),0)
        self.assertAlmostEqual(tiny['half_length_px'],.0256,places=10,msg='Fix rasterization, not the reviewed label extent')

    def test_crop_and_edge_handling_no_silent_absence(self):
        # A 4:3 frame is cropped vertically by the actual model affine.
        with self.assertRaisesRegex(ValueError,'outside retained output'):
            transform_target(source_label([[.5,0],[.5,0]],(640,480)),(640,480))
        with self.assertRaisesRegex(ValueError,'outside retained output'):
            transform_target(source_label([[1,.5],[1,.5]]),(1920,1080))
        clipped=transform_target(source_label([[0,.5],[.02,.5]]),(1920,1080))
        self.assertGreater(line_target(clipped).sum(),0)
        self.assertEqual(clipped['endpoints_px'][0][0],0)

    def test_absence_blank_unknown_and_partial_excluded(self):
        absent=transform_target({'visibility':'absent','position_px':None,'half_length_px':None,'axis_rad':None},(1920,1080))
        self.assertEqual(line_target(absent).sum(),0)
        for visibility in ['unknown','occluded','partial']:
            with self.assertRaisesRegex(ValueError,'reviewed fully visible'):
                transform_target({'visibility':visibility},(1920,1080))

    def test_dataset_rgb_tensor_target_channel_alignment_and_torch(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            captures,splits,handoff=self.fixture(Path(tmp))
            dataset=ReviewedBlurBallDataset(handoff,captures,splits,'train',loss_choice='quality_focal')
            x,ys,metadata=dataset.sample(0)
            self.assertEqual(x.shape,(9,288,512)); self.assertEqual(ys[0].shape,(3,288,512))
            self.assertEqual(x.dtype,np.float32); self.assertEqual(ys[0].dtype,np.float32)
            import torch
            for i in range(3):
                image=Image.open(captures/'train'/f'frame_{i+1}.jpg').convert('RGB')
                resized=cv2.warpAffine(np.array(image),source_to_output_affine(image.size),(512,288),flags=cv2.INTER_LINEAR)
                # Equivalent inspected ToTensor/Normalize operations; this env
                # lacks the complete torchvision.transforms distribution.
                tensor=torch.from_numpy(resized.transpose(2,0,1).copy()).to(torch.float32).div(255)
                expected=((tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]).numpy()
                np.testing.assert_array_equal(x[3*i:3*i+3],expected)
                expected_geometry=transform_target(source_label([[.2,.3],[.3,.4]],(192,108)),(192,108))
                np.testing.assert_array_equal(ys[0][i],line_target(expected_geometry))
                self.assertEqual(metadata['frames'][i]['source_frame'],i+1)
            self.assertFalse(np.array_equal(x[0:3],x[3:6]))
            torch_dataset=ReviewedBlurBallDataset(handoff,captures,splits,'train',loss_choice='mse',as_torch=True)
            tx,ty=torch_dataset[0]
            batch=next(iter(torch.utils.data.DataLoader(torch_dataset,batch_size=1,num_workers=0)))
            self.assertEqual(tuple(batch[0].shape),(1,9,288,512))
            self.assertEqual(tuple(batch[1][0].shape),(1,3,288,512))
            self.assertEqual(tx.dtype,torch.float32); self.assertEqual(ty[0].dtype,torch.float32)

    def test_loss_must_be_explicit_and_never_trains(self):
        self.assertEqual(loss_config('quality_focal')['beta'],2)
        self.assertEqual(loss_config('mse')['sub_name'],'mse')
        for choice in [None,'auto','']:
            with self.assertRaisesRegex(ValueError,'Choose loss explicitly'): loss_config(choice)

    def test_after_audit_changes_fail_at_dataset_load(self):
        for change in ['image','annotation','index','split','export']:
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
                captures,splits,handoff=self.fixture(Path(tmp))
                dataset=ReviewedBlurBallDataset(handoff,captures,splits,'train',loss_choice='quality_focal')
                path={'image':captures/'train/frame_1.jpg','annotation':captures/'train/annotations.json','index':captures/'train/frames.jsonl','split':splits,'export':handoff/'train.jsonl'}[change]
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'changed after audit'): dataset[0]

    def test_partial_windows_excluded_without_relabeling(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            captures,splits,handoff=self.fixture(Path(tmp))
            annotations=captures/'train/annotations.json'
            data=json.loads(annotations.read_text()); data['frame_2.jpg']['geometry']['visibility']='partial'; annotations.write_text(json.dumps(data))
            second=Path(tmp)/'partial_export'; export(captures,splits,second)
            with self.assertRaisesRegex(ValueError,'No fully reviewed'):
                audit_export(second,captures,splits,loss_choice='quality_focal')

    def test_missing_split_or_unknown_supervision_refuses_handoff(self):
        for invalid in ['empty_test', 'unknown_train']:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
                captures,splits,handoff=self.fixture(Path(tmp))
                if invalid=='empty_test':
                    data=json.loads(splits.read_text()); data['test']=[]; splits.write_text(json.dumps(data))
                else:
                    annotations=captures/'train/annotations.json'
                    data=json.loads(annotations.read_text())
                    for value in data.values():
                        value['reviewed_classes']['ball']=False
                        value['geometry']={'visibility':'unknown','blur_streak':None}
                    annotations.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'Independent nonempty|No complete reviewed'):
                    audit_export(handoff,captures,splits,loss_choice='quality_focal')


if __name__=='__main__': unittest.main()
