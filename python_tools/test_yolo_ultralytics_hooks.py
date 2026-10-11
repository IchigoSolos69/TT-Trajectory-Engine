"""CPU verification against installed Ultralytics, without optimization."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer,DetectionValidator
from ultralytics.utils.instance import Instances

from yolo_ultralytics_hooks import CameraSafeTransform,DomainBallValidator,make_trainer
from yolo_training_policy import BalancedDomainSampler


class UltralyticsHookTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def private_fixture(self):
        records=[]
        images=self.root/"images"/"train"; labels=self.root/"labels"/"train"
        images.mkdir(parents=True); labels.mkdir(parents=True)
        combinations=[("match",True),("match",False),("iphone",True),("iphone",False),("pc_camera",True),("pc_camera",False)]
        for index,(domain,positive) in enumerate(combinations):
            image=images/f"frame_{index}.jpg"
            Image.new("RGB",(64,48),(index*30,40,60)).save(image)
            (labels/f"frame_{index}.txt").write_text("0 .5 .5 .1 .1" if positive else "")
            records.append(dict(private_image=image.as_posix(),domain=domain,positive=positive,role="train",
                                source_id=f"source:{index}",frame_id=1))
        listing=self.root/"train.txt"; listing.write_text("\n".join(r["private_image"] for r in records)+"\n")
        return dict(records=records,validation_domain_lists={}),listing

    def test_installed_trainer_builds_real_dataset_with_sampler_and_camera_transform(self):
        private,listing=self.private_fixture()
        cls=make_trainer(private,dict(augmentation="camera_safe",geometry="reviewed_extent"),42,4)
        self.assertTrue(issubclass(cls,DetectionTrainer))
        trainer=cls.__new__(cls)
        trainer.args=get_cfg(overrides=dict(imgsz=64,batch=4,workers=0,mosaic=0.,mixup=0.,copy_paste=0.,plots=False))
        trainer.model=SimpleNamespace(stride=torch.tensor([8,16,32]))
        trainer.data=dict(names={0:"ball"},nc=1,channels=3)
        trainer.device=torch.device("cpu"); trainer.world_size=0
        loader=trainer.get_dataloader(listing.as_posix(),4,-1,"train")
        self.assertIsInstance(loader.sampler,BalancedDomainSampler)
        self.assertTrue(any(isinstance(t,CameraSafeTransform) for t in loader.dataset.transforms.transforms))
        batch=next(iter(loader))
        self.assertEqual(tuple(batch["img"].shape),(4,3,64,64))
        trainer.train_loader=loader; trainer.batch_size=4; trainer.epoch=2
        trainer._fixed_epoch_start(None)
        self.assertEqual(loader.sampler.epoch,2)
        self.assertEqual(trainer._oom_retries,3)
        self.assertEqual(next(t for t in loader.dataset.transforms.transforms if isinstance(t,CameraSafeTransform)).epoch,2)
        with self.assertRaisesRegex(ValueError,"Batch changed"): trainer.get_dataloader(listing.as_posix(),8,-1,"train")

    def test_actual_validator_scales_prediction_to_source_frame_and_uses_metric_api(self):
        image=(self.root/"val_frame.jpg").resolve()
        reference=dict(source_id="reviewed-source",frame_id=9,timestamp_s=.133,
            source_size=[64,48],domain="iphone",role="val",rally_id="rally-1",
            annotation=dict(reviewed=True,visibility="visible",center_px=[20,20],tags=[]))
        validator=DomainBallValidator(save_dir=self.root/"validation",args=dict(plots=False))
        validator.device=torch.device("cpu")
        validator.reference_records={image.as_posix():reference}; validator.ball_score_threshold=.25
        validator.ball_rows=[]
        batch=dict(batch_idx=torch.tensor([0]),cls=torch.tensor([[0.]]),bboxes=torch.tensor([[.3,.3,.1,.1]]),
            ori_shape=[(48,64)],img=torch.zeros((1,3,128,128)),ratio_pad=[((2.,2.),(0.,16.))],im_file=[str(image)])
        pred=dict(cls=torch.tensor([0.]),conf=torch.tensor([.9]),bboxes=torch.tensor([[32.,48.,48.,64.]]))
        with patch.object(DetectionValidator,"update_metrics",return_value=None): validator.update_metrics([pred],batch)
        self.assertEqual(validator.ball_rows[0]["prediction"]["center_px"],[20.,20.])
        with patch.object(DetectionValidator,"get_stats",return_value={}): validator.get_stats()
        self.assertEqual(validator.ball_report["tp"],1)
        self.assertEqual(validator.ball_report["f1"],1)
        validator.reference_records[image.as_posix()]["role"]="test"
        with self.assertRaisesRegex(ValueError,"Only B"): validator.update_metrics([pred],batch)

    def test_memory_guard_is_connected_and_has_no_silent_batch_retry(self):
        cls=make_trainer(dict(records=[],validation_domain_lists={}),dict(augmentation="common_preset",geometry="center_square30"),42,8)
        trainer=cls.__new__(cls); trainer.device=torch.device("cuda:0")
        with patch("torch.cuda.max_memory_allocated",return_value=8*1024**3),patch("torch.cuda.max_memory_reserved",return_value=8*1024**3):
            with self.assertRaisesRegex(RuntimeError,"7 GiB"): trainer._memory_check(None)

    def test_augmentation_seed_uses_source_identity_not_private_output_directory(self):
        a=(self.root/"candidateA"/"frame_1.jpg").resolve().as_posix()
        b=(self.root/"candidateB"/"frame_1.jpg").resolve().as_posix()
        pixels=np.arange(64*48*3,dtype=np.uint8).reshape(48,64,3)
        def labels(path):
            return dict(im_file=path,img=pixels.copy(),instances=Instances(
                np.array([[.5,.5,.1,.1]],dtype=np.float32),bbox_format="xywh",normalized=True))
        transform=CameraSafeTransform(42,source_identities={a:"original:7",b:"original:7"})
        first,second=transform(labels(a)),transform(labels(b))
        np.testing.assert_array_equal(first["img"],second["img"])
        np.testing.assert_array_equal(first["instances"].bboxes,second["instances"].bboxes)


if __name__=="__main__": unittest.main()
