"""Connected Ultralytics 8.4.155 sampler, augmentation and validation hooks.

Imported only for explicit fitting or CPU hook verification. No automatic fit,
downloads, export or deployment occur on import.
"""
from __future__ import annotations
from copy import copy, deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import ops

from yolo_training_policy import (BalancedDomainSampler, DOMAINS, MEMORY_BUDGET_BYTES,
                                  camera_augment, checkpoint_fitness, example_seed)


class EpochDataLoader(DataLoader):
    # BaseTrainer calls reset when closing mosaic. Ordinary DataLoader creates a
    # fresh iterator each epoch; this also avoids stale-prefetched sampler epochs.
    def reset(self):
        return None


class CameraSafeTransform:
    def __init__(self, seed, epoch=0, extent_targets=True, source_identities=None):
        self.seed, self.epoch, self.extent_targets = seed, epoch, extent_targets
        self.source_identities=source_identities

    def __call__(self, labels):
        labels = dict(labels)
        instances = deepcopy(labels["instances"])
        instances.convert_bbox("xyxy")
        height, width = labels["img"].shape[:2]
        instances.denormalize(width, height)
        private_path=Path(labels["im_file"]).resolve().as_posix()
        identity=(self.source_identities[private_path] if self.source_identities is not None
                  else Path(private_path).name)
        image, boxes, _ = camera_augment(labels["img"], instances.bboxes,
            seed=example_seed(self.seed,self.epoch,identity), extent_targets=self.extent_targets)
        instances.update(bboxes=boxes)
        labels["instances"], labels["img"] = instances, image
        return labels


class DomainBallValidator(DetectionValidator):
    """Keep normal IoU diagnostics; source-space ball metrics select checkpoints."""
    def init_metrics(self, model):
        super().init_metrics(model)
        self.ball_rows = []

    def update_metrics(self, preds, batch):
        for index, pred in enumerate(preds):
            raw = self._prepare_batch(index,batch)
            identity = Path(raw["im_file"]).resolve().as_posix()
            if identity not in self.reference_records:
                raise ValueError("Validation image has no reviewed training-only reference")
            record = self.reference_records[identity]
            if record["role"] != "val":
                raise ValueError("Only B/validation references may select checkpoints")
            eligible = (pred["cls"] == 0) & (pred["conf"] >= self.ball_score_threshold)
            prediction = None
            if bool(eligible.any()):
                scores = pred["conf"][eligible]
                best_index = int(scores.argmax())
                box = pred["bboxes"][eligible][best_index:best_index+1].clone()
                box = ops.scale_boxes(raw["imgsz"],box,raw["ori_shape"],raw["ratio_pad"])[0]
                prediction = dict(center_px=[float((box[0]+box[2])/2),float((box[1]+box[3])/2)],
                                  state="measured",score=float(scores[best_index]))
            self.ball_rows.append({key:deepcopy(record.get(key)) for key in
                ("source_id","frame_id","timestamp_s","source_size","domain","rally_id","annotation")} | {"prediction":prediction})
        super().update_metrics(preds,batch)

    def get_stats(self):
        from ball_release_metrics import score_frames
        result = super().get_stats()
        self.ball_report = score_frames(self.ball_rows,tolerance1080_px=4)
        return result


def make_trainer(private, variant, seed, expected_batch):
    """Return the actual custom Trainer class passed to YOLO.train()."""
    references = {Path(r["private_image"]).resolve().as_posix():r for r in private["records"]}
    source_identities={path:f"{r['source_id']}:{r['frame_id']}" for path,r in references.items()}
    domain_lists = private["validation_domain_lists"]

    class PairedCameraTrainer(DetectionTrainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.add_callback("on_train_epoch_start",self._fixed_epoch_start)
            self.add_callback("on_train_batch_end",self._memory_check)

        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path,batch_size,rank,mode)
            if rank not in (-1,0) or self.world_size > 1:
                raise ValueError("Audited paired sampler currently supports one process/GPU only")
            if batch_size != expected_batch or self.args.batch != expected_batch:
                raise ValueError("Batch changed; restart the entire comparison group with common 16/8/4")
            dataset = self.build_dataset(dataset_path,mode,batch_size)
            ordered = [references[Path(name).resolve().as_posix()] for name in dataset.im_files]
            sampler = BalancedDomainSampler(ordered,seed)
            if variant["augmentation"] == "camera_safe":
                dataset.transforms = Compose([LetterBox(new_shape=(self.args.imgsz,self.args.imgsz)),
                    CameraSafeTransform(seed,extent_targets=variant["geometry"]=="reviewed_extent",source_identities=source_identities),
                    Format(bbox_format="xywh",normalize=True,batch_idx=True,bgr=0)])
            loader = EpochDataLoader(dataset,batch_size=batch_size,sampler=sampler,
                num_workers=self.args.workers,collate_fn=dataset.collate_fn,
                generator=torch.Generator().manual_seed(seed),pin_memory=self.device.type=="cuda")
            return loader

        def _fixed_epoch_start(self,_):
            self._oom_retries = 3  # inspected BaseTrainer refuses automatic halving after this limit
            if self.batch_size != expected_batch:
                raise ValueError("Ultralytics altered the frozen comparison batch")
            sampler = self.train_loader.sampler
            if not isinstance(sampler,BalancedDomainSampler):
                raise ValueError("Actual training loader is missing the balanced sampler")
            sampler.set_epoch(self.epoch)
            for transform in getattr(self.train_loader.dataset.transforms,"transforms",[]):
                if isinstance(transform,CameraSafeTransform): transform.epoch=self.epoch
            if self.device.type=="cuda": torch.cuda.reset_peak_memory_stats(self.device)
            self._memory_check(None)

        def _memory_check(self,_):
            if self.device.type=="cuda":
                allocated=torch.cuda.max_memory_allocated(self.device)
                reserved=torch.cuda.max_memory_reserved(self.device)
                if max(allocated,reserved)>MEMORY_BUDGET_BYTES:
                    raise RuntimeError("Frozen comparison exceeded 7 GiB; rerun all compared variants at common batch 8 or 4")

        def validate(self):
            reports,flat={},{}
            for domain in DOMAINS:
                loader=self.get_dataloader(domain_lists[domain],self.args.batch,-1,"val")
                args=copy(self.args)
                args.plots=False
                validator=DomainBallValidator(loader,save_dir=self.save_dir/f"validation_{domain}",args=args,_callbacks=self.callbacks)
                validator.reference_records=references
                validator.ball_score_threshold=.25
                metrics=validator(self)
                reports[domain]=validator.ball_report
                for key in ("precision","recall","f1"):
                    flat[f"ball/{domain}_{key}"]=float(reports[domain][key]) if reports[domain][key] is not None else float("nan")
            fitness=checkpoint_fitness(reports)
            if self.best_fitness is None or fitness > self.best_fitness: self.best_fitness=fitness
            with (self.save_dir/"domain_ball_validation.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(dict(epoch=self.epoch,fitness=fitness,fitness_policy="MINIMUM_DOMAIN_F1",
                    score_threshold=.25,tolerance1080_px=4,domains=reports,test_labels_consumed=False),allow_nan=False)+"\n")
            return flat,fitness

        def final_eval(self):
            # BaseTrainer's final_eval uses pooled IoU validation and may replace
            # metrics with a different objective. Preserve our per-domain audit;
            # best.pt was already saved using this trainer's minimax fitness.
            self.metrics["ball/selected_checkpoint_fitness"] = self.best_fitness

    return PairedCameraTrainer
