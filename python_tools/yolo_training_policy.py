"""Auditable sampling, camera augmentation and experiment policies.

This module has no Ultralytics or torch dependency. It never assigns sessions,
reviews predictions, or changes source annotations.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import random

DOMAINS = ("match", "iphone", "pc_camera")
BATCH_CHOICES = (16, 8, 4)
FINAL_SEEDS = (42, 43, 44)
MEMORY_BUDGET_BYTES = 7 * 1024**3
CAMERA_AUGMENTATION = {
    "gamma": [0.8, 1.25], "brightness_multiplier": [0.85, 1.15],
    "contrast": [0.8, 1.2], "jpeg_quality": [50, 95],
    "motion_kernel_sizes": [3, 5, 7], "motion_probability": 0.5,
    "kernel_centered": True, "mosaic": 0, "mixup": 0, "copy_paste": 0,
    "geometry_policy": "EXTENT_EXPANDS_WITH_SYMMETRIC_KERNEL_CENTER_TARGET_UNCHANGED",
}


class BalancedDomainSampler:
    """Exact epoch quotas, with replacement only inside reviewed groups.

    Every epoch is a multiple of 24: half match, one quarter each camera;
    each camera contributes two visible/partial observations per absence.
    Repeated samples remain the original reviewed examples, never new labels.
    """
    def __init__(self, records, seed=42, epoch_samples=None):
        self.records = records
        self.seed, self.epoch = int(seed), 0
        self.groups = {("match", None): []}
        self.groups.update({(domain, positive): [] for domain in DOMAINS[1:] for positive in (True, False)})
        for index, row in enumerate(records):
            if row.get("domain") not in DOMAINS or type(row.get("positive")) is not bool:
                raise ValueError("Sampler requires explicit domain and reviewed positive/absence status")
            if row.get("role") != "train":
                raise ValueError("Only train observations may enter the sampler")
            key = ("match", None) if row["domain"] == "match" else (row["domain"], row["positive"])
            self.groups[key].append(index)
        if any(not group for group in self.groups.values()):
            raise ValueError("Balanced sampling requires match and visible/absent examples from both cameras")
        requested = len(records) if epoch_samples is None else int(epoch_samples)
        if requested <= 0:
            raise ValueError("Epoch sample count must be positive")
        self.epoch_samples = math.ceil(requested / 24) * 24

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.epoch_samples

    def __iter__(self):
        rng = random.Random(self.seed + 1000003 * self.epoch)
        camera_count = self.epoch_samples // 4
        quotas = {("match", None): self.epoch_samples // 2}
        quotas.update({(domain, True): camera_count * 2 // 3 for domain in DOMAINS[1:]})
        quotas.update({(domain, False): camera_count // 3 for domain in DOMAINS[1:]})
        result = []
        for key, count in quotas.items():
            pool = self.groups[key]
            # Cycle shuffled pools rather than sample a single popular image.
            for offset in range(0, count, len(pool)):
                shuffled = list(pool)
                rng.shuffle(shuffled)
                result.extend(shuffled[:min(len(pool), count - offset)])
        rng.shuffle(result)
        return iter(result)

    def summary(self):
        counts = Counter((self.records[i]["domain"], self.records[i]["positive"]) for i in self)
        return {domain: {"positive": counts[(domain, True)], "absent": counts[(domain, False)]} for domain in DOMAINS}


def camera_augment(image, boxes_xyxy, *, seed, extent_targets=True, forced=None):
    """Photometric camera effects and symmetric blur, without moving centers.

    Inputs are BGR uint8 pixels and absolute xyxy boxes in those pixels. Blur
    is skipped when a target's expanded extent would touch the image boundary;
    clipping a streak would otherwise silently shift its supervised center.
    """
    import cv2
    import numpy as np
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Camera augmentation requires BGR8 pixels")
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    height, width = image.shape[:2]
    if not np.isfinite(boxes).all() or (len(boxes) and (
        np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2] > width) or np.any(boxes[:, 3] > height)
        or np.any(boxes[:, 2:] <= boxes[:, :2]))):
        raise ValueError("Invalid pixel box extent")
    rng = random.Random(int(seed))
    params = dict(gamma=rng.uniform(.8, 1.25), brightness=rng.uniform(.85, 1.15),
                  contrast=rng.uniform(.8, 1.2), jpeg_quality=rng.randint(50, 95),
                  kernel_size=rng.choice((3, 5, 7)) if rng.random() < .5 else 1,
                  angle=rng.choice((0, 45, 90, 135)))
    if forced:
        if set(forced) - set(params):
            raise ValueError("Unknown augmentation parameter")
        params.update(forced)
    if not (.8 <= params["gamma"] <= 1.25 and .85 <= params["brightness"] <= 1.15
            and .8 <= params["contrast"] <= 1.2 and 50 <= params["jpeg_quality"] <= 95
            and params["kernel_size"] in (1, 3, 5, 7) and params["angle"] in (0, 45, 90, 135)):
        raise ValueError("Augmentation outside the camera-safe policy")
    values = (image.astype(np.float32) / 255) ** params["gamma"]
    values = ((values - .5) * params["contrast"] + .5) * params["brightness"]
    result = np.clip(np.rint(values * 255), 0, 255).astype(np.uint8)
    size, angle = params["kernel_size"], params["angle"]
    radius = (size - 1) // 2
    dx = radius if angle != 90 else 0
    dy = radius if angle != 0 else 0
    expanded = boxes + np.array([-dx, -dy, dx, dy], dtype=np.float32)
    boundary = len(boxes) and (np.any(expanded[:, :2] < 0) or np.any(expanded[:, 2] > width) or np.any(expanded[:, 3] > height))
    params["motion_skipped_boundary"] = bool(boundary)
    if size > 1 and not boundary:
        kernel = np.zeros((size, size), dtype=np.float32)
        if angle == 0:
            kernel[radius, :] = 1
        elif angle == 90:
            kernel[:, radius] = 1
        elif angle == 45:
            np.fill_diagonal(kernel, 1)
        else:
            np.fill_diagonal(np.fliplr(kernel), 1)
        result = cv2.filter2D(result, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT_101)
        if extent_targets:
            boxes = expanded
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, int(params["jpeg_quality"])])
    if not ok:
        raise RuntimeError("Camera JPEG augmentation failed")
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if result is None or result.shape != image.shape:
        raise RuntimeError("Camera JPEG augmentation changed geometry")
    return result, boxes, params


def checkpoint_fitness(domain_metrics):
    """Minimax per-domain F1: the largest validation domain cannot swamp others."""
    if set(domain_metrics) != set(DOMAINS):
        raise ValueError("Checkpoint selection requires all three validation domains")
    for metric in domain_metrics.values():
        if metric.get("recall") is None or metric.get("f1") is None:
            raise ValueError("Undefined validation metric cannot select a checkpoint")
    return min(float(metric["f1"]) for metric in domain_metrics.values())


def center_square_labels(labels, image_size, source_extent_px=30, centers_px=None):
    """Versioned historical center-square target, clipped only at source bounds."""
    width, height = image_size
    result = {"ball": []}
    for index,(x, y, w, h) in enumerate(labels["ball"]):
        cx, cy = (x+w/2)*width, (y+h/2)*height
        if centers_px is not None:
            cx,cy=centers_px[index]
        left, right = max(0, cx-source_extent_px/2), min(width, cx+source_extent_px/2)
        top, bottom = max(0, cy-source_extent_px/2), min(height, cy+source_extent_px/2)
        result["ball"].append([left/width, top/height, (right-left)/width, (bottom-top)/height])
    return result


def example_seed(seed, epoch, image_identity):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{image_identity}".encode()).digest()[:8], "little")
