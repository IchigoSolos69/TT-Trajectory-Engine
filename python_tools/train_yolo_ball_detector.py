import sys
import types
import os
import json
import math
import cv2
import numpy as np
import torch

def custom_nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        i = order[0].item()
        keep.append(i)
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

ops_mod = types.ModuleType("torchvision.ops")
ops_mod.nms = custom_nms

tv_mod = types.ModuleType("torchvision")
tv_mod.ops = ops_mod

sys.modules["torchvision"] = tv_mod
sys.modules["torchvision.ops"] = ops_mod

from ultralytics import YOLO

def find_best_weights():
    candidate_paths = [
        "runs/detect/python_tools/runs/detect/yolo11_ball_game1/weights/best.pt",
        "python_tools/runs/detect/yolo11_ball_game1/weights/best.pt",
        "runs/detect/yolo11_ball_game1/weights/best.pt",
        "runs/detect/python_tools/runs/detect/yolo11_ball_game1/weights/last.pt"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None

def eval_on_test1(weights_path=None, imgsz=640):
    if weights_path is None:
        weights_path = find_best_weights()

    if weights_path is None or not os.path.exists(weights_path):
        print(f"Error: Trained weights file not found.")
        return

    print("\n=========================================================================")
    print(f"  EVALUATING FINE-TUNED DETECTOR ON HELD-OUT TEST_1 ({weights_path})")
    print("=========================================================================")

    model = YOLO(weights_path)
    video_path = "python_tools/dataset/raw_videos/test_1.mp4"
    json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"

    if not os.path.exists(video_path) or not os.path.exists(json_path):
        print("Error: test_1 video or ball markup missing.")
        return

    with open(json_path, 'r') as f:
        gt_markup = json.load(f)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    gt_ball_frames = 0
    gt_ball_free_frames = 0

    detected_gt_ball_count = 0
    fp_ball_free_count = 0

    distances = []
    recalls_5px = 0
    recalls_10px = 0
    recalls_15px = 0

    precisions_5px = 0
    precisions_10px = 0
    precisions_15px = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_id += 1

        frame_str = str(frame_id)
        gt_pos = gt_markup.get(frame_str, None)

        is_gt_ball = (gt_pos is not None and isinstance(gt_pos, dict) and "x" in gt_pos and "y" in gt_pos and gt_pos["x"] is not None)

        results = model.predict(frame, imgsz=imgsz, conf=0.15, verbose=False)[0]
        ball_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0: # Ball class
                xywh = box.xywh[0].tolist()
                conf = float(box.conf[0].item())
                ball_detections.append((xywh[0], xywh[1], conf))

        if is_gt_ball:
            gt_ball_frames += 1
            gt_x = float(gt_pos["x"])
            gt_y = float(gt_pos["y"])

            min_dist = float('inf')
            best_pred = None

            for (pred_x, pred_y, conf) in ball_detections:
                dist = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_pred = (pred_x, pred_y)

            if best_pred is not None:
                detected_gt_ball_count += 1
                distances.append(min_dist)

                if min_dist <= 5.0:
                    recalls_5px += 1
                if min_dist <= 10.0:
                    recalls_10px += 1
                if min_dist <= 15.0:
                    recalls_15px += 1
        else:
            gt_ball_free_frames += 1
            if len(ball_detections) > 0:
                fp_ball_free_count += 1

        if frame_id % 2000 == 0:
            print(f"  Processed {frame_id} / {total_frames} frames...")

    cap.release()

    if gt_ball_frames == 0:
        print("No ground truth ball frames found in test_1.")
        return

    overall_recall = (detected_gt_ball_count / gt_ball_frames) * 100.0
    rec5 = (recalls_5px / gt_ball_frames) * 100.0
    rec10 = (recalls_10px / gt_ball_frames) * 100.0
    rec15 = (recalls_15px / gt_ball_frames) * 100.0

    prec5 = (recalls_5px / detected_gt_ball_count * 100.0) if detected_gt_ball_count > 0 else 0.0
    prec10 = (recalls_10px / detected_gt_ball_count * 100.0) if detected_gt_ball_count > 0 else 0.0
    prec15 = (recalls_15px / detected_gt_ball_count * 100.0) if detected_gt_ball_count > 0 else 0.0

    mean_dist = np.mean(distances) if distances else float('nan')
    std_dist = np.std(distances) if distances else float('nan')
    median_dist = np.median(distances) if distances else float('nan')
    fpr_uncapped = (fp_ball_free_count / gt_ball_free_frames * 100.0) if gt_ball_free_frames > 0 else 0.0

    print(f"\n=========================================================================")
    print(f"  HELD-OUT TEST_1 DETECTOR EVALUATION RESULTS")
    print(f"=========================================================================")
    print(f"  Total Video Frames Processed : {total_frames}")
    print(f"  GT Ball Frames               : {gt_ball_frames}")
    print(f"  GT Ball-Free Frames          : {gt_ball_free_frames}")
    print(f"  Detections Found on Ball GT  : {detected_gt_ball_count} ({overall_recall:.2f}% Raw Recall)")
    print(f"-------------------------------------------------------------------------")
    print(f"  Mean Localization Error      : {mean_dist:.2f} ± {std_dist:.2f} px")
    print(f"  Median Localization Error    : {median_dist:.2f} px")
    print(f"-------------------------------------------------------------------------")
    print(f"  Recall @ <= 5 px             : {recalls_5px} / {gt_ball_frames} ({rec5:.2f}%) | Precision: {prec5:.2f}%")
    print(f"  Recall @ <= 10 px            : {recalls_10px} / {gt_ball_frames} ({rec10:.2f}%) | Precision: {prec10:.2f}%")
    print(f"  Recall @ <= 15 px            : {recalls_15px} / {gt_ball_frames} ({rec15:.2f}%) | Precision: {prec15:.2f}%")
    print(f"-------------------------------------------------------------------------")
    print(f"  Uncapped False Positive Rate : {fp_ball_free_count} / {gt_ball_free_frames} ({fpr_uncapped:.2f}%) on ball-free frames")
    print(f"=========================================================================\n")

if __name__ == "__main__":
    eval_on_test1()
