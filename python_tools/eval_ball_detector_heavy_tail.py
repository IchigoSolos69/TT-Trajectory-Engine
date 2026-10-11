import json
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import os
import glob
import math
import numpy as np
import cv2
import torch
import torchvision

# Monkey-patch torchvision.ops.nms if C++ ops are missing
def custom_nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    boxes_wh = boxes_np.copy()
    boxes_wh[:, 2] = boxes_np[:, 2] - boxes_np[:, 0]
    boxes_wh[:, 3] = boxes_np[:, 3] - boxes_np[:, 1]
    
    indices = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores_np.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    indices = np.array(indices).flatten()
    return torch.from_numpy(indices).to(device=boxes.device, dtype=torch.long)

if not hasattr(torchvision, 'ops'):
    class Ops: pass
    ops = Ops()
    ops.nms = custom_nms
    torchvision.ops = ops

from ultralytics import YOLO

def run_heavy_tail_evaluation(model_path, test_seq="test_1", imgsz=640, conf_thresh=0.15):
    print(f"=== Evaluating {model_path} on {test_seq} (imgsz={imgsz}, conf={conf_thresh}) ===")
    
    markup_path = f"python_tools/dataset/raw_markup/{test_seq}/ball_markup.json"
    if not os.path.exists(markup_path):
        print(f"Error: {markup_path} not found.")
        return None
        
    with open(markup_path) as f:
        gt_data = json.load(f)
        
    gt_frames = {}
    for k, v in gt_data.items():
        if isinstance(v, dict) and "x" in v and "y" in v and v["x"] is not None:
            if float(v["x"]) >= 0 and float(v["y"]) >= 0:
                gt_frames[int(k)] = (float(v["x"]), float(v["y"]))
                
    video_path = str(resolve_public_dataset_video(test_seq))
        
    if not os.path.exists(video_path):
        print(f"Error: video {video_path} not found.")
        return None
        
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    total_frames = 0
    
    errors_annotated = []
    annotated_detection_count = 0
    total_detections_all_frames = 0
    fp_unannotated_frames = 0
    
    hit_5_count = 0
    hit_10_count = 0
    hit_15_count = 0
    
    detected_gt_ball_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        total_frames = frame_idx
        
        results = model.predict(frame, imgsz=imgsz, conf=conf_thresh, device='cpu', verbose=False)[0]
        boxes = results.boxes
        
        ball_detections = []
        for b in boxes:
            cls_id = int(b.cls[0].item())
            if cls_id == 0:
                xywh = b.xywh[0].tolist()
                conf = float(b.conf[0].item())
                ball_detections.append((xywh[0], xywh[1], conf))
                
        num_dets = len(ball_detections)
        total_detections_all_frames += num_dets
        
        if frame_idx in gt_frames:
            gt_x, gt_y = gt_frames[frame_idx]
            if num_dets > 0:
                detected_gt_ball_frames += 1
                min_dist = float('inf')
                for (pred_x, pred_y, conf) in ball_detections:
                    dist = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                
                errors_annotated.append(min_dist)
                annotated_detection_count += 1
                
                if min_dist <= 5.0:
                    hit_5_count += 1
                if min_dist <= 10.0:
                    hit_10_count += 1
                if min_dist <= 15.0:
                    hit_15_count += 1
        else:
            if num_dets > 0:
                fp_unannotated_frames += 1
                
    cap.release()
    
    errors_arr = np.array(errors_annotated)
    n_gt = len(gt_frames)
    recall_raw = (detected_gt_ball_frames / n_gt * 100.0) if n_gt > 0 else 0.0
    
    med_err = float(np.median(errors_arr)) if len(errors_arr) > 0 else 0.0
    mean_err = float(np.mean(errors_arr)) if len(errors_arr) > 0 else 0.0
    std_err = float(np.std(errors_arr)) if len(errors_arr) > 0 else 0.0
    p90_err = float(np.percentile(errors_arr, 90)) if len(errors_arr) > 0 else 0.0
    p95_err = float(np.percentile(errors_arr, 95)) if len(errors_arr) > 0 else 0.0
    p99_err = float(np.percentile(errors_arr, 99)) if len(errors_arr) > 0 else 0.0
    
    out_of_15_count = int(np.sum(errors_arr > 15.0))
    out_of_15_pct = (out_of_15_count / len(errors_arr) * 100.0) if len(errors_arr) > 0 else 0.0
    
    hit_5_rate = (hit_5_count / len(errors_arr) * 100.0) if len(errors_arr) > 0 else 0.0
    hit_10_rate = (hit_10_count / len(errors_arr) * 100.0) if len(errors_arr) > 0 else 0.0
    hit_15_rate = (hit_15_count / len(errors_arr) * 100.0) if len(errors_arr) > 0 else 0.0
    
    full_frame_prec_15 = (hit_15_count / total_detections_all_frames * 100.0) if total_detections_all_frames > 0 else 0.0
    
    metrics = {
        "sequence": test_seq,
        "total_video_frames": total_frames,
        "annotated_gt_frames": n_gt,
        "detected_gt_frames": detected_gt_ball_frames,
        "raw_recall_pct": float(recall_raw),
        "total_detections_all_frames": total_detections_all_frames,
        "annotated_detections_count": len(errors_arr),
        "fp_unannotated_frames": fp_unannotated_frames,
        "median_error_px": med_err,
        "mean_error_px": mean_err,
        "std_error_px": std_err,
        "p90_error_px": p90_err,
        "p95_error_px": p95_err,
        "p99_error_px": p99_err,
        "out_of_15px_count": out_of_15_count,
        "out_of_15px_pct": out_of_15_pct,
        "hit_rate_5px_pct": hit_5_rate,
        "hit_rate_10px_pct": hit_10_rate,
        "hit_rate_15px_pct": hit_15_rate,
        "full_frame_precision_15px_pct": full_frame_prec_15
    }
    
    print("\n--- METRICS BREAKDOWN ---")
    print(f"Total Video Frames: {total_frames}")
    print(f"GT Ball Frames: {n_gt} | Detected GT Frames: {detected_gt_ball_frames} ({recall_raw:.2f}% Raw Recall)")
    print(f"Total Detections across Video: {total_detections_all_frames}")
    print(f"Detections on Annotated Frames: {len(errors_arr)}")
    print(f"Localization Error Distribution (px):")
    print(f"  Median: {med_err:.2f} px")
    print(f"  Mean:   {mean_err:.2f} ± {std_err:.2f} px")
    print(f"  p90:    {p90_err:.2f} px")
    print(f"  p95:    {p95_err:.2f} px")
    print(f"  p99:    {p99_err:.2f} px")
    print(f"  >15px:  {out_of_15_count} detections ({out_of_15_pct:.2f}%)")
    print(f"Hit Rates (among detections on annotated frames):")
    print(f"  <= 5px:  {hit_5_rate:.2f}% ({hit_5_count}/{len(errors_arr)})")
    print(f"  <= 10px: {hit_10_rate:.2f}% ({hit_10_count}/{len(errors_arr)})")
    print(f"  <= 15px: {hit_15_rate:.2f}% ({hit_15_count}/{len(errors_arr)})")
    print(f"Full-Frame Precision (<= 15px / all video detections): {full_frame_prec_15:.2f}% ({hit_15_count}/{total_detections_all_frames})")
    
    with open("python_tools/detector_heavy_tail_results.json", "w") as out_f:
        json.dump(metrics, out_f, indent=2)
        
    return metrics

if __name__ == "__main__":
    best_weights = "runs/detect/python_tools/runs/detect/yolo11_ball_game1/weights/best.pt"
    if not os.path.exists(best_weights):
        best_weights = "models/best.pt"
    run_heavy_tail_evaluation(best_weights, test_seq="test_1")
