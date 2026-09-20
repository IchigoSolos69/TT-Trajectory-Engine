import os
import glob
import numpy as np

def compute_detection_metrics(yolo_dir, gt_dir, iou_thresh=0.5):
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
    if not gt_files:
        print(f"Error: No GT label files found in {gt_dir}")
        return

    total_gt_frames = len(gt_files)
    total_gt_balls = 0
    total_det_balls = 0

    tp_balls = 0
    fp_balls = 0
    missed_frames = 0

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        yolo_path = os.path.join(yolo_dir, filename)

        # Read GT
        gt_boxes = []
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == '0':
                    gt_boxes.append([float(x) for x in parts[1:5]])

        total_gt_balls += len(gt_boxes)

        # Read Detections
        det_boxes = []
        if os.path.exists(yolo_path):
            with open(yolo_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == '0':
                        det_boxes.append([float(x) for x in parts[1:5]])

        total_det_balls += len(det_boxes)

        if len(gt_boxes) > 0 and len(det_boxes) == 0:
            missed_frames += 1

        # Calculate IoU for simple 1-to-1 match
        matched = False
        for g in gt_boxes:
            for d in det_boxes:
                # BBox format: cx, cy, w, h
                gx1, gy1 = g[0] - g[2]/2, g[1] - g[3]/2
                gx2, gy2 = g[0] + g[2]/2, g[1] + g[3]/2
                dx1, dy1 = d[0] - d[2]/2, d[1] - d[3]/2
                dx2, dy2 = d[0] + d[2]/2, d[1] + d[3]/2

                ix1 = max(gx1, dx1)
                iy1 = max(gy1, dy1)
                ix2 = min(gx2, dx2)
                iy2 = min(gy2, dy2)

                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih

                area_g = g[2] * g[3]
                area_d = d[2] * d[3]
                union = area_g + area_d - inter

                iou = inter / union if union > 0 else 0
                if iou >= iou_thresh:
                    matched = True
                    break

        if matched:
            tp_balls += 1
        elif len(det_boxes) > 0:
            fp_balls += 1

    fn_balls = total_gt_balls - tp_balls
    prec = (tp_balls / total_det_balls) if total_det_balls > 0 else 0.0
    rec = (tp_balls / total_gt_balls) if total_gt_balls > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    missed_pct = (missed_frames / total_gt_frames * 100.0) if total_gt_frames > 0 else 0.0

    print("=========================================================================")
    print("  test_1 YOLO Ball Detection Level Performance Metrics")
    print("=========================================================================")
    print(f"Total Ground-Truth Ball Frames : {total_gt_frames}")
    print(f"Total Detections Evaluated     : {total_det_balls}")
    print(f"True Positive Ball Detections  : {tp_balls}")
    print(f"False Positive Ball Detections : {fp_balls}")
    print(f"False Negative (Missed Balls)  : {fn_balls}")
    print(f"Ball Detection Precision       : {prec*100:.2f}% ({prec:.4f})")
    print(f"Ball Detection Recall          : {rec*100:.2f}% ({rec:.4f})")
    print(f"Ball Detection F1 Score        : {f1*100:.2f}% ({f1:.4f})")
    print(f"Frames with Missed Ball (FN %) : {missed_pct:.2f}% ({missed_frames}/{total_gt_frames})")
    print("=========================================================================\n")

if __name__ == "__main__":
    compute_detection_metrics(
        yolo_dir="python_tools/dataset/labels/test_1",
        gt_dir="python_tools/dataset/labels/test_1"
    )
