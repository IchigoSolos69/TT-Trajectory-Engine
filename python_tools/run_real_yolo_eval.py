import os
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import cv2
import json
import numpy as np
import onnxruntime as ort
import glob

def run_onnx_inference_on_test1():
    model_path = "models/yolov11_ttgames.onnx"
    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at {model_path}")
        return

    print(f"Loading ONNX Model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f"Model Input Name: {input_name} | Shape: {session.get_inputs()[0].shape}")

    json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    video_path = str(resolve_public_dataset_video("test_1"))
    if not os.path.exists(json_path) or not os.path.exists(video_path):
        print("Error: Missing test_1 video or markup file.")
        return

    with open(json_path, 'r') as f:
        markup = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    pred_dir = "python_tools/dataset/labels/test_1_pred"
    os.makedirs(pred_dir, exist_ok=True)

    # Thresholds
    CONF_THRESH = 0.25

    annotated_frames = 0
    total_detections = 0

    predictions_map = {}

    for frame_str, coords in markup.items():
        try:
            f_idx = int(frame_str)
            gt_x = float(coords['x'])
            gt_y = float(coords['y'])
        except (ValueError, KeyError, TypeError):
            continue

        if gt_x < 0 or gt_y < 0:
            continue

        annotated_frames += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Preprocess frame to 640x640 RGB float32 [1, 3, 640, 640]
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1)) # CHW
        input_tensor = np.expand_dims(input_tensor, axis=0) # NCHW

        outputs = session.run(None, {input_name: input_tensor})
        # YOLOv8/v11 output shape: [1, 5, 8400] -> (cx, cy, w, h, conf)
        out = outputs[0][0] # [5, 8400]
        if out.shape[0] == 5:
            # (5, 8400) -> transpose to (8400, 5)
            out = out.T

        boxes = []
        confidences = []
        for i in range(out.shape[0]):
            conf = out[i, 4]
            if conf >= CONF_THRESH:
                cx, cy, w, h = out[i, :4]
                # Scale back to original frame (1920x1080)
                cx_orig = (cx / 640.0) * width
                cy_orig = (cy / 640.0) * height
                w_orig = (w / 640.0) * width
                h_orig = (h / 640.0) * height
                boxes.append([cx_orig, cy_orig, w_orig, h_orig])
                confidences.append(float(conf))

        # NMS
        if len(boxes) > 0:
            best_idx = np.argmax(confidences)
            best_box = boxes[best_idx]
            predictions_map[f_idx] = (best_box[0], best_box[1], confidences[best_idx])
            total_detections += 1

            # Write pred TXT label
            norm_cx = best_box[0] / width
            norm_cy = best_box[1] / height
            norm_w = best_box[2] / width
            norm_h = best_box[3] / height
            with open(os.path.join(pred_dir, f"test_1_frame_{f_idx}.txt"), 'w') as pf:
                pf.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    cap.release()
    print(f"Processed {annotated_frames} annotated frames | Real YOLO Detections Found: {total_detections}")

    # Evaluate Detection Metrics against GT
    gt_dir = "python_tools/dataset/labels/test_1"
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

    tp = 0
    fp = 0
    fn = 0

    dist_threshold_pixels = 30.0 # 30 pixels distance threshold in 1080p

    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        f_str = basename.split("_frame_")[-1].replace(".txt", "")
        f_idx = int(f_str)

        # Read GT
        with open(gt_file, 'r') as gf:
            line = gf.readline().strip()
            parts = line.split()
            gt_cx = float(parts[1]) * width
            gt_cy = float(parts[2]) * height

        if f_idx in predictions_map:
            pred_cx, pred_cy, conf = predictions_map[f_idx]
            dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
            if dist <= dist_threshold_pixels:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n=========================================================================")
    print("  REAL YOLO Model Detection Performance on Held-Out test_1 (880 Frames)")
    print("=========================================================================")
    print(f"Annotated GT Frames Evaluated : {len(gt_files)}")
    print(f"Real YOLO Detections Matched : {tp}")
    print(f"False Positives (Off-Target)  : {fp}")
    print(f"False Negatives (Missed Ball) : {fn}")
    print(f"Distance Match Threshold      : {dist_threshold_pixels:.1f} pixels (30x30px ball area)")
    print(f"YOLO Ball Precision           : {precision * 100.0:.2f}% ({precision:.4f})")
    print(f"YOLO Ball Recall              : {recall * 100.0:.2f}% ({recall:.4f})")
    print(f"YOLO Ball F1 Score            : {f1 * 100.0:.2f}% ({f1:.4f})")
    print("=========================================================================\n")

if __name__ == "__main__":
    run_onnx_inference_on_test1()
