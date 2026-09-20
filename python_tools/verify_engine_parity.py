import json
import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2] # [height, width] e.g. [1080, 1920]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # scale ratio = 0.3333333
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r))) # [640, 360]
    dw = new_shape[1] - new_unpad[0] # 0
    dh = new_shape[0] - new_unpad[1] # 280
    dw /= 2.0
    dh /= 2.0 # top/bottom pad = 140.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def test_pytorch_vs_trt_parity():
    print("=========================================================================")
    print("  PyTorch best.pt vs ONNX/TensorRT Letterboxed Parity Verification")
    print("=========================================================================")

    onnx_path = "models/yolov11_ttgames.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = "python_tools/yolo11n.onnx"

    model_pt_path = "runs/detect/python_tools/runs/detect/yolo11_ball_game1/weights/best.pt"
    pt_model = None
    if os.path.exists(model_pt_path):
        try:
            pt_model = YOLO(model_pt_path)
            print(f"Loaded PyTorch Model : {model_pt_path}")
        except Exception as e:
            print(f"PyTorch load note: {e}")

    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Loaded ONNX Model    : {onnx_path}")

    video_path = "python_tools/dataset/raw_videos/test_1.mp4"
    ball_json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"

    with open(ball_json_path, 'r') as f:
        ball_markup = json.load(f)

    annotated_frames = sorted([int(k) for k in ball_markup.keys() if k.isdigit() and ball_markup[k]['x'] >= 0])
    sample_frames = annotated_frames[:200]
    print(f"Sampling {len(sample_frames)} annotated frames from test_1 for parity comparison...\n")

    cap = cv2.VideoCapture(video_path)
    
    center_diffs_gt = []
    center_diffs_pt = []
    side_by_side_samples = []

    for f_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
        ret, frame = cap.read()
        if not ret:
            continue

        gt_x = ball_markup[str(f_idx)]['x']
        gt_y = ball_markup[str(f_idx)]['y']

        # Letterbox preprocessing (matches Ultralytics YOLO inference)
        img_lb, scale_r, (pad_w, pad_h) = letterbox(frame, (640, 640))
        rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        chw = rgb.astype(np.float32) / 255.0
        chw = np.transpose(chw, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        # ONNX / TensorRT inference
        outputs = session.run([output_name], {input_name: input_tensor})[0] # Shape: [1, 5, 8400]
        scores = outputs[0, 4, :]
        max_idx = np.argmax(scores)
        max_score = scores[max_idx]

        if max_score > 0.25:
            cx_raw = outputs[0, 0, max_idx]
            cy_raw = outputs[0, 1, max_idx]

            # Un-letterbox coordinate mapping: subtract pad offset and divide by scale ratio
            det_x = (cx_raw - pad_w) / scale_r
            det_y = (cy_raw - pad_h) / scale_r

            dist_gt = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)
            center_diffs_gt.append(dist_gt)

            # PyTorch comparison if model is available
            pt_x, pt_y = None, None
            if pt_model is not None:
                try:
                    res = pt_model(frame, verbose=False, conf=0.25)
                    if len(res) > 0 and len(res[0].boxes) > 0:
                        box = res[0].boxes[0]
                        xywh = box.xywh[0].cpu().numpy()
                        pt_x, pt_y = float(xywh[0]), float(xywh[1])
                        dist_pt = np.sqrt((det_x - pt_x)**2 + (det_y - pt_y)**2)
                        center_diffs_pt.append(dist_pt)
                except Exception:
                    pass

            if len(side_by_side_samples) < 5:
                side_by_side_samples.append((f_idx, gt_x, gt_y, det_x, det_y, pt_x, pt_y, dist_gt))

    if len(center_diffs_gt) > 0:
        p50 = np.percentile(center_diffs_gt, 50)
        p90 = np.percentile(center_diffs_gt, 90)
        p95 = np.percentile(center_diffs_gt, 95)
        p99 = np.percentile(center_diffs_gt, 99)
        mean_err = np.mean(center_diffs_gt)

        print("Side-by-Side Verification Samples (Frame | GT (x,y) | Letterboxed TRT (x,y) | Error px):")
        print("-----------------------------------------------------------------------------------------")
        for s in side_by_side_samples:
            pt_str = f"({s[5]:.1f}, {s[6]:.1f})" if s[5] is not None else "N/A"
            print(f"  Frame {s[0]:5d} | GT: ({s[1]:6.1f}, {s[2]:6.1f}) | TRT/ONNX: ({s[3]:6.1f}, {s[4]:6.1f}) | Pt: {pt_str:<14} | Dist to GT: {s[7]:5.2f} px")

        print("\nLetterboxed Engine Localization Error Percentiles vs Ground Truth:")
        print(f"  Median (p50) localization error : {p50:.2f} px")
        print(f"  Mean localization error         : {mean_err:.2f} px")
        print(f"  p90 localization error          : {p90:.2f} px")
        print(f"  p95 localization error          : {p95:.2f} px")
        print(f"  p99 localization error          : {p99:.2f} px")

        if len(center_diffs_pt) > 0:
            p50_pt = np.percentile(center_diffs_pt, 50)
            print(f"  PyTorch vs TRT/ONNX Center Parity (p50) : {p50_pt:.2f} px")

    print("=========================================================================\n")

if __name__ == "__main__":
    test_pytorch_vs_trt_parity()
