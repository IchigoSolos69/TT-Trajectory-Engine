import json
import os
import cv2
import numpy as np
import onnxruntime as ort

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2] # [1080, 1920]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # 0.3333333
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r))) # [640, 360]
    dw = (new_shape[1] - new_unpad[0]) / 2.0 # 0
    dh = (new_shape[0] - new_unpad[1]) / 2.0 # 140.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def main():
    onnx_path = "models/yolov11_ttgames.onnx"
    video_path = "python_tools/dataset/raw_videos/test_1.mp4"
    ball_json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"

    if not os.path.exists(onnx_path) or not os.path.exists(video_path) or not os.path.exists(ball_json_path):
        print("Missing required model or video files.")
        return

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    with open(ball_json_path, 'r') as f:
        ball_markup = json.load(f)

    annotated_frames = sorted([int(k) for k in ball_markup.keys() if k.isdigit() and ball_markup[k]['x'] >= 0])
    sample_frames = annotated_frames[:50]

    cap = cv2.VideoCapture(video_path)
    center_diffs = []

    print("=========================================================================")
    print("  LETTERBOX UN-PADDING PARITY TEST (First 50 Annotated Frames)")
    print("=========================================================================")
    print(f"{'Frame':<8} | {'GT Center (x,y)':<18} | {'Letterboxed TRT/ONNX (x,y)':<28} | {'Error (px)':<10}")
    print("-------------------------------------------------------------------------")

    for f_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
        ret, frame = cap.read()
        if not ret:
            continue

        gt_x = ball_markup[str(f_idx)]['x']
        gt_y = ball_markup[str(f_idx)]['y']

        img_lb, r, (pad_w, pad_h) = letterbox(frame)
        rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        chw = (rgb.astype(np.float32) / 255.0).transpose((2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        outputs = session.run([output_name], {input_name: input_tensor})[0]
        scores = outputs[0, 4, :]
        max_idx = np.argmax(scores)
        max_score = scores[max_idx]

        if max_score > 0.25:
            cx_raw = outputs[0, 0, max_idx]
            cy_raw = outputs[0, 1, max_idx]

            det_x = (cx_raw - pad_w) / r
            det_y = (cy_raw - pad_h) / r

            dist = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)
            center_diffs.append(dist)

            if len(center_diffs) <= 10:
                print(f"{f_idx:<8d} | ({gt_x:6.1f}, {gt_y:6.1f})     | ({det_x:6.1f}, {det_y:6.1f})                 | {dist:6.2f} px")

    if center_diffs:
        p50 = np.percentile(center_diffs, 50)
        p90 = np.percentile(center_diffs, 90)
        p95 = np.percentile(center_diffs, 95)
        mean_err = np.mean(center_diffs)

        print("-------------------------------------------------------------------------")
        print(f"Localization Metrics across {len(center_diffs)} Detections:")
        print(f"  Median Center Error (p50) : {p50:.2f} px")
        print(f"  Mean Center Error         : {mean_err:.2f} px")
        print(f"  p90 Center Error          : {p90:.2f} px")
        print(f"  p95 Center Error          : {p95:.2f} px")
    print("=========================================================================\n")

if __name__ == "__main__":
    main()
