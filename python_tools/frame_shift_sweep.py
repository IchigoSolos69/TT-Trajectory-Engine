import json
import cv2
import numpy as np
import onnxruntime as ort
import os

def run_frame_shift_sweep():
    model_path = "models/yolov11_ttgames.onnx"
    video_path = "python_tools/dataset/raw_videos/test_1.mp4"
    json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"

    if not os.path.exists(model_path) or not os.path.exists(video_path) or not os.path.exists(json_path):
        print("Error: missing files")
        return

    # Load GT Ball Markup
    with open(json_path, 'r') as f:
        markup_raw = json.load(f)
    
    gt_map = {}
    for k, v in markup_raw.items():
        if isinstance(v, dict) and "x" in v and "y" in v:
            x, y = float(v["x"]), float(v["y"])
            if x >= 0 and y >= 0:
                gt_map[int(k)] = (x, y)

    print(f"Loaded GT Ball Frames: {len(gt_map)}")

    # Init ONNX Session
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    imgW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    imgH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # We will process frames sequentially (1-based index)
    # Store predictions for all frames: frame_idx -> (det_x, det_y, conf)
    detections_map = {}

    frame_idx = 0
    CONF_THRESH = 0.15

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        # Check if frame_idx or frame_idx + offset is in gt_map (check range [frame_idx-5, frame_idx+5])
        needs_inference = False
        for off in range(-5, 6):
            if (frame_idx + off) in gt_map:
                needs_inference = True
                break
        
        if not needs_inference:
            continue

        # Aspect-Ratio Letterbox (matching C++ letterbox: 1920x1080 -> 640x640 with padH=140)
        scaleR = min(640.0 / imgW, 640.0 / imgH) # 1/3 = 0.3333333
        unpadW = int(round(imgW * scaleR)) # 640
        unpadH = int(round(imgH * scaleR)) # 360
        padW = (640 - unpadW) // 2 # 0
        padH = (640 - unpadH) // 2 # 140

        resized = cv2.resize(frame, (unpadW, unpadH))
        padded = cv2.copyMakeBorder(resized, padH, padH, padW, padW, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        outputs = session.run(None, {input_name: input_tensor})
        out = outputs[0][0].T # (8400, 5)

        best_det = None
        best_conf = 0.0

        for i in range(out.shape[0]):
            conf = out[i, 4]
            if conf > CONF_THRESH and conf > best_conf:
                cx, cy, w, h = out[i, :4]
                # Un-letterbox back to 1920x1080
                cx_orig = (cx - padW) / scaleR
                cy_orig = (cy - padH) / scaleR
                best_conf = float(conf)
                best_det = (cx_orig, cy_orig, best_conf)

        if best_det is not None:
            detections_map[frame_idx] = best_det

    cap.release()
    print(f"Inference complete across annotated frame regions. Total detections saved: {len(detections_map)}")

    # Evaluate offsets -2, -1, 0, +1, +2
    # offset definition: frame_idx matches gt_map[frame_idx + offset]
    print("\n=========================================================================")
    print("  FRAME-SHIFT SWEEP (Hit Rate <= 15.0 px on Annotated Frames)")
    print("=========================================================================")

    for offset in [-3, -2, -1, 0, 1, 2, 3]:
        hits_5 = 0
        hits_10 = 0
        hits_15 = 0
        total_eval = 0
        errors = []

        for f_idx, (gt_x, gt_y) in gt_map.items():
            # The candidate frame in video is (f_idx + offset)
            target_frame = f_idx + offset
            total_eval += 1

            if target_frame in detections_map:
                det_x, det_y, conf = detections_map[target_frame]
                dist = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)
                errors.append(dist)
                if dist <= 5.0: hits_5 += 1
                if dist <= 10.0: hits_10 += 1
                if dist <= 15.0: hits_15 += 1

        hit_rate_15 = (hits_15 / total_eval) * 100.0 if total_eval > 0 else 0.0
        hit_rate_10 = (hits_10 / total_eval) * 100.0 if total_eval > 0 else 0.0
        hit_rate_5  = (hits_5  / total_eval) * 100.0 if total_eval > 0 else 0.0
        med_err = np.median(errors) if len(errors) > 0 else 0.0

        print(f"  Offset {offset:+2d} | Hit <=5px: {hit_rate_5:5.2f}% | Hit <=10px: {hit_rate_10:5.2f}% | Hit <=15px: {hit_rate_15:5.2f}% | Median Err: {med_err:6.2f} px")

if __name__ == "__main__":
    run_frame_shift_sweep()
