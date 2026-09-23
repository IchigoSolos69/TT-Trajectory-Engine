import json
import cv2
import numpy as np
import onnxruntime as ort
import os

def test_preprocessing_methods():
    model_path = "models/yolov11_ttgames.onnx"
    video_path = "python_tools/dataset/raw_videos/test_1.mp4"
    json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"

    with open(json_path, 'r') as f:
        gt_raw = json.load(f)

    gt_map = {int(k): (float(v["x"]), float(v["y"])) for k, v in gt_raw.items() if float(v.get("x", -1)) >= 0 and float(v.get("y", -1)) >= 0}

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    imgW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    imgH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # We will test 2 preprocessing methods:
    # Method A: Direct Resize (640x640 squashed)
    # Method B: Aspect-Ratio Letterbox (640x640 with padH=140)

    errs_A = []
    errs_B = []

    hits_15_A = 0
    hits_15_B = 0
    total = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        if frame_idx not in gt_map:
            continue

        gt_x, gt_y = gt_map[frame_idx]
        total += 1

        # --- METHOD A: Direct Resize ---
        resized_A = cv2.resize(frame, (640, 640))
        rgb_A = cv2.cvtColor(resized_A, cv2.COLOR_BGR2RGB)
        tensor_A = (rgb_A.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out_A = session.run(None, {input_name: tensor_A})[0][0].T

        best_err_A = 9999.0
        best_conf_A = 0.0
        for i in range(out_A.shape[0]):
            conf = out_A[i, 4]
            if conf > 0.15:
                cx, cy, w, h = out_A[i, :4]
                pxX = (cx / 640.0) * imgW
                pxY = (cy / 640.0) * imgH
                err = np.sqrt((pxX - gt_x)**2 + (pxY - gt_y)**2)
                if conf > best_conf_A:
                    best_conf_A = conf
                    best_err_A = err
        
        if best_conf_A > 0:
            errs_A.append(best_err_A)
            if best_err_A <= 15.0: hits_15_A += 1

        # --- METHOD B: Aspect-Ratio Letterbox ---
        scaleR = min(640.0 / imgW, 640.0 / imgH) # 1/3
        unpadW = int(round(imgW * scaleR)) # 640
        unpadH = int(round(imgH * scaleR)) # 360
        padW = (640 - unpadW) // 2 # 0
        padH = (640 - unpadH) // 2 # 140

        resized_B = cv2.resize(frame, (unpadW, unpadH))
        padded_B = cv2.copyMakeBorder(resized_B, padH, padH, padW, padW, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb_B = cv2.cvtColor(padded_B, cv2.COLOR_BGR2RGB)
        tensor_B = (rgb_B.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out_B = session.run(None, {input_name: tensor_B})[0][0].T

        best_err_B = 9999.0
        best_conf_B = 0.0
        for i in range(out_B.shape[0]):
            conf = out_B[i, 4]
            if conf > 0.15:
                cx, cy, w, h = out_B[i, :4]
                pxX = (cx - padW) / scaleR
                pxY = (cy - padH) / scaleR
                err = np.sqrt((pxX - gt_x)**2 + (pxY - gt_y)**2)
                if conf > best_conf_B:
                    best_conf_B = conf
                    best_err_B = err
        
        if best_conf_B > 0:
            errs_B.append(best_err_B)
            if best_err_B <= 15.0: hits_15_B += 1

    cap.release()

    med_A = np.median(errs_A) if errs_A else 0
    med_B = np.median(errs_B) if errs_B else 0

    print("=========================================================================")
    print("  PREPROCESSING METHOD COMPARISON ON test_1 (880 GT Ball Frames)")
    print("=========================================================================")
    print(f"  Method A (Direct Resize 640x640) : Median Err = {med_A:5.2f} px | Hit <=15px = {hits_15_A/total*100:5.2f}% ({hits_15_A}/{total})")
    print(f"  Method B (Letterbox padH=140)   : Median Err = {med_B:5.2f} px | Hit <=15px = {hits_15_B/total*100:5.2f}% ({hits_15_B}/{total})")
    print("=========================================================================")

if __name__ == "__main__":
    test_preprocessing_methods()
