import cv2
import json
import numpy as np
import onnxruntime as ort

def main():
    video_path = "python_tools/dataset/raw_videos/test_1.mp4"
    ball_json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    onnx_path = "models/yolov11_ttgames.onnx"

    with open(ball_json_path, "r") as f:
        ball_markup = json.load(f)

    frame_target = 1672
    gt_data = ball_markup[str(frame_target)]
    gt_x, gt_y = gt_data["x"], gt_data["y"]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_target - 1)
    ret, frame = cap.read()
    cap.release()

    img_h, img_w = frame.shape[:2] # 1080, 1920
    scale_r = min(640.0 / img_w, 640.0 / img_h) # 0.3333333333333333
    unpad_w = int(round(img_w * scale_r)) # 640
    unpad_h = int(round(img_h * scale_r)) # 360
    pad_w = (640 - unpad_w) // 2 # 0
    pad_h = (640 - unpad_h) // 2 # 140

    resized = cv2.resize(frame, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_inputs = {ort_session.get_inputs()[0].name: chw}
    onnx_out = ort_session.run(None, onnx_inputs)[0][0] # [5, 8400]

    cx, cy, w, h, conf = onnx_out[0], onnx_out[1], onnx_out[2], onnx_out[3], onnx_out[4]
    top_indices = np.argsort(conf)[::-1][:5]

    print(f"=========================================================================")
    print(f"  SIDE-BY-SIDE DETECTOR PARITY ON ANNOTATED FRAME {frame_target}")
    print(f"  Ground Truth Location : ({gt_x}, {gt_y})")
    print(f"=========================================================================")
    print(f"{'Rank':<5} | {'Raw 640x640 (cx, cy, w, h)':<30} | {'Conf':<6} | {'Unletterboxed 1920x1080 (X, Y)':<30} | {'Err (px)':<8}")
    print(f"-" * 90)

    for i, idx in enumerate(top_indices):
        raw_cx, raw_cy = cx[idx], cy[idx]
        raw_w, raw_h = w[idx], h[idx]
        c_val = conf[idx]
        det_x = (raw_cx - pad_w) / scale_r
        det_y = (raw_cy - pad_h) / scale_r
        err = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)
        print(f"{i+1:<5} | ({raw_cx:6.1f}, {raw_cy:6.1f}, {raw_w:5.1f}, {raw_h:5.1f})     | {c_val:6.3f} | ({det_x:6.1f}, {det_y:6.1f})                 | {err:6.2f}px")

    print(f"=========================================================================\n")

if __name__ == "__main__":
    main()
