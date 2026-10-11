import cv2
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import json
import numpy as np
import os
import onnxruntime as ort

def main():
    video_path = str(resolve_public_dataset_video("test_1"))
    ball_json_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    onnx_path = "models/yolov11_ttgames.onnx"
    output_dir = "python_tools/gt_overlays"
    os.makedirs(output_dir, exist_ok=True)

    with open(ball_json_path, "r") as f:
        ball_markup = json.load(f)

    # Filter GT frames with valid annotations
    gt_frames = {}
    for f_str, data in ball_markup.items():
        try:
            f_idx = int(f_str)
            if data["x"] >= 0 and data["y"] >= 0:
                gt_frames[f_idx] = (data["x"], data["y"])
        except:
            continue

    print(f"Loaded {len(gt_frames)} valid GT ball annotations.")

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Pick 5 representative GT frames (evenly spaced across dataset with valid detections)
    sample_indices = [218, 556, 1672, 2107, 16599]

    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    
    print("\n--- SEQUENTIAL VIDEO READ & OVERLAY VERIFICATION ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        if current_frame in sample_indices:
            gt_x, gt_y = gt_frames[current_frame]

            # Manual Letterbox & ONNX Inference
            img_h, img_w = frame.shape[:2] # 1080, 1920
            scale_r = min(640.0 / img_w, 640.0 / img_h) # 0.3333333333333333
            unpad_w = int(round(img_w * scale_r)) # 640
            unpad_h = int(round(img_h * scale_r)) # 360
            pad_w = (640 - unpad_w) // 2 # 0
            pad_h = (640 - unpad_h) // 2 # 140

            resized = cv2.resize(frame, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
            padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # BGR -> RGB & Normalize 0..1
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...] # [1, 3, 640, 640]

            onnx_inputs = {ort_session.get_inputs()[0].name: chw}
            onnx_out = ort_session.run(None, onnx_inputs)[0] # [1, 5, 8400]
            
            # Postprocess ONNX output
            out = onnx_out[0] # [5, 8400]
            cx, cy, w, h, conf = out[0], out[1], out[2], out[3], out[4]
            best_idx = np.argmax(conf)
            best_conf = conf[best_idx]
            
            onnx_det_x, onnx_det_y = -1, -1
            if best_conf > 0.15:
                raw_cx, raw_cy = cx[best_idx], cy[best_idx]
                onnx_det_x = (raw_cx - pad_w) / scale_r
                onnx_det_y = (raw_cy - pad_h) / scale_r

            onnx_dist = np.sqrt((onnx_det_x - gt_x)**2 + (onnx_det_y - gt_y)**2) if onnx_det_x >= 0 else -1

            print(f"Frame {current_frame:5d} | GT: ({gt_x:4d}, {gt_y:4d}) | Model Det: ({onnx_det_x:6.1f}, {onnx_det_y:6.1f}) Conf: {best_conf:.3f} | Dist Err: {onnx_dist:5.2f}px")

            # Draw visual overlay on frame image
            vis_img = frame.copy()
            # GT Annotation: Solid RED circle + crosshair
            cv2.circle(vis_img, (int(gt_x), int(gt_y)), 8, (0, 0, 255), 2)
            cv2.circle(vis_img, (int(gt_x), int(gt_y)), 2, (0, 0, 255), -1)
            cv2.putText(vis_img, f"GT ({gt_x},{gt_y})", (int(gt_x) + 12, int(gt_y) - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Model Detection: GREEN circle + center dot
            if onnx_det_x >= 0:
                cv2.circle(vis_img, (int(onnx_det_x), int(onnx_det_y)), 8, (0, 255, 0), 2)
                cv2.circle(vis_img, (int(onnx_det_x), int(onnx_det_y)), 2, (0, 255, 0), -1)
                cv2.putText(vis_img, f"Det ({onnx_det_x:.1f},{onnx_det_y:.1f}) err:{onnx_dist:.1f}px", 
                            (int(onnx_det_x) + 12, int(onnx_det_y) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out_file = os.path.join(output_dir, f"overlay_frame_{current_frame}.png")
            cv2.imwrite(out_file, vis_img)

    cap.release()
    print(f"\nOverlay PNG files written to {output_dir}/")

if __name__ == "__main__":
    main()
