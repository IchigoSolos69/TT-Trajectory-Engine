import json
import cv2
import numpy as np
import onnxruntime as ort
import os

def overlay_fp_bounces():
    model_path = "models/yolov11_ttgames.onnx"
    video_path = "python_tools/dataset/raw_videos/test_1.mp4"
    ball_json = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    events_json = "python_tools/dataset/raw_markup/test_1/events_markup.json"
    out_dir = "python_tools/fp_samples"
    os.makedirs(out_dir, exist_ok=True)

    with open(ball_json, 'r') as f:
        ball_raw = json.load(f)
    gt_ball_frames = {int(k): (float(v["x"]), float(v["y"])) for k, v in ball_raw.items() if float(v.get("x", -1)) >= 0 and float(v.get("y", -1)) >= 0}

    with open(events_json, 'r') as f:
        events_raw = json.load(f)
    gt_bounces = sorted([int(k) for k, v in events_raw.items() if v == "bounce" or (isinstance(v, dict) and v.get("type") == "bounce")])

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    imgW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    imgH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dt = 1.0 / 120.0
    CONF_THRESH = 0.25

    # Run inference and physics bounce detection
    detections_by_frame = {}
    frame_idx = 0
    vy_history = []
    last_pixel_y = -1.0
    last_pixel_vy = 0.0
    bounce_cooldown = 0
    bounce_armed = True

    detected_bounces = []
    bounce_details = []

    frames_cache = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        # Save frame in cache if near candidates
        scaleR = min(640.0 / imgW, 640.0 / imgH) # 1/3
        unpadW = int(round(imgW * scaleR))
        unpadH = int(round(imgH * scaleR))
        padW = (640 - unpadW) // 2
        padH = (640 - unpadH) // 2

        resized = cv2.resize(frame, (unpadW, unpadH))
        padded = cv2.copyMakeBorder(resized, padH, padH, padW, padW, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        out = session.run(None, {input_name: tensor})[0][0].T

        best_det = None
        best_conf = 0.0
        for i in range(out.shape[0]):
            conf = out[i, 4]
            if conf > CONF_THRESH and conf > best_conf:
                cx, cy, w, h = out[i, :4]
                pxX = (cx - padW) / scaleR
                pxY = (cy - padH) / scaleR
                best_conf = float(conf)
                best_det = (pxX, pxY, float(w), float(h), float(conf))

        detections_by_frame[frame_idx] = best_det
        det_y = best_det[1] if best_det else -1.0
        det_x = best_det[0] if best_det else -1.0

        if bounce_cooldown > 0:
            bounce_cooldown -= 1

        if det_y >= 0 and last_pixel_y >= 0:
            pixel_vy = (det_y - last_pixel_y) / dt
            pixel_accel = (last_pixel_vy - pixel_vy) / dt
            vy_history.append(pixel_vy)
            if len(vy_history) > 4:
                vy_history.pop(0)

            max_inbound = max(vy_history[:-1]) if len(vy_history) > 1 else -9999.0
            delta_vy = max_inbound - pixel_vy

            cond_sign_flip = (last_pixel_vy > 80.0 and pixel_vy < -20.0 and pixel_accel > 1500.0)
            cond_windowed = (max_inbound > 60.0 and pixel_vy <= 0.0 and delta_vy > 400.0)

            if bounce_cooldown == 0 and bounce_armed and len(vy_history) >= 3:
                if cond_sign_flip or cond_windowed:
                    min_dist_gt_bounce = min([abs(frame_idx - gtb) for gtb in gt_bounces]) if gt_bounces else 9999
                    min_dist_gt_frame = min([abs(frame_idx - gtf) for gtf in gt_ball_frames]) if gt_ball_frames else 9999
                    is_tp = min_dist_gt_bounce <= 3

                    bounce_details.append({
                        "frame": frame_idx,
                        "is_tp": is_tp,
                        "dist_gt_bounce": min_dist_gt_bounce,
                        "dist_gt_frame_window": min_dist_gt_frame,
                        "x": det_x,
                        "y": det_y,
                        "conf": best_conf
                    })
                    bounce_cooldown = 15
                    bounce_armed = False

            if not cond_sign_flip and not cond_windowed:
                bounce_armed = True
            last_pixel_vy = pixel_vy
        last_pixel_y = det_y

    cap.release()

    fps_far = [b for b in bounce_details if not b["is_tp"] and b["dist_gt_frame_window"] > 10]
    print(f"Found {len(fps_far)} FPs far from GT windows.")

    # Randomly select up to 20 FP frames
    np.random.seed(42)
    sample_fps = list(np.random.choice(fps_far, size=min(20, len(fps_far)), replace=False))
    sample_fps.sort(key=lambda b: b["frame"])

    # Re-open video to extract and render overlay PNGs for sample frames
    cap = cv2.VideoCapture(video_path)
    current_f = 0
    saved_count = 0
    target_frames = {b["frame"]: b for b in sample_fps}

    print("\nRendering 20 FP Bounce Overlay PNGs...")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        current_f += 1

        if current_f in target_frames:
            info = target_frames[current_f]
            pxX, pxY, conf = int(info["x"]), int(info["y"]), info["conf"]

            # Draw Table ROI box (approx footprint: X 350-1570, Y 450-950)
            cv2.rectangle(frame, (350, 450), (1570, 950), (255, 255, 0), 2)
            cv2.putText(frame, "Table ROI", (360, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Draw FP Detection Box / Crosshair (Red)
            cv2.circle(frame, (pxX, pxY), 12, (0, 0, 255), 2)
            cv2.drawMarker(frame, (pxX, pxY), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, f"FP Detection (Conf: {conf:.2f})", (pxX + 15, pxY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw GT location if GT exists
            if current_f in gt_ball_frames:
                gtX, gtY = int(gt_ball_frames[current_f][0]), int(gt_ball_frames[current_f][1])
                cv2.circle(frame, (gtX, gtY), 10, (0, 255, 0), 2)
                cv2.putText(frame, "GT Ball", (gtX + 15, gtY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "GT Ball: None (Unannotated Region)", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Banner overlay
            cv2.rectangle(frame, (0, 0), (750, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"FP Bounce Frame #{current_f} | Dist to GT Window: {info['dist_gt_frame_window']} frames", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            save_path = os.path.join(out_dir, f"fp_bounce_frame_{current_f:05d}.png")
            cv2.imwrite(save_path, frame)
            saved_count += 1
            print(f"  [{saved_count}/20] Saved {save_path} (Frame {current_f}, Conf: {conf:.2f}, X:{pxX}, Y:{pxY})")

    cap.release()
    print(f"\nSuccessfully generated {saved_count} FP overlay images in '{out_dir}'.")

if __name__ == "__main__":
    overlay_fp_bounces()
