import json
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import cv2
import numpy as np
import onnxruntime as ort

def analyze_204_bounces():
    model_path = "models/yolov11_ttgames.onnx"
    video_path = str(resolve_public_dataset_video("test_1"))
    ball_json = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    events_json = "python_tools/dataset/raw_markup/test_1/events_markup.json"

    with open(ball_json, 'r') as f:
        ball_raw = json.load(f)
    gt_ball_frames = sorted([int(k) for k, v in ball_raw.items() if float(v.get("x", -1)) >= 0 and float(v.get("y", -1)) >= 0])

    with open(events_json, 'r') as f:
        events_raw = json.load(f)
    gt_bounces = sorted([int(k) for k, v in events_raw.items() if v == "bounce" or (isinstance(v, dict) and v.get("type") == "bounce")])

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    imgW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    imgH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    dt = 1.0 / 120.0
    CONF_THRESH = 0.20

    # Run full clip detection & naive physics bounce simulation (mirroring C++ physics_engine)
    detections_by_frame = {}
    
    frame_idx = 0
    vy_history = []
    last_pixel_y = -1.0
    last_pixel_vy = 0.0
    bounce_cooldown = 0
    bounce_armed = True

    detected_bounces = []
    bounce_details = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        # Letterbox 640x640
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
                    detected_bounces.append(frame_idx)
                    bounce_cooldown = 15
                    bounce_armed = False

                    # Measure track length around frame_idx
                    track_len = 0
                    for check_f in range(frame_idx - 5, frame_idx + 6):
                        if check_f in detections_by_frame and detections_by_frame[check_f] is not None:
                            track_len += 1

                    # Distance to nearest GT annotated frame
                    min_dist_gt_frame = min([abs(frame_idx - gtf) for gtf in gt_ball_frames]) if gt_ball_frames else 9999
                    
                    # Distance to nearest GT bounce
                    min_dist_gt_bounce = min([abs(frame_idx - gtb) for gtb in gt_bounces]) if gt_bounces else 9999

                    is_tp = min_dist_gt_bounce <= 3
                    w_box = best_det[2] if best_det else 0
                    h_box = best_det[3] if best_det else 0
                    conf_val = best_det[4] if best_det else 0

                    bounce_details.append({
                        "frame": frame_idx,
                        "is_tp": is_tp,
                        "dist_gt_bounce": min_dist_gt_bounce,
                        "dist_gt_frame_window": min_dist_gt_frame,
                        "track_len": track_len,
                        "x": det_x,
                        "y": det_y,
                        "w_box": w_box,
                        "h_box": h_box,
                        "conf": conf_val
                    })

            if not cond_sign_flip and not cond_windowed:
                bounce_armed = True

            last_pixel_vy = pixel_vy
        last_pixel_y = det_y

    cap.release()

    print(f"\n=========================================================================")
    print(f"  ANALYSIS OF {len(detected_bounces)} DETECTED BOUNCES (test_1)")
    print(f"=========================================================================")
    tps = [b for b in bounce_details if b["is_tp"]]
    fps = [b for b in bounce_details if not b["is_tp"]]

    print(f"True Positives  (TP) : {len(tps)}")
    print(f"False Positives (FP) : {len(fps)}")

    # Tabulate distance to nearest GT annotated window for FPs
    fp_near_gt_window = [b for b in fps if b["dist_gt_frame_window"] <= 10]
    fp_far_gt_window  = [b for b in fps if b["dist_gt_frame_window"] > 10]

    print(f"  - FPs inside GT annotated event windows (+-10 frames) : {len(fp_near_gt_window)}")
    print(f"  - FPs FAR OUTSIDE GT annotated windows (>10 frames)    : {len(fp_far_gt_window)}")

    # Single-frame blips (track length <= 3 out of 11-frame window)
    blip_fps = [b for b in fps if b["track_len"] <= 3]
    print(f"  - FPs caused by isolated single-frame blips (track_len <= 3) : {len(blip_fps)} / {len(fps)} ({len(blip_fps)/len(fps)*100:.1f}%)")

    # Table ROI check (table is approx x in [350, 1570], y in [450, 950])
    roi_fps = [b for b in fps if (b["x"] < 350 or b["x"] > 1570 or b["y"] < 450 or b["y"] > 950)]
    print(f"  - FPs OUTSIDE Table ROI (x in [350,1570], y in [450,950])     : {len(roi_fps)} / {len(fps)}")

    # Box size check (ball bbox in 640x640 space is usually between 4px and 25px)
    box_fps = [b for b in fps if (b["w_box"] < 4 or b["w_box"] > 25 or b["h_box"] < 4 or b["h_box"] > 25)]
    print(f"  - FPs with invalid bbox size (w or h outside [4, 25] px)      : {len(box_fps)} / {len(fps)}")

if __name__ == "__main__":
    analyze_204_bounces()
