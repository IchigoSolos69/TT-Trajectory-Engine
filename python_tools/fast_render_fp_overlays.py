import json
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import cv2
import numpy as np
import os
import random

def fast_render_fp_overlays():
    video_path = str(resolve_public_dataset_video("test_1"))
    ball_json = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    events_json = "python_tools/dataset/raw_markup/test_1/events_markup.json"
    fp_json = "python_tools/fp_bounces_list.json"
    out_dir = "python_tools/fp_samples"
    os.makedirs(out_dir, exist_ok=True)

    with open(fp_json, 'r') as f:
        fp_data = json.load(f)
    raw_bounces = fp_data.get("raw_bounces", [])

    with open(ball_json, 'r') as f:
        ball_raw = json.load(f)
    gt_ball_frames = {int(k): (float(v["x"]), float(v["y"])) for k, v in ball_raw.items() if float(v.get("x", -1)) >= 0 and float(v.get("y", -1)) >= 0}

    with open(events_json, 'r') as f:
        events_raw = json.load(f)
    gt_bounces = sorted([int(k) for k, v in events_raw.items() if v == "bounce" or (isinstance(v, dict) and v.get("type") == "bounce")])

    # Find FPs >10 frames away from any GT bounce
    fp_far_bounces = []
    for b_frame in raw_bounces:
        dist_gt_bounce = min([abs(b_frame - gtb) for gtb in gt_bounces]) if gt_bounces else 9999
        dist_gt_ball = min([abs(b_frame - gtf) for gtf in gt_ball_frames]) if gt_ball_frames else 9999
        if dist_gt_bounce > 3:
            fp_far_bounces.append({
                "frame": b_frame,
                "dist_gt_bounce": dist_gt_bounce,
                "dist_gt_ball": dist_gt_ball
            })

    print(f"Total FP Bounces (Raw Feed): {len(fp_far_bounces)}")
    
    # Randomly sample 20 FP bounce frames (seeded for reproducibility)
    random.seed(42)
    sample_fps = sorted(random.sample(fp_far_bounces, min(20, len(fp_far_bounces))), key=lambda x: x["frame"])

    cap = cv2.VideoCapture(video_path)
    imgW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    imgH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved_files = []
    print(f"\nRendering {len(sample_fps)} FP Overlay PNGs into '{out_dir}'...")

    for idx, item in enumerate(sample_fps, 1):
        target_f = item["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f - 1)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Draw Table ROI Box (approx camera footprint: X 350-1570, Y 450-950)
        cv2.rectangle(frame, (350, 450), (1570, 950), (255, 255, 0), 2)
        cv2.putText(frame, "Table ROI Footprint", (360, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Indicate GT ball status
        if target_f in gt_ball_frames:
            gtX, gtY = int(gt_ball_frames[target_f][0]), int(gt_ball_frames[target_f][1])
            cv2.circle(frame, (gtX, gtY), 10, (0, 255, 0), 2)
            cv2.putText(frame, "GT Ball", (gtX + 15, gtY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unannotated Inter-Rally Region (No GT Ball)", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Header banner
        cv2.rectangle(frame, (0, 0), (900, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FP Bounce Sample [{idx}/20] | Frame #{target_f} | Nearest GT Event: {item['dist_gt_bounce']} frames", 
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_path = os.path.join(out_dir, f"fp_bounce_frame_{target_f:05d}.png")
        cv2.imwrite(out_path, frame)
        saved_files.append(out_path)
        print(f"  [{idx}/{len(sample_fps)}] Saved {out_path} (Frame {target_f}, Nearest GT: {item['dist_gt_bounce']} frames)")

    cap.release()
    print(f"\nAll {len(saved_files)} FP overlay images rendered successfully!")

if __name__ == "__main__":
    fast_render_fp_overlays()
