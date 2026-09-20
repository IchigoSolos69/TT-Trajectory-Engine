import json
import os
import numpy as np

def inspect_missed_bounces():
    ball_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    events_path = "python_tools/dataset/raw_markup/test_1/events_markup.json"

    if not os.path.exists(ball_path) or not os.path.exists(events_path):
        print("Error: Missing test_1 markup files.")
        return

    with open(ball_path, 'r') as f:
        ball_markup = json.load(f)

    with open(events_path, 'r') as f:
        events = json.load(f)

    gt_bounces = [int(k) for k, v in events.items() if v == "bounce"]
    gt_bounces.sort()

    # Original engine bounce detection logic
    detected_bounces = []
    frames = sorted([int(k) for k in ball_markup.keys() if k.isdigit()])

    dt = 1.0 / 120.0
    last_pixel_y = -1.0
    last_pixel_vy = 0.0
    bounce_cooldown = 0

    for f_idx in frames:
        curr = ball_markup[str(f_idx)]
        px = float(curr.get("x", -1))
        py = float(curr.get("y", -1))

        if px < 0 or py < 0:
            last_pixel_y = -1.0
            last_pixel_vy = 0.0
            continue

        if bounce_cooldown > 0:
            bounce_cooldown -= 1

        if last_pixel_y >= 0:
            pixel_vy = (py - last_pixel_y) / dt
            pixel_accel_spike = (last_pixel_vy - pixel_vy) / dt

            is_table = (px >= 200 and px <= 1720 and py >= 350 and py <= 1000)

            # Original baseline threshold
            if bounce_cooldown == 0 and last_pixel_vy > 80.0 and pixel_vy < -20.0 and pixel_accel_spike > 1500.0 and is_table:
                detected_bounces.append(f_idx)
                bounce_cooldown = 15

            last_pixel_vy = pixel_vy
        last_pixel_y = py

    tolerance = 3
    tps = []
    fns = []

    matched_gt = set()
    for det in detected_bounces:
        for gt in gt_bounces:
            if abs(det - gt) <= tolerance:
                matched_gt.add(gt)
                break

    for gt in gt_bounces:
        if gt not in matched_gt:
            fns.append(gt)

    print("=========================================================================")
    print(f"  test_1 INSPECTION OF SURROUNDING pixelVy OVER +/- 5 FRAMES FOR 8 FN MISSES")
    print("=========================================================================")
    print(f"GT Bounces: {len(gt_bounces)} | Original Detected: {len(detected_bounces)} | FN Missed: {len(fns)}")

    for idx, fn_frame in enumerate(fns, 1):
        print(f"\n--- FN #{idx:02d} (GT Bounce at Frame {fn_frame}) ---")
        print(f"{'Frame':>8} | {'pixelY (px)':>12} | {'pixelVy (px/s)':>15} | {'accelSpike':>15} | Threshold Evaluation Notes")
        print("-" * 75)

        for offset in range(-5, 6):
            target_f = fn_frame + offset
            target_str = str(target_f)
            prev_str = str(target_f - 1)

            if target_str in ball_markup and prev_str in ball_markup:
                py_curr = float(ball_markup[target_str].get("x", -1)), float(ball_markup[target_str].get("y", -1))
                py_prev = float(ball_markup[prev_str].get("x", -1)), float(ball_markup[prev_str].get("y", -1))

                if py_curr[1] > 0 and py_prev[1] > 0:
                    vy = (py_curr[1] - py_prev[1]) / dt

                    # Calculate accel spike
                    prev2_str = str(target_f - 2)
                    accel = 0.0
                    if prev2_str in ball_markup:
                        py_prev2 = float(ball_markup[prev2_str].get("y", -1))
                        if py_prev2 > 0:
                            vy_prev = (py_prev[1] - py_prev2) / dt
                            accel = (vy_prev - vy) / dt

                    note = ""
                    if vy < -20.0:
                        note += "Rebound OK (Vy < -20) "
                    elif vy < 0:
                        note += "Shallow Rebound (Vy in [-20, 0]) "
                    else:
                        note += "No Upward Rebound (Vy > 0) "

                    if vy > 80.0:
                        note += "| High Inbound (Vy > 80)"

                    marker = " <-- GT BOUNCE FRAME" if offset == 0 else ""
                    print(f"{target_f:8d} | {py_curr[1]:12.1f} | {vy:15.1f} | {accel:15.1f} | {note}{marker}")
                else:
                    print(f"{target_f:8d} | Missing / Off-screen")

if __name__ == "__main__":
    inspect_missed_bounces()
