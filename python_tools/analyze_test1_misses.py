import json
import os
import subprocess
import numpy as np

def analyze_misses_and_fps():
    events_path = "python_tools/dataset/raw_markup/test_1/events_markup.json"
    ball_path = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    binary_path = "backend_cpp/build/Release/eval_bounce.exe"
    if not os.path.exists(binary_path):
        binary_path = "backend_cpp/build/eval_bounce.exe"

    if not os.path.exists(events_path) or not os.path.exists(ball_path):
        print("Error: Missing test_1 markup files.")
        return

    with open(events_path, 'r') as f:
        events = json.load(f)

    with open(ball_path, 'r') as f:
        ball_markup = json.load(f)

    gt_bounces = [int(k) for k, v in events.items() if v == "bounce"]
    gt_bounces.sort()

    res = subprocess.run([binary_path, ball_path], capture_output=True, text=True)
    detected_bounces = []
    for line in res.stdout.splitlines():
        if line.startswith("DETECTED_BOUNCES:"):
            raw = line.replace("DETECTED_BOUNCES:", "").strip()
            if raw:
                detected_bounces = [int(x) for x in raw.split(",") if x]

    tolerance = 3
    tps = []
    fps = []
    fns = []

    matched_gt = set()

    for det in detected_bounces:
        match_found = False
        for gt in gt_bounces:
            if abs(det - gt) <= tolerance:
                tps.append((det, gt))
                matched_gt.add(gt)
                match_found = True
                break
        if not match_found:
            fps.append(det)

    for gt in gt_bounces:
        if gt not in matched_gt:
            fns.append(gt)

    print("=========================================================================")
    print(f"  test_1 BOUNCE MISSED (FN: {len(fns)}) & FALSE POSITIVE (FP: {len(fps)}) ANALYSIS")
    print("=========================================================================")
    print(f"Total GT Bounces: {len(gt_bounces)} | Detected: {len(detected_bounces)} | TP: {len(tps)} | FN: {len(fns)} | FP: {len(fps)}")

    def get_frame_info(f_idx):
        info = {"x": -1, "y": -1, "speed_px_frame": 0.0, "gap": False, "loc": "Unknown"}
        str_idx = str(f_idx)
        if str_idx not in ball_markup:
            info["gap"] = True
            return info

        curr = ball_markup[str_idx]
        px, py = float(curr.get("x", -1)), float(curr.get("y", -1))
        if px < 0 or py < 0:
            info["gap"] = True
            return info

        info["x"] = px
        info["y"] = py

        # Check image location (1920x1080)
        # Table region in camera view: X ~ [300, 1620], Y ~ [400, 950]
        if py < 450 or py > 900:
            info["loc"] = "Table Edge / Off-Table"
        elif abs(px - 960) < 150:
            info["loc"] = "Net Area"
        else:
            info["loc"] = "Center Table"

        # Calculate speed over 5-frame window
        prev_str = str(f_idx - 2)
        next_str = str(f_idx + 2)
        if prev_str in ball_markup and next_str in ball_markup:
            p_prev = ball_markup[prev_str]
            p_next = ball_markup[next_str]
            x1, y1 = float(p_prev.get("x", -1)), float(p_prev.get("y", -1))
            x2, y2 = float(p_next.get("x", -1)), float(p_next.get("y", -1))
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                info["speed_px_frame"] = dist / 4.0

        return info

    print("\n--- BUCKETED MISSED BOUNCES (9 False Negatives) ---")
    speed_buckets = {"Low Speed (< 10 px/f)": 0, "Medium Speed (10-25 px/f)": 0, "High Speed / Blur (> 25 px/f)": 0}
    loc_buckets = {"Center Table": 0, "Net Area": 0, "Table Edge / Off-Table": 0}
    gap_count = 0

    for idx, fn_frame in enumerate(fns, 1):
        info = get_frame_info(fn_frame)
        gap_str = "YES (Missing Markup)" if info["gap"] else "No"
        if info["gap"]:
            gap_count += 1

        sp = info["speed_px_frame"]
        if sp < 10.0:
            speed_cat = "Low Speed (< 10 px/f)"
        elif sp <= 25.0:
            speed_cat = "Medium Speed (10-25 px/f)"
        else:
            speed_cat = "High Speed / Blur (> 25 px/f)"
        speed_buckets[speed_cat] += 1

        loc_cat = info["loc"]
        if loc_cat in loc_buckets:
            loc_buckets[loc_cat] += 1

        print(f"FN #{idx:02d} | Frame {fn_frame:4d} | Loc: ({info['x']:6.1f}, {info['y']:6.1f}) [{loc_cat:22s}] | Speed: {sp:5.1f} px/f ({sp*120*0.001:4.1f} m/s approx) | Annotation Gap: {gap_str}")

    print("\nFN Breakdown by Ball Speed (Motion Blur Proxy):")
    for cat, cnt in speed_buckets.items():
        print(f"  - {cat:30s}: {cnt} ({cnt/len(fns)*100:.1f}%)")

    print("\nFN Breakdown by Image Location:")
    for cat, cnt in loc_buckets.items():
        print(f"  - {cat:30s}: {cnt} ({cnt/len(fns)*100:.1f}%)")

    print(f"\nFN Hits Annotation Gaps: {gap_count} / {len(fns)} ({gap_count/len(fns)*100:.1f}%)")

    print("\n--- FALSE POSITIVE ANALYSIS (3 False Positives) ---")
    for idx, fp_frame in enumerate(fps, 1):
        info = get_frame_info(fp_frame)
        print(f"FP #{idx:02d} | Frame {fp_frame:4d} | Loc: ({info['x']:6.1f}, {info['y']:6.1f}) [{info['loc']:22s}] | Speed: {info['speed_px_frame']:5.1f} px/f")

if __name__ == "__main__":
    analyze_misses_and_fps()
