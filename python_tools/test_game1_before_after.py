import json
import os

def eval_game1_metrics():
    ball_path = "python_tools/dataset/raw_markup/game_1/ball_markup.json"
    events_path = "python_tools/dataset/raw_markup/game_1/events_markup.json"

    with open(ball_path, 'r') as f:
        ball_markup = json.load(f)

    with open(events_path, 'r') as f:
        events = json.load(f)

    gt_bounces = [int(k) for k, v in events.items() if v == "bounce"]
    gt_bounces.sort()

    frames = sorted([int(k) for k in ball_markup.keys() if k.isdigit()])
    dt = 1.0 / 120.0

    # 1. Baseline Adjacent-Frame Detector
    det_baseline = []
    last_pixel_y = -1.0
    last_pixel_vy = 0.0
    cooldown = 0

    for f_idx in frames:
        curr = ball_markup[str(f_idx)]
        px = float(curr.get("x", -1))
        py = float(curr.get("y", -1))

        if px < 0 or py < 0:
            last_pixel_y = -1.0
            last_pixel_vy = 0.0
            continue

        if cooldown > 0:
            cooldown -= 1

        if last_pixel_y >= 0:
            pixel_vy = (py - last_pixel_y) / dt
            pixel_accel_spike = (last_pixel_vy - pixel_vy) / dt
            is_table = (px >= 200 and px <= 1720 and py >= 350 and py <= 1000)

            if cooldown == 0 and last_pixel_vy > 80.0 and pixel_vy < -20.0 and pixel_accel_spike > 1500.0 and is_table:
                det_baseline.append(f_idx)
                cooldown = 15

            last_pixel_vy = pixel_vy
        last_pixel_y = py

    # 2. Windowed 2-Frame Velocity Delta Detector
    det_windowed = []
    last_pixel_y = -1.0
    history_vy = []
    cooldown_w = 0

    for f_idx in frames:
        curr = ball_markup[str(f_idx)]
        px = float(curr.get("x", -1))
        py = float(curr.get("y", -1))

        if px < 0 or py < 0:
            last_pixel_y = -1.0
            history_vy = []
            continue

        if cooldown_w > 0:
            cooldown_w -= 1

        if last_pixel_y >= 0:
            pixel_vy = (py - last_pixel_y) / dt
            history_vy.append(pixel_vy)
            if len(history_vy) > 4:
                history_vy.pop(0)

            is_table = (px >= 200 and px <= 1720 and py >= 350 and py <= 1000)

            if cooldown_w == 0 and len(history_vy) >= 3 and is_table:
                # Check 2-frame velocity change delta: max_inbound - min_outbound
                inbound_vy = max(history_vy[:-1])
                outbound_vy = history_vy[-1]
                delta_vy = inbound_vy - outbound_vy

                if (inbound_vy > 60.0 and outbound_vy <= 0.0 and delta_vy > 400.0):
                    det_windowed.append(f_idx)
                    cooldown_w = 15

        last_pixel_y = py

    def score(detected):
        tolerance = 3
        tp = 0
        fp = 0
        matched_gt = set()
        for det in detected:
            match = False
            for gt in gt_bounces:
                if abs(det - gt) <= tolerance:
                    tp += 1
                    matched_gt.add(gt)
                    match = True
                    break
            if not match:
                fp += 1
        fn = len(gt_bounces) - len(matched_gt)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return len(detected), tp, fp, fn, prec, rec, f1

    det_b, tp_b, fp_b, fn_b, prec_b, rec_b, f1_b = score(det_baseline)
    det_w, tp_w, fp_w, fn_w, prec_w, rec_w, f1_w = score(det_windowed)

    print("=========================================================================")
    print("  game_1 (TRAINING SET) BOUNCE DETECTOR PERFORMANCE: BEFORE & AFTER TUNING")
    print("=========================================================================")
    print(f"Ground-Truth Bounces on game_1: {len(gt_bounces)}")
    print("\n--- BASELINE (Single-Frame Adjacent Sign-Flip Detector) ---")
    print(f"  Detected: {det_b} | TP: {tp_b} | FP: {fp_b} | FN: {fn_b}")
    print(f"  Precision: {prec_b*100:.2f}% | Recall: {rec_b*100:.2f}% | F1 Score: {f1_b*100:.2f}%")

    print("\n--- TUNED (2-Frame Windowed Delta Trigger) ---")
    print(f"  Detected: {det_w} | TP: {tp_w} | FP: {fp_w} | FN: {fn_w}")
    print(f"  Precision: {prec_w*100:.2f}% | Recall: {rec_w*100:.2f}% | F1 Score: {f1_w*100:.2f}%")
    print("=========================================================================\n")

if __name__ == "__main__":
    eval_game1_metrics()
