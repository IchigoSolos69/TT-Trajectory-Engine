import json
import os
import numpy as np

def test_bounce_logic(ball_json_path, events_json_path, min_drop_vy=60.0, max_rebound_vy=100.0, min_delta_vy=80.0, accel_thresh=1200.0):
    if not os.path.exists(ball_json_path) or not os.path.exists(events_json_path):
        return 0, 0, 0, 0, 0.0, 0.0, 0.0

    with open(ball_json_path, 'r') as f:
        ball_markup = json.load(f)

    with open(events_json_path, 'r') as f:
        events = json.load(f)

    gt_bounces = [int(k) for k, v in events.items() if v == "bounce"]
    gt_bounces.sort()

    frames = sorted([int(k) for k in ball_markup.keys() if k.isdigit()])

    detected_bounces = []

    last_pixel_y = -1.0
    last_pixel_vy = 0.0
    bounce_cooldown = 0
    dt = 1.0 / 120.0

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
            delta_vy = last_pixel_vy - pixel_vy

            # Table homography bounds check (px, py in 1920x1080)
            is_table_box = (px >= 200 and px <= 1720 and py >= 350 and py <= 1000)

            # Tuned condition: velocity drop spike OR sign flip
            if bounce_cooldown == 0 and is_table_box:
                condition_sign_flip = (last_pixel_vy > min_drop_vy and pixel_vy < 20.0 and delta_vy > min_delta_vy)
                condition_accel = (pixel_accel_spike > accel_thresh and delta_vy > min_delta_vy and last_pixel_vy > 40.0)

                if (condition_sign_flip or condition_accel):
                    detected_bounces.append(f_idx)
                    bounce_cooldown = 15

            last_pixel_vy = pixel_vy
        last_pixel_y = py

    # Score against GT with +/- 3 frame tolerance
    tolerance = 3
    tp = 0
    fp = 0
    matched_gt = set()

    for det in detected_bounces:
        match_found = False
        for gt in gt_bounces:
            if abs(det - gt) <= tolerance:
                tp += 1
                matched_gt.add(gt)
                match_found = True
                break
        if not match_found:
            fp += 1

    fn = len(gt_bounces) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return len(gt_bounces), len(detected_bounces), tp, fp, fn, prec, rec, f1

def main():
    game1_ball = "python_tools/dataset/raw_markup/game_1/ball_markup.json"
    game1_events = "python_tools/dataset/raw_markup/game_1/events_markup.json"

    test1_ball = "python_tools/dataset/raw_markup/test_1/ball_markup.json"
    test1_events = "python_tools/dataset/raw_markup/test_1/events_markup.json"

    print("--- Tuning Bounce Detector on game_1 (Training Split) ---")
    gt, det, tp, fp, fn, prec, rec, f1 = test_bounce_logic(game1_ball, game1_events)
    print(f"game_1 Score -> GT: {gt} | Det: {det} | TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"  Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%\n")

    print("--- Evaluating Tuned Detector on Held-Out test_1 ---")
    gt_t, det_t, tp_t, fp_t, fn_t, prec_t, rec_t, f1_t = test_bounce_logic(test1_ball, test1_events)
    print(f"test_1 Score -> GT: {gt_t} | Det: {det_t} | TP: {tp_t} | FP: {fp_t} | FN: {fn_t}")
    print(f"  Precision: {prec_t*100:.2f}% | Recall: {rec_t*100:.2f}% | F1: {f1_t*100:.2f}%\n")

if __name__ == "__main__":
    main()
