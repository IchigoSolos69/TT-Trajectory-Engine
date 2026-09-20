import os
import json
import math
import random
import subprocess
import numpy as np

def get_git_commit_hash():
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        return out.decode('utf-8').strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def load_ground_truth(game_dir):
    ball_path = os.path.join(game_dir, "ball_markup.json")
    event_path = os.path.join(game_dir, "events_markup.json")

    with open(ball_path, 'r') as f:
        ball_data = json.load(f)
    with open(event_path, 'r') as f:
        event_data = json.load(f)

    bounce_frames = set()
    for frame_str, ev_name in event_data.items():
        if ev_name in ["bounce", "table_bounce", "bounce_table"] or "bounce" in str(ev_name).lower():
            bounce_frames.add(int(frame_str))

    frames_sorted = []
    positions = {}
    for frame_str, pos in ball_data.items():
        if pos is not None and isinstance(pos, dict) and "x" in pos and "y" in pos:
            fid = int(frame_str)
            frames_sorted.append(fid)
            positions[fid] = (float(pos["x"]), float(pos["y"]))

    frames_sorted.sort()
    return frames_sorted, positions, bounce_frames

def run_physics_bounce_detector(frames, positions, noise_sigma=0.0, dropout_rate=0.0, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    detected_bounces = []
    cooldown = 0

    vy_history = []
    prev_y = None
    prev_frame = None

    for fid in frames:
        if cooldown > 0:
            cooldown -= 1

        # Simulate frame dropout
        if dropout_rate > 0 and random.random() < dropout_rate:
            continue

        raw_x, raw_y = positions[fid]

        # Add Gaussian positional noise
        if noise_sigma > 0:
            noisy_x = raw_x + np.random.normal(0, noise_sigma)
            noisy_y = raw_y + np.random.normal(0, noise_sigma)
        else:
            noisy_x, noisy_y = raw_x, raw_y

        if prev_y is not None and prev_frame is not None:
            df = fid - prev_frame
            if df > 0:
                # Instantaneous pixel velocity (scaled to 120 FPS frame rate)
                vy = (noisy_y - prev_y) * (120.0 / df)
                vy_history.append(vy)
                if len(vy_history) > 3:
                    vy_history.pop(0)

                # 2-frame windowed velocity delta trigger: inbound (moving down +vy) vs outbound (moving up -vy)
                if len(vy_history) >= 2:
                    inbound_vy = vy_history[0]  # Positive vy means downward pixel movement
                    outbound_vy = vy_history[-1] # Negative vy means upward pixel movement
                    vy_delta = inbound_vy - outbound_vy

                    # Windowed delta trigger (> 400 px/s) + Cooldown (11 frames)
                    if vy_delta > 400.0 and cooldown == 0:
                        # Table bounds sanity filter
                        if 50 <= noisy_x <= 1870 and 100 <= noisy_y <= 1000:
                            detected_bounces.append(fid)
                            cooldown = 11

        prev_y = noisy_y
        prev_frame = fid

    return detected_bounces

def evaluate_predictions(detected_bounces, gt_bounces, tolerance_frames=5):
    matched_gt = set()
    tp = 0
    fp = 0

    for det in detected_bounces:
        match_found = False
        for gt in gt_bounces:
            if abs(det - gt) <= tolerance_frames and gt not in matched_gt:
                matched_gt.add(gt)
                tp += 1
                match_found = True
                break
        if not match_found:
            fp += 1

    fn = len(gt_bounces) - len(matched_gt)
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1

def main():
    git_hash = get_git_commit_hash()
    print("=========================================================================")
    print(f"  PHYSICS ENGINE NOISE SENSITIVITY BENCHMARK (Git Commit: {git_hash[:8]})")
    print("=========================================================================")

    dataset_dir = "python_tools/dataset/raw_markup"
    games = ["game_1", "test_1"]

    for gname in games:
        gdir = os.path.join(dataset_dir, gname)
        if not os.path.exists(gdir):
            continue

        frames, positions, gt_bounces = load_ground_truth(gdir)
        print(f"\n--- Dataset Sequence: {gname} ({len(frames)} ball frames, {len(gt_bounces)} GT bounces) ---")
        print(f"{'Gaussian Noise (px)':<20} | {'Dropout (%)':<12} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1 (%)':<8}")
        print("-" * 100)

        sigmas = [0.0, 0.5, 1.0, 2.0, 3.0]
        dropouts = [0.0, 0.05, 0.10, 0.15, 0.20]

        for sigma in sigmas:
            for drop in dropouts:
                detected = run_physics_bounce_detector(frames, positions, noise_sigma=sigma, dropout_rate=drop)
                tp, fp, fn, prec, rec, f1 = evaluate_predictions(detected, gt_bounces)
                print(f"{sigma:<20.1f} | {drop*100:<12.0f} | {tp:<4} | {fp:<4} | {fn:<4} | {prec:<14.2f} | {rec:<12.2f} | {f1:<8.2f}")

if __name__ == "__main__":
    main()
