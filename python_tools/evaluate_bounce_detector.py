import subprocess
import json
import os
import sys

def run_cpp_eval(binary_path, json_ball_path):
    if not os.path.exists(binary_path):
        print(f"Error: Missing C++ executable at {binary_path}")
        return []
    if not os.path.exists(json_ball_path):
        print(f"Error: Missing ball markup at {json_ball_path}")
        return []

    res = subprocess.run([binary_path, json_ball_path], capture_output=True, text=True)
    detected_bounces = []
    for line in res.stdout.splitlines():
        if line.startswith("DETECTED_BOUNCES:"):
            raw = line.replace("DETECTED_BOUNCES:", "").strip()
            if raw:
                detected_bounces = [int(x) for x in raw.split(",") if x]
    return detected_bounces

def score_bounce_detection(detected_frames, events_json_path, tolerance_frames=3):
    if not os.path.exists(events_json_path):
        return 0, 0, 0, 0.0, 0.0, 0.0

    with open(events_json_path, 'r') as f:
        events_data = json.load(f)

    gt_bounce_frames = [int(f) for f, event in events_data.items() if event == "bounce"]
    gt_bounce_frames.sort()

    tp = 0
    fp = 0
    matched_gt = set()

    for det in detected_frames:
        match_found = False
        for gt in gt_bounce_frames:
            if abs(det - gt) <= tolerance_frames:
                tp += 1
                matched_gt.add(gt)
                match_found = True
                break
        if not match_found:
            fp += 1

    fn = len(gt_bounce_frames) - len(matched_gt)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return len(gt_bounce_frames), tp, fp, fn, precision, recall, f1

def main():
    binary_path = "backend_cpp/build/Release/eval_bounce.exe"
    if not os.path.exists(binary_path):
        binary_path = "backend_cpp/build/eval_bounce.exe"

    print("=========================================================================")
    print("  Authoritative C++ PhysicsEngine OpenTTGames Dataset Bounce Evaluation")
    print("=========================================================================")

    sequences = [
        ("Game 1 (Validation)", "python_tools/dataset/raw_markup/game_1/ball_markup.json", "python_tools/dataset/raw_markup/game_1/events_markup.json"),
        ("Test 1 (Held-out Test)", "python_tools/dataset/raw_markup/test_1/ball_markup.json", "python_tools/dataset/raw_markup/test_1/events_markup.json")
    ]

    for name, ball_path, events_path in sequences:
        if not os.path.exists(ball_path) or not os.path.exists(events_path):
            continue

        detected = run_cpp_eval(binary_path, ball_path)
        gt_count, tp, fp, fn, prec, rec, f1 = score_bounce_detection(detected, events_path, tolerance_frames=3)

        print(f"\n--- Sequence: {name} ---")
        print(f"Ground Truth Bounces  : {gt_count}")
        print(f"C++ Engine Detected   : {len(detected)}")
        print(f"True Positives (TP)   : {tp}")
        print(f"False Positives (FP)  : {fp}")
        print(f"False Negatives (FN)  : {fn}")
        print(f"Match Tolerance Window: +/- 3 frames (+/- 25 ms at 120 FPS)")
        print(f"Bounce Precision      : {prec * 100.0:.2f}% ({prec:.4f})")
        print(f"Bounce Recall         : {rec * 100.0:.2f}% ({rec:.4f})")
        print(f"Bounce F1 Score       : {f1 * 100.0:.2f}% ({f1:.4f})")

    print("\n=========================================================================\n")

if __name__ == "__main__":
    main()
