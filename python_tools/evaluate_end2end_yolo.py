import json
import os
import glob
import subprocess

def evaluate_yolo_detections(label_dir, events_json_path, binary_path, tolerance_frames=3):
    if not os.path.exists(label_dir) or not os.path.exists(events_json_path):
        return 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0

    with open(events_json_path, 'r') as f:
        events_data = json.load(f)

    gt_bounce_frames = [int(f) for f, event in events_data.items() if event == "bounce"]
    gt_bounce_frames.sort()

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    yolo_frame_map = {}
    for lf in label_files:
        basename = os.path.basename(lf)
        try:
            frame_str = basename.split("_frame_")[-1].replace(".txt", "")
            f_idx = int(frame_str)
            with open(lf, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == '0':
                        cx = float(parts[1]) * 1920.0
                        cy = float(parts[2]) * 1080.0
                        yolo_frame_map[f_idx] = (cx, cy)
                        break
        except Exception:
            continue

    temp_json = "temp_yolo_markup.json"
    temp_dict = {str(k): {"x": v[0], "y": v[1]} for k, v in yolo_frame_map.items()}
    with open(temp_json, 'w') as f:
        json.dump(temp_dict, f)

    res = subprocess.run([binary_path, temp_json], capture_output=True, text=True)
    detected_frames = []
    for line in res.stdout.splitlines():
        if line.startswith("DETECTED_BOUNCES:"):
            raw = line.replace("DETECTED_BOUNCES:", "").strip()
            if raw:
                detected_frames = [int(x) for x in raw.split(",") if x]

    if os.path.exists(temp_json):
        os.remove(temp_json)

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

    return len(gt_bounce_frames), len(detected_frames), tp, fp, fn, precision, recall, f1

def main():
    binary_path = "backend_cpp/build/Release/eval_bounce.exe"
    if not os.path.exists(binary_path):
        binary_path = "backend_cpp/build/eval_bounce.exe"

    print("=========================================================================")
    print("  End-to-End System Evaluation (YOLO Detections -> C++ PhysicsEngine)")
    print("=========================================================================")

    sequences = [
        ("Held-out Test 1 (Primary System Metric)", "python_tools/dataset/labels/test_1", "python_tools/dataset/raw_markup/test_1/events_markup.json"),
        ("Game 1 (Training Split)", "python_tools/dataset/labels/train", "python_tools/dataset/raw_markup/game_1/events_markup.json")
    ]

    for name, label_dir, events_path in sequences:
        if not os.path.exists(label_dir) or not os.path.exists(events_path):
            continue

        gt_count, det_count, tp, fp, fn, prec, rec, f1 = evaluate_yolo_detections(label_dir, events_path, binary_path)

        print(f"\n--- Sequence: {name} ---")
        print(f"Ground Truth Bounces  : {gt_count}")
        print(f"YOLO + Engine Detected: {det_count}")
        print(f"True Positives (TP)   : {tp}")
        print(f"False Positives (FP)  : {fp}")
        print(f"False Negatives (FN)  : {fn}")
        print(f"Match Tolerance Window: +/- 3 frames (+/- 25 ms at 120 FPS)")
        print(f"System Precision      : {prec * 100.0:.2f}% ({prec:.4f})")
        print(f"System Recall         : {rec * 100.0:.2f}% ({rec:.4f})")
        print(f"System F1 Score       : {f1 * 100.0:.2f}% ({f1:.4f})")

    print("\n=========================================================================\n")

if __name__ == "__main__":
    main()
