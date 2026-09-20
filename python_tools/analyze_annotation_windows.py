import json
import os
import glob
import numpy as np

def analyze_dataset_markup(sequence_dir):
    ball_path = os.path.join(sequence_dir, "ball_markup.json")
    events_path = os.path.join(sequence_dir, "events_markup.json")
    
    if not os.path.exists(ball_path):
        print(f"Skipping {sequence_dir}: {ball_path} not found.")
        return None
        
    with open(ball_path) as f:
        ball_data = json.load(f)
        
    # Extract annotated positive ball frames (x >= 0 or x != -1)
    pos_frames = sorted([int(k) for k, v in ball_data.items() if v.get("x", -1) >= 0])
    
    # Calculate contiguous runs
    runs = []
    if pos_frames:
        start = pos_frames[0]
        for a, b in zip(pos_frames, pos_frames[1:] + [None]):
            if b != a + 1:
                runs.append((start, a, a - start + 1))
                start = b

    run_lengths = [r[2] for r in runs]
    
    # Analyze events if available
    events_info = []
    if os.path.exists(events_path):
        with open(events_path) as f:
            events_data = json.load(f)
        # Handle dict or list structure of events_markup.json
        if isinstance(events_data, dict):
            for k, v in events_data.items():
                event_type = v if isinstance(v, str) else v.get("type", "unknown")
                events_info.append((int(k), event_type))
        elif isinstance(events_data, list):
            for item in events_data:
                if isinstance(item, dict):
                    events_info.append((int(item["frame"]), item.get("type", "unknown")))
                
    events_info.sort(key=lambda x: x[0])
    bounce_frames = [e[0] for e in events_info if "bounce" in e[1].lower()]
    
    # Measure distance from GT bounces to nearest annotated frame run center
    distances = []
    for b in bounce_frames:
        min_d = min([abs(b - ((r[0] + r[1]) / 2.0)) for r in runs]) if runs else float('inf')
        distances.append(min_d)
        
    res = {
        "sequence": os.path.basename(sequence_dir),
        "total_positive_frames": len(pos_frames),
        "total_runs": len(runs),
        "min_run_len": int(np.min(run_lengths)) if run_lengths else 0,
        "max_run_len": int(np.max(run_lengths)) if run_lengths else 0,
        "median_run_len": float(np.median(run_lengths)) if run_lengths else 0,
        "mean_run_len": float(np.mean(run_lengths)) if run_lengths else 0,
        "total_bounces": len(bounce_frames),
        "mean_dist_bounce_to_run_center": float(np.mean(distances)) if distances else 0
    }
    
    print(f"=== {res['sequence']} Analysis ===")
    print(f"  Positive Ball Frames: {res['total_positive_frames']}")
    print(f"  Contiguous Event Windows (Runs): {res['total_runs']}")
    print(f"  Run Lengths: min={res['min_run_len']}, median={res['median_run_len']:.1f}, max={res['max_run_len']}")
    print(f"  Total Ground-Truth Bounces: {res['total_bounces']}")
    if res['total_bounces'] > 0:
        print(f"  Avg Distance Bounce to Window Center: {res['mean_dist_bounce_to_run_center']:.2f} frames")
    print()
    return res

if __name__ == "__main__":
    markup_dirs = glob.glob("python_tools/dataset/raw_markup/*")
    for d in sorted(markup_dirs):
        if os.path.isdir(d):
            analyze_dataset_markup(d)
