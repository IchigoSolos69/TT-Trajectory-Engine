import json
import numpy as np

def tabulate_bounces():
    # Load 880 GT ball frames
    with open("python_tools/dataset/raw_markup/test_1/ball_markup.json", "r") as f:
        ball_raw = json.load(f)
    gt_ball_frames = set([int(k) for k, v in ball_raw.items() if float(v.get("x", -1)) >= 0 and float(v.get("y", -1)) >= 0])

    # Load 31 GT bounce frames
    with open("python_tools/dataset/raw_markup/test_1/events_markup.json", "r") as f:
        events_raw = json.load(f)
    gt_bounces = sorted([int(k) for k, v in events_raw.items() if v == "bounce" or (isinstance(v, dict) and v.get("type") == "bounce")])

    print(f"Total GT Ball Frames: {len(gt_ball_frames)} | Total GT Bounces: {len(gt_bounces)}")
    print("GT Bounces:", gt_bounces)

if __name__ == "__main__":
    tabulate_bounces()
