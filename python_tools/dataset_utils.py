import json
import os
import glob

def load_gt_ball_markup(ball_json_path):
    """
    Authoritative single-source-of-truth loader for ball GT markup.
    Returns a dict mapping integer frame_id -> (x, y) for all frames with x >= 0 and y >= 0.
    """
    if not os.path.exists(ball_json_path):
        return {}
    with open(ball_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    gt_frames = {}
    for k, v in data.items():
        if isinstance(v, dict) and "x" in v and "y" in v:
            x_val = v.get("x")
            y_val = v.get("y")
            if x_val is not None and y_val is not None:
                try:
                    fx, fy = float(x_val), float(y_val)
                    if fx >= 0.0 and fy >= 0.0:
                        gt_frames[int(k)] = (fx, fy)
                except ValueError:
                    continue
    return gt_frames

def load_gt_events_markup(events_json_path):
    """
    Authoritative single-source-of-truth loader for event GT markup.
    Returns:
      bounce_frames: set of frame indices where event is a bounce
      all_events: dict mapping frame_id -> event_type_string
      counts: dict mapping event_type -> count
    """
    if not os.path.exists(events_json_path):
        return set(), {}, {}
        
    with open(events_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bounce_frames = set()
    all_events = {}
    counts = {}
    
    if isinstance(data, dict):
        for k, v in data.items():
            event_type = v if isinstance(v, str) else v.get("type", "unknown")
            try:
                frame_idx = int(k)
                all_events[frame_idx] = event_type
                counts[event_type] = counts.get(event_type, 0) + 1
                if "bounce" in event_type.lower():
                    bounce_frames.add(frame_idx)
            except ValueError:
                continue
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                frame_idx = int(item.get("frame", -1))
                event_type = item.get("type", "unknown")
                if frame_idx >= 0:
                    all_events[frame_idx] = event_type
                    counts[event_type] = counts.get(event_type, 0) + 1
                    if "bounce" in event_type.lower():
                        bounce_frames.add(frame_idx)
                        
    return bounce_frames, all_events, counts

if __name__ == "__main__":
    for seq in ["game_1", "test_1"]:
        ball_p = f"python_tools/dataset/raw_markup/{seq}/ball_markup.json"
        event_p = f"python_tools/dataset/raw_markup/{seq}/events_markup.json"
        
        gt_ball = load_gt_ball_markup(ball_p)
        bounces, all_ev, counts = load_gt_events_markup(event_p)
        
        print(f"=== {seq} Standardized Counts ===")
        print(f"  Total Annotated Positive Ball Frames: {len(gt_ball)}")
        print(f"  Total Bounces: {len(bounces)}")
        print(f"  Event Breakdown: {counts}\n")
