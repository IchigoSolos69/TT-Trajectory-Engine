import os
import json
import shutil
import random
import cv2

def prepare_dataset():
    dataset_root = "python_tools/dataset"
    yolo_dir = os.path.join(dataset_root, "yolo_ball_game1")

    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)

    train_img_dir = os.path.join(yolo_dir, "images", "train")
    val_img_dir = os.path.join(yolo_dir, "images", "val")
    train_lbl_dir = os.path.join(yolo_dir, "labels", "train")
    val_lbl_dir = os.path.join(yolo_dir, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    ball_json_path = os.path.join(dataset_root, "raw_markup", "game_1", "ball_markup.json")
    video_path = os.path.join(dataset_root, "raw_videos", "game_1.mp4")

    if not os.path.exists(ball_json_path) or not os.path.exists(video_path):
        print(f"Error: Required files for game_1 missing ({ball_json_path} or {video_path}).")
        return False

    with open(ball_json_path, 'r') as f:
        ball_markup = json.load(f)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Loaded game_1.mp4: {total_frames} frames @ {fps} FPS ({width}x{height})")

    # 80/20 chronological split with a 1.0 second (120 frames) buffer gap
    train_cutoff = int(total_frames * 0.80)
    buffer_frames = int(fps * 1.0) # 120 frames buffer
    val_start = train_cutoff + buffer_frames

    print(f"Train split range : Frames 0 .. {train_cutoff}")
    print(f"Buffer gap range  : Frames {train_cutoff + 1} .. {val_start - 1} (1.0s gap skipped)")
    print(f"Val split range   : Frames {val_start} .. {total_frames}")

    # Classify all frames into train/val and positive/negative
    train_pos_frames = []
    train_neg_frames = []
    val_pos_frames = []
    val_neg_frames = []

    for fid in range(1, total_frames + 1):
        if train_cutoff < fid < val_start:
            continue
        
        fstr = str(fid)
        pos = ball_markup.get(fstr, None)
        is_pos = (pos is not None and isinstance(pos, dict) and "x" in pos and "y" in pos and pos["x"] is not None and pos["y"] is not None)
        
        if fid <= train_cutoff:
            if is_pos: train_pos_frames.append(fid)
            else: train_neg_frames.append(fid)
        elif fid >= val_start:
            if is_pos: val_pos_frames.append(fid)
            else: val_neg_frames.append(fid)

    # Cap negative frames at 3:1 ratio relative to positive ball frames
    random.seed(42)
    max_train_negs = len(train_pos_frames) * 3
    max_val_negs = len(val_pos_frames) * 3

    if len(train_neg_frames) > max_train_negs:
        selected_train_negs = set(random.sample(train_neg_frames, max_train_negs))
    else:
        selected_train_negs = set(train_neg_frames)

    if len(val_neg_frames) > max_val_negs:
        selected_val_negs = set(random.sample(val_neg_frames, max_val_negs))
    else:
        selected_val_negs = set(val_neg_frames)

    frames_to_export = set(train_pos_frames) | set(val_pos_frames) | selected_train_negs | selected_val_negs

    print(f"\nSubsampling & Capping Strategy:")
    print(f"  - Train Positives : {len(train_pos_frames)}")
    print(f"  - Train Negatives : {len(selected_train_negs)} (Capped 3:1 from {len(train_neg_frames)})")
    print(f"  - Val Positives   : {len(val_pos_frames)}")
    print(f"  - Val Negatives   : {len(selected_val_negs)} (Capped 3:1 from {len(val_neg_frames)})")

    box_w = 30.0 / width
    box_h = 30.0 / height
    frame_id = 0

    train_count = 0
    val_count = 0
    neg_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_id += 1

        if frame_id not in frames_to_export:
            continue

        is_train = (frame_id <= train_cutoff)
        target_img_dir = train_img_dir if is_train else val_img_dir
        target_lbl_dir = train_lbl_dir if is_train else val_lbl_dir

        frame_str = str(frame_id)
        ball_pos = ball_markup.get(frame_str, None)

        img_filename = f"game1_f{frame_id:06d}.jpg"
        lbl_filename = f"game1_f{frame_id:06d}.txt"

        img_out_path = os.path.join(target_img_dir, img_filename)
        lbl_out_path = os.path.join(target_lbl_dir, lbl_filename)

        cv2.imwrite(img_out_path, frame)

        if ball_pos is not None and isinstance(ball_pos, dict) and "x" in ball_pos and "y" in ball_pos and ball_pos["x"] is not None:
            bx = float(ball_pos["x"]) / width
            by = float(ball_pos["y"]) / height
            bx = max(0.0, min(1.0, bx))
            by = max(0.0, min(1.0, by))

            with open(lbl_out_path, 'w') as lf:
                lf.write(f"0 {bx:.6f} {by:.6f} {box_w:.6f} {box_h:.6f}\n")

            if is_train: train_count += 1
            else: val_count += 1
        else:
            with open(lbl_out_path, 'w') as lf:
                pass
            neg_count += 1

    cap.release()

    data_yaml_path = os.path.join(yolo_dir, "data.yaml")
    yaml_content = f"""path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val

names:
  0: ball
"""
    with open(data_yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset Preparation Complete:")
    print(f"  - Train Ball Frames : {train_count}")
    print(f"  - Val Ball Frames   : {val_count}")
    print(f"  - Capped Negatives  : {neg_count}")
    print(f"  - Saved to YAML     : {data_yaml_path}")
    return True

if __name__ == "__main__":
    prepare_dataset()
