import cv2
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import json
import os

def convert_test1_labels(json_path, video_path, image_out_dir, label_out_dir, prefix="test_1"):
    print(f"Extracting held-out {prefix} frames and labels from {video_path}...")

    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    if not os.path.exists(json_path) or not os.path.exists(video_path):
        print(f"Error: Missing files for {prefix}")
        return

    with open(json_path, 'r') as f:
        markup_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    BALL_W_PIXELS = 30.0
    BALL_H_PIXELS = 30.0
    frames_processed = 0

    for frame_str, coords in markup_data.items():
        try:
            frame_idx = int(frame_str)
            x_pixel = float(coords['x'])
            y_pixel = float(coords['y'])
        except (ValueError, KeyError, TypeError):
            continue

        if x_pixel < 0 or y_pixel < 0:
            continue

        x_center = x_pixel / width
        y_center = y_pixel / height
        norm_w = BALL_W_PIXELS / width
        norm_h = BALL_H_PIXELS / height

        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"

        label_name = f"{prefix}_frame_{frame_idx}.txt"
        with open(os.path.join(label_out_dir, label_name), 'w') as lf:
            lf.write(yolo_line)

        frames_processed += 1

    cap.release()
    print(f"Done! Extracted {frames_processed} YOLO ground-truth labels for held-out {prefix}.")

if __name__ == "__main__":
    convert_test1_labels(
        json_path="python_tools/dataset/raw_markup/test_1/ball_markup.json",
        video_path=str(resolve_public_dataset_video("test_1")),
        image_out_dir="python_tools/dataset/images/test_1",
        label_out_dir="python_tools/dataset/labels/test_1",
        prefix="test_1"
    )
