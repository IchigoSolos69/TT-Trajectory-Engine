import cv2
import json
import os

def convert_to_yolo_format(json_path, video_path, image_out_dir, label_out_dir, prefix):
    print(f"Processing {prefix} video from {video_path}...")
    
    # Ensure output directories exist
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"Error: Could not find markup file at {json_path}")
        return

    if not os.path.exists(video_path):
        print(f"Error: Could not find video file at {video_path}")
        return

    with open(json_path, 'r') as f:
        markup_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if width <= 0 or height <= 0:
        print(f"Error: Invalid video dimensions for {video_path}")
        cap.release()
        return

    # Fixed bounding box size for table tennis ball (30x30 pixels in camera space)
    BALL_W_PIXELS = 30.0
    BALL_H_PIXELS = 30.0

    frames_processed = 0

    # Iterate through dict keys (frame numbers)
    for frame_str, coords in markup_data.items():
        try:
            frame_idx = int(frame_str)
            x_pixel = float(coords['x'])
            y_pixel = float(coords['y'])
        except (ValueError, KeyError, TypeError):
            continue

        # Skip frames where ball is missing / off-screen (represented as -1, -1)
        if x_pixel < 0 or y_pixel < 0:
            continue

        # Seek video to frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue

        # Convert pixel coordinates to YOLO normalized coordinates (0.0 to 1.0)
        x_center = x_pixel / width
        y_center = y_pixel / height
        norm_w = BALL_W_PIXELS / width
        norm_h = BALL_H_PIXELS / height

        # YOLO Class 0 = Ball
        class_id = 0
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"

        # Save Frame Image
        img_name = f"{prefix}_frame_{frame_idx}.jpg"
        cv2.imwrite(os.path.join(image_out_dir, img_name), frame)

        # Save YOLO Label TXT
        label_name = f"{prefix}_frame_{frame_idx}.txt"
        with open(os.path.join(label_out_dir, label_name), 'w') as lf:
            lf.write(yolo_line)

        frames_processed += 1
        if frames_processed % 100 == 0:
            print(f"Extracted {frames_processed} labeled frames...")

    cap.release()
    print(f"Done! Successfully generated {frames_processed} YOLO frames for {prefix}.")

if __name__ == "__main__":
    json_path = "dataset/raw_markup/game_1/ball_markup.json"
    video_path = "dataset/raw_videos/game_1.mp4"

    # Fallback check if JSON path is in single directory format
    if not os.path.exists(json_path) and os.path.exists("dataset/raw_markup/game_1.json"):
        json_path = "dataset/raw_markup/game_1.json"

    convert_to_yolo_format(
        json_path=json_path, 
        video_path=video_path,
        image_out_dir="dataset/images/train",
        label_out_dir="dataset/labels/train",
        prefix="game_1"
    )
