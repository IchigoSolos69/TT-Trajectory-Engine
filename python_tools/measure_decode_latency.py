import cv2
import time
import numpy as np
import os

def measure_clip_decode(video_path):
    if not os.path.exists(video_path):
        print(f"Error: video {video_path} not found.")
        return

    print(f"--- Measuring Pure Sequential Video Decode Latency on {video_path} ---")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {w}x{h} @ {fps:.2f} FPS | Total Frames: {total_frames}")

    frame_times = []
    t_start = time.perf_counter()

    read_count = 0
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret or frame is None:
            break

        frame_times.append((t1 - t0) * 1000.0)
        read_count += 1

    t_end = time.perf_counter()
    cap.release()

    total_time_s = t_end - t_start
    decode_fps = read_count / total_time_s if total_time_s > 0 else 0

    frame_times = np.array(frame_times)
    p50 = np.percentile(frame_times, 50)
    p95 = np.percentile(frame_times, 95)
    p99 = np.percentile(frame_times, 99)
    max_t = np.max(frame_times)
    min_t = np.min(frame_times)
    mean_t = np.mean(frame_times)

    print(f"Processed {read_count} frames in {total_time_s:.3f} s ({decode_fps:.1f} decode FPS)")
    print(f"Decode Latency Per Frame:")
    print(f"  Mean: {mean_t:.3f} ms")
    print(f"  p50 : {p50:.3f} ms")
    print(f"  p95 : {p95:.3f} ms")
    print(f"  p99 : {p99:.3f} ms")
    print(f"  Max : {max_t:.3f} ms")
    print(f"  Min : {min_t:.3f} ms")

if __name__ == "__main__":
    measure_clip_decode("python_tools/dataset/raw_videos/test_1.mp4")
