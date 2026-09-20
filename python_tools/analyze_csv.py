import pandas as pd
import numpy as np

def analyze_latency_csv(csv_path="realtime_wallclock_latency.csv"):
    df = pd.read_csv(csv_path)
    total_frames = len(df)
    times = df["wall_clock_total_ms"].values

    mean_t = np.mean(times)
    p50_t = np.percentile(times, 50)
    p95_t = np.percentile(times, 95)
    p99_t = np.percentile(times, 99)
    max_t = np.max(times)
    min_t = np.min(times)

    on_time = np.sum(times <= 8.333)
    overruns = total_frames - on_time
    on_time_pct = (on_time / total_frames) * 100.0

    print("=========================================================================")
    print("  EXACT STATISTICAL ANALYSIS OF COMMITTED CSV (realtime_wallclock_latency.csv)")
    print("=========================================================================")
    print(f"Total Frames Evaluated : {total_frames}")
    print(f"Mean Frame Duration    : {mean_t:.3f} ms")
    print(f"Median Latency (p50)   : {p50_t:.3f} ms")
    print(f"95th Percentile (p95)  : {p95_t:.3f} ms")
    print(f"99th Percentile (p99)  : {p99_t:.3f} ms")
    print(f"Max Frame Spike        : {max_t:.3f} ms")
    print(f"Min Frame Time         : {min_t:.3f} ms")
    print(f"Frames On-Time (<=8.33ms): {on_time} / {total_frames} ({on_time_pct:.2f}%)")
    print(f"Deadline Overruns      : {overruns} / {total_frames} ({100.0 - on_time_pct:.2f}%)")
    print("=========================================================================\n")

if __name__ == "__main__":
    analyze_latency_csv()
