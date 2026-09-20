import csv
import numpy as np
import os

def analyze_latency(csv_path="realtime_wallclock_latency.csv"):
    print("=========================================================================")
    print("  Empirical Full-System C++ Latency Benchmark Breakdown")
    print("=========================================================================")

    if not os.path.exists(csv_path):
        print(f"Error: Missing CSV benchmark file at {csv_path}")
        return

    comments = []
    rows = []

    with open(csv_path, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith('#'):
                comments.append(line_str)
            elif line_str:
                rows.append(line_str)

    if comments:
        print("CSV Header Metadata:")
        for c in comments:
            print(f"  {c}")
        print()

    if not rows:
        print("No data rows found in CSV.")
        return

    reader = csv.DictReader(rows)
    all_totals = []
    all_decode = []
    all_prep = []
    all_infer = []
    
    # Run-by-run grouping (each run is 16,800 frames)
    run_frames = []
    current_run = []

    for r in reader:
        try:
            tot = float(r.get('total_ms', r.get('total', 0.0)))
            dec = float(r.get('decode_ms', 0.0))
            prep = float(r.get('preprocess_ms', 0.0))
            inf = float(r.get('inference_ms', 0.0))

            all_totals.append(tot)
            all_decode.append(dec)
            all_prep.append(prep)
            all_infer.append(inf)

            current_run.append(tot)
            if len(current_run) == 16800:
                run_frames.append(current_run)
                current_run = []
        except Exception:
            continue

    if current_run:
        run_frames.append(current_run)

    DEADLINE_MS = 8.333

    print(f"Total Instrumented Frames (Pooled): {len(all_totals)}")
    print("-------------------------------------------------------------------------")
    
    for i, r_totals in enumerate(run_frames):
        p50 = np.percentile(r_totals, 50)
        p95 = np.percentile(r_totals, 95)
        p99 = np.percentile(r_totals, 99)
        on_time = [t for t in r_totals if t <= DEADLINE_MS]
        on_time_pct = (len(on_time) / len(r_totals)) * 100.0
        print(f"Run {i+1} (16,800 frames) -> p50: {p50:5.2f} ms | p95: {p95:5.2f} ms | p99: {p99:5.2f} ms | On-Time (<= 8.33ms): {len(on_time)}/{len(r_totals)} ({on_time_pct:.2f}%)")

    pooled_p50 = np.percentile(all_totals, 50)
    pooled_p95 = np.percentile(all_totals, 95)
    pooled_p99 = np.percentile(all_totals, 99)
    pooled_on_time = [t for t in all_totals if t <= DEADLINE_MS]
    pooled_pct = (len(pooled_on_time) / len(all_totals)) * 100.0

    print("-------------------------------------------------------------------------")
    print(f"POOLED OVERALL ({len(all_totals)} frames):")
    print(f"  p50 (Median)   : {pooled_p50:.2f} ms")
    print(f"  p95            : {pooled_p95:.2f} ms")
    print(f"  p99            : {pooled_p99:.2f} ms")
    print(f"  On-Time (<=8.33ms) : {len(pooled_on_time)} / {len(all_totals)} ({pooled_pct:.2f}%)")
    
    mean_prep = np.mean(all_prep)
    mean_dec = np.mean(all_decode)
    mean_inf = np.mean(all_infer)
    mean_tot = np.mean(all_totals)
    
    print("\nStage Breakdown (Mean & Share of Total Frame Time):")
    print(f"  Preprocess (CPU image resize/norm): {mean_prep:.2f} ms ({mean_prep/mean_tot*100.0:.1f}%)")
    print(f"  Video Decode (FFmpeg h264 CPU)    : {mean_dec:.2f} ms ({mean_dec/mean_tot*100.0:.1f}%)")
    print(f"  TensorRT FP16 GPU Inference       : {mean_inf:.2f} ms ({mean_inf/mean_tot*100.0:.1f}%)")
    print(f"  Combined CPU Decode + Preprocess  : {mean_dec + mean_prep:.2f} ms ({(mean_dec+mean_prep)/mean_tot*100.0:.1f}% of frame time)")
    print("=========================================================================\n")

if __name__ == "__main__":
    csv_path = "realtime_wallclock_latency.csv"
    if not os.path.exists(csv_path):
        csv_path = "backend_cpp/build/latency_benchmark.csv"
    analyze_latency(csv_path)
