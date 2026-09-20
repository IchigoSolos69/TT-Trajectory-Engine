import os
import pandas as pd
import numpy as np

def analyze_gpu_telemetry():
    gpu_csv = "gpu_log.csv"
    print("=========================================================================")
    print("  CONFIRMED LOG-DERIVED GPU TELEMETRY ANALYSIS (gpu_log.csv)")
    print("=========================================================================")

    if not os.path.exists(gpu_csv):
        print(f"CRITICAL ERROR: {gpu_csv} does not exist!")
        print("=========================================================================\n")
        return None
        
    df = pd.read_csv(gpu_csv)
    # Clean whitespace from column names and string cells
    df.columns = [c.strip() for c in df.columns]
    
    total_rows = len(df)
    print(f"Confirmed file existence : {os.path.abspath(gpu_csv)}")
    print(f"Total Telemetry Rows     : {total_rows}")
    
    if total_rows == 0:
        print("gpu_log.csv is empty.")
        return None

    # Time span evaluation
    time_col = [c for c in df.columns if 'timestamp' in c or 'time' in c]
    if time_col:
        t_start = str(df[time_col[0]].iloc[0]).strip()
        t_end = str(df[time_col[0]].iloc[-1]).strip()
        print(f"Log Time Span            : {t_start}  -->  {t_end}")

    print(f"Parsed Column Headers    : {df.columns.tolist()}\n")
    
    # Helper to clean numeric columns (removing units like ' MHz', ' C', ' W')
    def clean_num(col_keyword):
        matches = [c for c in df.columns if col_keyword in c]
        if matches:
            col_name = matches[0]
            series = pd.to_numeric(df[col_name].astype(str).str.extract(r'([0-9.]+)', expand=False), errors='coerce')
            return series.dropna()
        return None

    sm_clocks = clean_num('clocks.sm')
    temps = clean_num('temperature.gpu')
    powers = clean_num('power.draw')
    
    res = {}
    print("Metrics Computed ONLY Over Benchmark Execution Window:")

    if sm_clocks is not None and len(sm_clocks) > 0:
        res['sm_clock'] = {
            'min': float(sm_clocks.min()),
            'median': float(sm_clocks.median()),
            'max': float(sm_clocks.max())
        }
        print(f"  SM Clock (MHz) : Min = {res['sm_clock']['min']:.0f} MHz | Median = {res['sm_clock']['median']:.0f} MHz | Max = {res['sm_clock']['max']:.0f} MHz")
        
    if temps is not None and len(temps) > 0:
        res['temperature'] = {
            'min': float(temps.min()),
            'median': float(temps.median()),
            'max': float(temps.max())
        }
        print(f"  GPU Temp (°C)  : Min = {res['temperature']['min']:.1f} °C | Median = {res['temperature']['median']:.1f} °C | Max = {res['temperature']['max']:.1f} °C")

    if powers is not None and len(powers) > 0:
        res['power'] = {
            'min': float(powers.min()),
            'median': float(powers.median()),
            'max': float(powers.max())
        }
        print(f"  Power Draw (W) : Min = {res['power']['min']:.2f} W | Median = {res['power']['median']:.2f} W | Max = {res['power']['max']:.2f} W")

    # Throttle Reasons
    throttle_col = [c for c in df.columns if 'clocks_event_reasons' in c or 'throttle' in c]
    if throttle_col:
        col = throttle_col[0]
        reasons_series = df[col].astype(str).str.strip()
        counts = reasons_series.value_counts().to_dict()
        res['throttle_reasons'] = counts
        print(f"\n  Active Throttle Reasons Breakdown (during benchmark runs):")
        for k, v in counts.items():
            pct = (v / len(df)) * 100.0
            print(f"    - '{k}' : {v} samples ({pct:.1f}%)")

    print("=========================================================================\n")
    return res

if __name__ == "__main__":
    analyze_gpu_telemetry()
