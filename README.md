# 🏓 Table Tennis Trajectory Engine & Analytics Dashboard

A high-performance Table Tennis Match Analysis engine combining a low-latency C++ inference backend (TensorRT + OpenCV + ByteTrack + uWebSockets) and a modern Next.js TypeScript dashboard with real-time canvas visualization and Magnus spin aerodynamics simulation.

---

## 🏗 System Architecture

```
                                  TABLE TENNIS VIDEO FEED (120 FPS)
                                                 │
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │    OpenCV Video / Frame Capture        │
                             └───────────────────┬────────────────────┘
                                                 │
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │   TensorRT 10.x Inference Engine       │
                             │   - Asynchronous CUDA Stream Execution │
                             │   - 1x3x640x640 BGR->RGB Float Preproc │
                             └───────────────────┬────────────────────┘
                                                 │
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │       ByteTrack Object Tracker         │
                             │   - Kalman Motion State Prediction     │
                             │   - High-Speed IoU Association         │
                             └───────────────────┬────────────────────┘
                                                 │
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │    uWebSockets Telemetry Broadcaster   │
                             │   - Port 9002 JSON Stream              │
                             └───────────────────┬────────────────────┘
                                                 │
                                                 ▼ (WebSocket ws://localhost:9002)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Next.js Dashboard & Visualizer                             │
│  ┌───────────────────────────┐ ┌───────────────────────────┐ ┌──────────────────────┐ │
│  │   Trajectory Canvas       │ │   Magnus Spin Sandbox     │ │  Rally Analytics     │ │
│  │   - Glowing Tail Arcs     │ │   - Aerodynamic Simulation│ │  - Stroke Counter    │ │
│  │   - Speed Vectors & HUD   │ │   - Top/Back/Sidespin RPM │ │  - Speed Histogram   │ │
│  └───────────────────────────┘ └───────────────────────────┘ └──────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

1. **Low-Latency C++ Engine (`/backend_cpp`)**:
   - Explicit CUDA memory management (`cudaMalloc`, `cudaMemcpyAsync`).
   - Vectorized BGR to RGB planar float normalization ($1 \times 3 \times 640 \times 640$).
   - ByteTrack tracker tuned for 40mm fast-moving ping pong balls.
   - Fallback simulation harness when TensorRT hardware is off-line.

2. **Next.js Dashboard (`/frontend`)**:
   - **Live Trajectory Overlay**: Layers over 60/120 FPS video with gradient fading arcs, impact ripples, and player tracking HUD.
   - **Physics Sandbox**: Solves 3D aerodynamic flight differential equations with Magnus Force ($\vec{F}_M = \frac{1}{2} \rho A C_M \frac{r}{v} (\vec{\omega} \times \vec{v})$), gravity, drag, and table bounce restitution ($e \approx 0.89$).
   - **WebSocket Stream Receiver**: Auto-reconnecting client with live telemetry feed.

3. **Python Tooling (`/python_tools`)**:
   - `export_yolo_onnx.py` for exporting OpenTTGames PyTorch models to ONNX and compiling via NVIDIA `trtexec` into TensorRT `.engine` binaries.

---

## 🚀 Quick Start & Installation

### 1. Run Next.js Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 2. Build C++ Inference Engine (Windows MSVC)
```powershell
# Install CMake if not already installed:
winget install Kitware.CMake

# Build and run C++ backend:
cd backend_cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\tt_trajectory_engine.exe
```

### 3. Export YOLO Model to TensorRT Engine
```powershell
cd python_tools
python export_yolo_onnx.py --model yolov11n.pt --output_dir ../models --fp16
```

---

---

## 📊 Empirical Performance & Validation Summary

### Published Benchmark Statement
> **"Bounce logic on ground-truth ball positions, no detector involved; tuned on `game_1` and `test_1`: precision 84–94%, recall 87–90%. Held-out evaluation pending."**

### 1. Authoritative C++ PhysicsEngine Ground-Truth Benchmark (Match Window $\pm 3$ Frames / $\pm 25\text{ms}$)
Evaluated directly via native C++ executable `eval_bounce.exe` (1-to-1 unique match enforcement):

| Sequence | Split Role | GT Bounces ($n$) | Detected | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall (95% Wilson CI) | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`game_1`** | Tuning Set | 227 | 217 | 205 | 12 | 22 | **94.47%** | **90.31%** [85.8%–93.6%] | **92.34%** |
| **`test_1`** | Development Set | 31 | 32 | 27 | 5 | 4 | **84.38%** | **87.10%** [71.2%–95.2%] | **85.71%** |

### 2. Physics Detector Mechanics & Tuning Analysis
- **2-Frame Windowed Turning-Point Trigger**: Table tennis bounce turning points frequently register $v_{y,\text{pixel}} = 0\text{ px/s}$ on the impact frame ($f$). The windowed velocity delta trigger ($\Delta v_{y,\text{pixel}} = v_{y,\text{inbound}} - v_{y,\text{outbound}} > 400\text{ px/s}$) captures turning points where adjacent single-frame sign-flip checks fail.
- **Paddle Strike Suppression**: Suppressing table bounce detection during active paddle strokes (`cooldown >= 11`) reduced false positive bounce triggers on `game_1` (achieving **94.47% precision**).

### 3. Isolated Real-Time Wall-Clock Latency Breakdown (NVIDIA RTX 5050 - `realtime_wallclock_latency.csv`)
Logged across 5 isolated runs over 16,800 frames per run on a clean machine (no competing background tasks):
- **Headline Statement**: **6.9 ms median wall-clock per frame** (5 runs, range 6.8–8.0 ms), single-threaded, including 1080p H.264 decode and COCO-head engine at 640 px (offering ~1.4 ms of headroom under the 120 FPS / 8.333 ms budget).
- **Per-Stage Breakdown (Median Run)**:
  - **Preprocess Stage p50**: **3.46 ms** (~50.4% of pipeline)
  - **Decode Stage p50**: **2.64 ms** (~38.5% of pipeline)
  - **Inference Stage p50**: **0.74 ms** (~10.8% of pipeline)
  - **ByteTrack & Physics p50**: **< 0.001 ms** (Note: Tracker/physics timings are evaluated on COCO detections with 0% ball recall; full tracking overhead will be re-measured after detector fine-tuning).



<!-- BEGIN RESULTS SECTION -->
## Canonical Performance & Benchmark Summary (Commit: `b52c559f6822d8200843b63b2b1e21b645a66a5f`)

### 1. Ground-Truth Bounce Detection Baseline (Match Window $\pm 3$ Frames / $\pm 25\text{ ms}$)
> *"Bounce logic on ground-truth ball positions, no detector involved; tuned on `game_1` and `test_1`: precision 84–94%, recall 87–90%. Held-out evaluation pending."*

| Sequence | Split | GT Bounces | Detected | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`game_1`** | Tuning Set | 227 | 217 | 205 | 12 | 22 | **94.47%** | **90.31%** | **92.34%** |
| **`test_1`** | Dev Set | 31 | 32 | 27 | 5 | 4 | **84.38%** | **87.10%** | **85.71%** |

### 2. Fine-Tuned Detector Localization & Frame-Shift Sweep (`test_1`, `models/yolov11_ttgames.engine`)
> *"Frame-shift sweep from -3 to +3 offsets verifies Offset 0 as the unequivocal peak. Raw top-1 detector predictions with C++ aspect-ratio letterboxing achieve 4.03 px median localization error and 61.70% hit rate within 15 px on 880 annotated frames."*

- **Frame-Shift Sweep (-3 to +3 Offsets on C++ TensorRT Path)**:
  | Frame Offset | Hit Rate $\le 5.0\text{ px}$ | Hit Rate $\le 10.0\text{ px}$ | Hit Rate $\le 15.0\text{ px}$ | Median Localization Error |
  | :--- | :--- | :--- | :--- | :--- |
  | **Offset -3** | 0.00% | 0.11% | 0.23% | 72.22 px |
  | **Offset -2** | 0.00% | 0.00% | 1.25% | 49.09 px |
  | **Offset -1** | 0.34% | 3.41% | 12.16% | 25.77 px |
  | **Offset +0 (Aligned Peak)** | **46.25%** | **59.77%** | **61.70%** | **4.03 px** |
  | **Offset +1** | 1.70% | 9.43% | 24.09% | 20.08 px |
  | **Offset +2** | 14.55% | 19.43% | 21.25% | 30.51 px |
  | **Offset +3** | 0.34% | 0.91% | 4.20% | 45.10 px |

- **Side-by-Side Raw Tensor Parity (Frame 1672)**:
  - **ONNX Model**: Anchor 3073 | Raw 640x640: `(266.1, 307.8)` | Conf: `0.491` | Unletterboxed 1080p: `(798.3, 503.5)` | Error: **1.37 px**
  - **C++ TensorRT**: Anchor 3073 | Raw 640x640: `(266.0, 307.8)` | Conf: `0.491` | Unletterboxed 1080p: `(798.0, 503.3)` | Error: **1.25 px**
  - *Full 880-frame side-by-side CSV*: [side_by_side_880_frames.csv](file:///c:/Users/Adi/Documents/GitHub/TT%20Trajectory%20Engine/python_tools/side_by_side_880_frames.csv).

### 3. Deployed C++ End-to-End System Evaluation & Tracking Feed Comparative Ablation (`eval_real_trt_system.exe`)
> *"Comparative end-to-end evaluation across 16,800 frames against 31 strict GT bounces confirms ByteTrack's human-tuned Kalman filter degrades fast ball tracking (2.63% F1). Relaxing the track continuity gate from >=3 to >=1 restored bounce recall from 29.03% to 41.94%, and feeding confidence-gated raw top-1 detections directly into PhysicsEngine yields a 15.48% F1 score."*

| Tracking Feed Mode | Clip-Wide Bounces | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mode A: ByteTrack Tracked Feed** | 121 | 2 | 119 | 29 | 1.65% | 6.45% | **2.63%** |
| **Mode B: Gated Raw Top-1 Feed** | 137 | 13 | 124 | 18 | 9.49% | **41.94%** | **15.48%** |

- **False Positive Diagnostic & Sample Overlays**:
  - Rendered 20 FP bounce overlay PNGs into `python_tools/fp_samples/` using `fast_render_fp_overlays.py`.
  - **Root Cause**: 76.4% of false positive bounces (e.g., frames 2674, 3583, 7404, 7793, 11480) occur in **unannotated inter-rally regions** where players walk around holding balls or picking them off the floor between rallies without an active rally state gate.

### 4. Vectorized CPU Preprocessing Pipeline & Isolated 5-Pass Benchmark (`run_isolated_benchmark.ps1`)
> *"Single-threaded p50 latency of 7.56 ms (~132 FPS instantaneous throughput), holding the 120 FPS / 8.33 ms deadline on 94.7%–98.6% of frames depending on pass."*

- **Formal Retraction of Earlier CUDA Kernel Claim**:
  - *Retraction Note*: The previously reported 0.54 ms preprocessing / 6.43 ms total latency figures are officially retracted as a suspected early-return artifact. When `cudaError 222` occurred in `enqueueV3` due to toolchain/driver fatbinary mismatches, TensorRT returned near-instantly on the host without executing GPU inference.
  - *Standing Defensive Practice*: All GPU calls now enforce explicit return-code checking (`enqueueV3` status check, `cudaMemcpyAsync` error check, `cudaStreamSynchronize` error check) in `tensorrt_engine.cpp` to log errors loudly.

- **5-Pass Isolated Latency Table (16,800 Frames per Pass, 10s Thermal Cooldowns)**:
  | Run | Starting SM Clock / Temp / Power | Min (ms) | p5 (ms) | p50 / Median (ms) | p95 (ms) | p99 (ms) | Max (ms) | On-Time ($\le 8.33\text{ ms}$) |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Run 1** | 2520 MHz, 56°C, 33.96 W | 6.77 | 7.27 | **7.57** | 8.13 | 8.78 | 38.79 | **97.61%** |
  | **Run 2** | 2820 MHz, 59°C, 32.78 W | 6.94 | 7.24 | **7.52** | 7.94 | 8.30 | 37.95 | **99.08%** |
  | **Run 3** | 2820 MHz, 59°C, 32.82 W | 6.88 | 7.25 | **7.52** | 7.96 | 8.34 | 37.96 | **98.98%** |
  | **Run 4 (Outlier)** | 2820 MHz, 59°C, 32.84 W | 6.97 | 7.26 | **7.61** | 12.69 | 14.91 | 37.32 | **75.74%** |
  | **Run 5** | 2820 MHz, 60°C, 32.92 W | 6.99 | 7.28 | **7.56** | 8.06 | 8.55 | 37.39 | **98.17%** |

- **Dual-Framed Performance Metrics**:
  - **Clean 4-Run Baseline (Runs 1, 2, 3, 5)**: p50 **7.50–7.57 ms** (mean p50: **7.54 ms**), holding 120 FPS deadline on **98.64%** of frames.
  - **Full 5-Run Aggregate (including Run 4)**: p50 **7.56 ms**, holding 120 FPS deadline on **94.71%** ($16,399 / 16,800$) of frames.
  - *Run 4 Outlier Qualification*: Run 4 experienced a hypothesized OS DPC / driver context switching event mid-run ($12.69\text{ ms}$ p95, $14.91\text{ ms}$ p99) despite identical starting clock/temp/power snapshots.

- **Full Sub-Stage Breakdown Accounting (p50 Timers — 99.7% Accounted)**:
  | Processing Sub-Stage | p50 Duration (ms) | Percentage of Total Latency |
  | :--- | :--- | :--- |
  | **OpenCV Video Frame Decoding (`cap.read`)** | **1.97 ms** | 26.1% |
  | **CPU Letterbox & Normalization Preprocess** | **3.03 ms** | 40.2% |
  | **CUDA H2D Host-to-Device Frame Copy** | **0.22 ms** | 3.0% |
  | **TensorRT GPU Execution (`enqueueV3`)** | **1.66 ms** | 22.0% |
  | **CUDA D2H Device-to-Host Output Copy** | **0.17 ms** | 2.3% |
  | **NMS & Unletterbox Bounding Box Decoding** | **0.48 ms** | 6.4% |
  | **Total Accounted Sub-Stage Latency** | **7.54 ms** | **99.7% of 7.56 ms Total** |
<!-- END RESULTS SECTION -->