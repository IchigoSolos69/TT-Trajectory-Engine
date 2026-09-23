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
## Canonical Performance & Benchmark Summary (Commit: `3342fd599729a17c7804e1de7252b01df6b2740e`)

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

### 4. Fused CUDA GPU Preprocessing Pipeline & Isolated 5-Pass Benchmark (`run_isolated_benchmark.ps1`)
> *"6.43 ms median total frame wall-clock latency across five process-isolated passes (range 6.18–7.65 ms p95), achieving a 98.97% on-time real-time cadence (16,627 / 16,800 frames) within the 120 FPS (8.33 ms) budget on an RTX 5050 Laptop GPU."*

- **Isolated Sub-Stage Breakdown & Overhead Analysis (6.43 ms Median Total Frame Wall-Clock)**:
  | Processing Sub-Stage | Baseline CPU Preprocess | Fused CUDA GPU Preprocess Pipeline | Percentage of Frame Window |
  | :--- | :--- | :--- | :--- |
  | **OpenCV CPU Aspect Letterboxing Preprocess** | 3.50 ms | **0.54 ms** (CUDA H2D + Preprocess) | 8.40% |
  | **OpenCV CPU Video Frame Decoding** | 2.30 ms | **2.30 ms** | 35.77% |
  | **TensorRT GPU Execution (`enqueueV3`)** | 2.11 ms | **1.81 ms** | 28.15% |
  | **CPU NMS & Postprocess Bounding Box Decoding** | 1.08 ms | **0.54 ms** | 8.40% |
  | **CUDA D2H Device-to-Host Output Copy** | 0.19 ms | **0.84 ms** | 13.06% |
  | **ByteTrack & Physics Update** | 0.02 ms | **0.02 ms** | 0.31% |
  | **Loop & Calibration Overhead** | 1.77 ms | **0.38 ms** | 5.91% |
  | **Total Frame Wall-Clock Latency (p50)** | **11.20 ms** | **6.43 ms** (98.97% $\le 8.33\text{ ms}$) | **100.00%** |

- **GPU Clock Ramp Proof (`test_gpu_burst_clocks.exe` + `nvidia-smi`)**:
  - Isolated Burst GPU Compute (10ms CPU Idle Window): **2.11 ms** (GPU SM Clock downclocks to **1897 MHz**).
  - 100 Consecutive Back-to-Back Warm Inferences: **1.64 ms** (GPU SM Clock boosts to **2812 MHz**).
  - Empirical Idle Window Penalty: **+0.47 ms** (32.5% frequency drop during single-threaded CPU idle windows between frames).
<!-- END RESULTS SECTION -->