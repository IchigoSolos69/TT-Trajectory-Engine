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
## Canonical Performance & Benchmark Summary (Commit: `c4698cc3dfbcefde42e529fc1a82741b5c544e33`)

### Ground-Truth Bounce Detection Baseline (Match Window $\pm 3$ Frames / $\pm 25\text{ ms}$)
> *"Bounce logic on ground-truth ball positions, no detector involved; tuned on `game_1` and `test_1`: precision 84–94%, recall 87–90%. Held-out evaluation pending."*

| Sequence | Split | GT Bounces | Detected | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`game_1`** | Tuning Set | 227 | 217 | 205 | 12 | 22 | **94.47%** | **90.31%** | **92.34%** |
| **`test_1`** | Dev Set | 31 | 32 | 27 | 5 | 4 | **84.38%** | **87.10%** | **85.71%** |


### Detector Heavy-Tail Localization & Hit Rate (`test_1`, Commit: `dd34a7b3db0866b0c8bb6c5f3535ac989513252f`)
- **GT Ball Frames**: 880 | **Detected GT Frames**: 787 (89.43% Raw Recall)
- **Localization Error Distribution (px)**:
  - **Median (p50)**: **2.85 px** | **Mean**: **4.64 ± 7.65 px**
  - **p90**: **8.05 px** | **p95**: **10.72 px** | **p99**: **52.48 px**
  - **Error > 15 px**: 21 detections (2.67%)
- **Hit Rate among Detections on Annotated Frames**:
  - $\le 5\text{ px}$: **74.59%** (587 / 787)
  - $\le 10\text{ px}$: **94.03%** (740 / 787)
  - $\le 15\text{ px}$: **97.33%** (766 / 787)

### Fine-Tuned FP16 TensorRT Engine Real-Time Wall-Clock Latency & Telemetry
> *"10.27 ms median total frame wall-clock latency across five process-isolated passes (range 10.11–12.40 ms), single-threaded, including 1080p H.264 decode, 640px TensorRT 11.3 FP16 ball engine (7.27 MB), ByteTrack, and physics. Plainly stated: single-threaded wall-clock processing does not hold a 120 FPS (8.33 ms) cadence without hardware decoding and CUDA preprocessing offload."*

- **Fine-Tuned Engine Metadata**:
  - **Engine Path**: `models/yolov11_ttgames.engine` | **File Size**: 7,268,804 bytes (~7.27 MB FP16)
  - **SHA-256 Checksum**: `847d5ed2c02ee27993451c3286c0aedde59e78779a1d0349fda6706bd8c6d42e`
  - **Runtime & Library**: TensorRT Library Version `11.3.0` (DLL `nvinfer_11.dll`)
  - **Tensor Dimensions**: Input `[1, 3, 640, 640]`, Output `[1, 5, 8400]` (1-class ball detector)
- **Per-Run Wall-Clock Latency (p50)**:
  - Run 1: **10.11 ms** | Run 2: **10.27 ms** | Run 3: **10.60 ms** | Run 4: **12.40 ms** | Run 5: **10.23 ms**
  - **Median of Medians**: **10.27 ms** (Spread: 10.11 ms – 12.40 ms)
  - **Stage Breakdown (p50)**: Native TensorRT GPU Inference: **4.45 ms** (43.3%) | Preprocess: **3.49 ms** (34.0%) | Decode: **2.33 ms** (22.7%) | ByteTrack & Physics: **< 0.001 ms**.
- **Latency Percentiles & Cadence (16,800 frames, Clean Process-Isolated Run)**:
  - **p50 (Median)**: **10.11 ms** | **p95**: **11.59 ms** | **p99**: **13.77 ms**
  - **120 Hz Cadence On-Time Percentage ($\le 8.33\text{ ms}$)**: **0.00%** (0 / 16,800 frames within 8.33 ms budget)
- **Hardware Telemetry Summary**:
  - **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU @ 1605–2565 MHz | Temp: 62–71 °C (Median 70 °C) | Power: 23.53 W – 38.70 W | Status: `0x0 (Zero active throttling; 99.8% unthrottled during benchmark window)`.
  - **CPU**: Processor Performance 80% - 100%.
<!-- END RESULTS SECTION -->