# Table Tennis Trajectory Engine

A local Windows application for detecting table-tennis balls, analyzing recordings
and live camera footage, and inspecting synchronized footage, 2D and 3D views.
It combines a C++/OpenCV/TensorRT backend, an Electron desktop shell, a
Next.js/TypeScript interface and Python training/evaluation tools.

**Status:** the application runs locally, but a validated high-precision live
sports-analysis release remains unfinished. Detection, physical measurements and
experimental intelligence have separate acceptance gates. Successful builds or
convincing visualizations do not establish analytical accuracy.

Documentation updated **25 October 2026**. Results retain their original dates
and scope; this update is not a new training run or benchmark.
The [requirement ledger](REQUIREMENTS_LEDGER.md) tracks outstanding acceptance
and the final whole-project audit.

## Contents

- [Run the application](#run-the-application)
- [Features and status](#features-and-status)
- [Architecture and measurement conventions](#architecture-and-measurement-conventions)
- [Models](#models)
- [Build and develop](#build-and-develop)
- [Training data and review](#training-data-and-review)
- [YOLO improvement and hybrid roadmap](#yolo-improvement-and-hybrid-roadmap)
- [Evaluation and performance](#evaluation-and-performance)
- [Verification](#verification)
- [Data and repository layout](#data-and-repository-layout)
- [Troubleshooting](#troubleshooting)
- [Documentation and research](#documentation-and-research)

## Run the application

### Existing Windows release

Double-click **launch_desktop.bat**, or run from the repository root:

~~~powershell
.\launch_desktop.bat
~~~

**launch_engine.bat** delegates to the same desktop launcher. It prefers the
retained **paired training v2** package, then **paired training v1**, and otherwise
uses the installed source Electron runtime. After building another package, open
its new EXE directly: the launcher does not automatically choose an arbitrarily
named release folder.

Keep the complete portable release folder together. Normal footage analysis needs
no external browser, Node.js, Python installation or internet connection. It
requires **Windows x64 and an NVIDIA GPU/driver compatible with the serialized
TensorRT engines**. The tested machine uses an RTX 5050 Laptop GPU; compatibility
with other GPUs is not established. The application is unsigned and portable,
rather than an installer. There is no CPU inference fallback.

**A clean Git clone is not a ready-to-run model distribution.** Engines, weights,
datasets, native binaries and packaged releases are Git-ignored. Supply compatible
model resources and build, or use an existing complete local release.

Optional **full-original absence-interval preparation** is training work and
requires an external Python environment with OpenCV and NumPy. This runtime is
not bundled or automatically downloaded.

### Select footage or a camera

1. **Open a game / Open recording…** selects a local MOV, MP4, AVI, MKV, WebM or
   M4V file. Ctrl+O also opens the native recording picker. Actual codec support
   depends on OpenCV/FFmpeg.
2. **Start a live session** enters camera standby. Select the camera and press
   **AUTHORIZE WEBCAM** before capture begins. Stop/end session releases it.
3. **Ball model** chooses default YOLO or explicitly experimental BlurBall.
   Switching restarts the owned backend with a separate analysis identity.
4. **Training review** opens saved captures without starting a camera.
5. Press **H** for cinematic view. **Local data** or **Help → Open local data and
   logs** locates saved work.

Simulation uses only recording footage; live uses only the authorized camera.
Live mode rejects files, and failures do not substitute another source or
synthetic detections. Authorization loss, source changes and dashboard
disconnects revoke capture. Normal close stops application-owned services.

The desktop selects available **loopback HTTP/WebSocket ports** instead of
requiring 3000/9002. It preserves unrelated services. The Electron renderer is
sandboxed with context isolation and restricted native IPC; OpenCV acquires the
camera only after authorization.

## Features and status

| Area | Available behavior | Limit |
|---|---|---|
| Desktop | Native picker, source/model controls, owned services, logs and persistent drafts | Manual OS-dialog acceptance, forced-termination cleanup and prolonged soak remain incomplete |
| Footage overlays | Separate ball/person/table detectors, labels, image-space trails and velocity vectors | Scores are not calibrated probabilities; venue false positives remain |
| Recording analysis | Independent C++ worker saves indexed per-frame results and reuses compatible caches | Full-length/disk-fault acceptance and native VFR/MOV timing remain unfinished |
| Playback | Pause, restart, seek, frame stepping and 0.25×–4× presentation | Seeking beyond analyzed coverage waits; preview rate differs from analysis coverage |
| 2D/3D | Selected-frame table-plane projection, regulation table/net, perspective/top/side views | Frame alignment is not physical 3D reconstruction; airborne height is unavailable |
| Strike velocity | Latest strike and up to 100 persistent events with source frames/timestamps | Planar speed/contact estimates need labeled real-world validation |
| Spin | Conditional surface-feature optical-flow rotation, otherwise UNRESOLVED | Camera-axis rotation only; real-camera RPM accuracy and full spin remain unvalidated |
| Table geometry | Table region and contour refinement using known dimensions | Blue/green refinement may fail on grey, occluded or unusual tables |
| Physics sandbox | Gravity, drag, Magnus effect, repeated bounces, net interaction and floor termination | User-controlled simulation, separate from measured footage |
| Training | Capture guidance, sparse images, consecutive bursts, ball-only review, streaks and annotation history | Suggestions need explicit review; sampled captures are not continuous match video |
| Intelligence | Experimental covariance/states, persistent corners, motion gates, event candidates and hard-example capture | Validation fails or is incomplete; normal launcher does not enable this path |

### Complete recording analysis

Analysis and presentation are separate. A C++ worker processes source frames
sequentially and persists results under **artifacts/analyses/**. Playback speed,
pause and seek do not authorize analytical frame skipping. The analyzed portion
can play while processing continues; seeking ahead waits for coverage. Decode
or analysis failures must appear as partial/error coverage.

Cache identity includes source, models, runtime and relevant settings. Analysis
identity is separate from presentation generation, preventing stale deliveries
after seek/source changes. JPEG decoding precedes publication of matching
footage and telemetry. Preview throttling does not discard recording results.

Historical accelerated replay sampled frames; indexed analysis superseded that
behavior. Older measurements in [VERIFICATION.md](VERIFICATION.md) retain their
original scope. **Native iPhone/VFR MOV frame-count, seeking and true-PTS timing
acceptance remains pending**; nominal FPS does not establish exact physical speed.

## Architecture and measurement conventions

~~~mermaid
flowchart TD
    Shell[Electron source and model controls] --> UI[Next.js / React renderer]
    File[Selected recording] --> Worker[C++ sequential analysis worker]
    Camera[Authorized camera] --> Live[C++ live acquisition]
    Worker --> Vision[YOLO or experimental BlurBall + scene detector]
    Live --> Vision
    Vision --> State[C++ observations, tracking, geometry and analytics]
    State --> Journal[Indexed recording results]
    Journal --> Replay[Selected-frame replay]
    Replay --> Transport[Loopback WebSocket: footage + telemetry]
    State -->|Live preview| Transport
    Transport --> UI
    UI --> Views[Footage, 2D radar, 3D and event views]
    UI --> Review[Explicit capture and annotation review]
    Review --> Data[Audited datasets]
    Data --> Experiments[Isolated Python training and native evaluation]
    Experiments --> Gate[Release gates and manual model promotion]
~~~

- **Native runtime:** C++17, OpenCV, TensorRT, CUDA streams and explicit device
  buffers, with a custom WebSocket server carrying JPEG previews and JSON.
  Current YOLO preprocessing uses CPU OpenCV letterboxing/RGB normalization and
  CUDA transfers; this is not a zero-copy pipeline.
- **Tracking:** local ByteTrack-style high/low-score association, IoU and
  constant-velocity prediction. It is not the complete upstream ByteTrack/Kalman
  implementation. Experimental covariance tracking is separate.
- **Interface:** Electron, Next.js 14, React 18, TypeScript, Tailwind CSS, canvas,
  React Three Fiber/Three.js and animation libraries.
- **Analytics ownership:** C++ supplies observations, recording state and
  experimental trajectory/calibration/event data. The renderer displays results
  and submits controls/annotations. Remaining frontend analytical fallbacks are
  tracked in the ledger. The standalone physics sandbox currently uses a
  TypeScript integrator; a shared C++ counterfactual integrator is planned.
- **Contract:** [telemetry types](frontend/src/types/telemetry.ts) describe source,
  frame/time, analysis/presentation identity, observations, model evidence,
  calibration and timings. Full versioned provenance/nullability migration is
  still incomplete.

### Position, velocity, spin and uncertainty

The coordinate basis uses a **1.525 m × 2.740 m** table, **0.760 m** table height
and **0.1525 m** net. Homography maps pixels onto the table plane. An airborne
ball's projected point is not necessarily its true horizontal world location.
Both spatial views show the same projection; unsupported height is unavailable.

Strike speed is derived planar motion, dependent on calibration, detections and
timing. Direction reversals can be false contacts. A person box or table-end
association does not establish player identity or a named stroke.

Observed spin requires a ball at least 24 pixels across, six surface features and
consistent robust fits. It estimates camera-axis rotation, not a complete
topspin/backspin vector. A synthetic 16.667 RPM case measured 16.466 RPM;
real-camera spin accuracy remains unmeasured. RPM is not generated from speed
or copied from sandbox settings.

Experimental trajectory states are **MEASURED, PREDICTED, REACQUIRED, REJECTED and
LOST**. Measured means an observation updated the filter, not ground truth.
Covariance is propagated into planar ellipses/corridors. Nominal 95% contours
have not passed empirical coverage acceptance; they are not a calibrated 95%
correctness guarantee. Volumetric tubes require supported 3D covariance.

## Models

| Component | Role and tensor contract | Status |
|---|---|---|
| YOLO11 ball | One RGB frame: [1,3,640,640] → [1,5,8400]; ball only | Default; FP16 internals, FP32 I/O |
| Scene YOLO | One RGB frame: [1,3,640,640] → [1,6,8400]; person/table | Separate model; FP16 internals, FP32 I/O |
| BlurBall | Three RGB frames: [1,9,288,512] → [1,3,288,512]; temporal heatmaps | Published pretrained FP32 engine; opt-in experimental |
| Five-frame ROI recovery | Temporal crop heatmap/presence candidate | Trained/exported offline; precision/absence gates fail |
| Hybrid selector v2 | Select original YOLO/BlurBall position or abstain | Native research evaluator; not a released hybrid |
| Temporal YOLO-feature student | YOLO11n encoder + three-frame fusion + heatmap/visibility heads | Optimizer prototype; no accepted fitted student/native integration |

BlurBall recording analysis can use **two frames of lookahead**. Live uses only
t−2/t−1/t and the latest-frame output, with two warmup frames and history resets
after source changes/gaps. Lookahead results are not live results. YOLO and
BlurBall are both neural networks; temporal context, spatial features,
supervision and localization are the useful architectural differences.

The separate five-frame ROI recovery experiment expires after four detector
misses or 50 ms. Inferred points cannot extend their own detector-anchor lifetime.
Its evaluated precision fails acceptance, so this is not a default recovery path.

The desktop packager requires these local resources:

~~~text
models/
  yolov11_ttgames.engine
  yolov11_ttgames.manifest.json
  scene.engine
  scene.manifest.json
  blurball_fp32.engine
  blurball.manifest.json
  BlurBall-LICENSE.txt
~~~

Model/release manifests record hashes and provenance. TensorRT engines are
hardware/runtime-specific, rather than portable checkpoints. Shape-aware YOLO
export/native decoding supports inspected 640/960 and P3/P2 contracts; this does
not establish a trained P2 model or accepted 960 deployment. One 960 parity case
still exceeds the one-pixel p95 gate. Independent Python/native BlurBall
preprocessing retains an OpenCV-version discrepancy despite tight identical-input
inference parity.

BlurBall's upstream source notice is MIT. Separate checkpoint redistribution
terms were not supplied in its metadata; verify those before publicly distributing
weights or engines. See [implementation](BLURBALL_IMPLEMENTATION.md) and the
[authors' repository](https://github.com/cogsys-tuebingen/BlurBall).

## Build and develop

### Requirements

- Windows x64, a compatible NVIDIA GPU/driver and storage for footage, extracted
  frames, builds, packages and checkpoints.
- Visual Studio 2022 C++ build tools, CMake ≥3.18, CUDA, TensorRT and OpenCV with
  the required video decoder libraries.
- Node.js/npm for frontend/Electron development; use committed lockfiles. Service
  and integration scripts require a Node runtime with built-in global WebSocket;
  the packaged application supplies its own runtime.
- A compatible Python ML environment for training/export/evaluation. Tools use
  PyTorch, Ultralytics, OpenCV, NumPy, Pillow and ONNX-related packages as
  applicable. There is no single committed Python environment lock. Experiment
  plans bind installed versions and configuration hashes.

Current [CMake](backend_cpp/CMakeLists.txt) and packaging contain machine-specific
paths: TensorRT at **C:/TensorRT** with **nvinfer_11.lib**, OpenCV under
**C:/opencv/build**, CUDA runtime search paths for **v13.4**, and CUDA architecture
settings **89/75**. Inspect/adapt these to your toolchain. Fresh-machine,
cross-platform and arbitrary-GPU builds are not established.

### Desktop build

From the repository root, after supplying compatible models:

~~~powershell
npm.cmd --prefix frontend ci
cmake -S backend_cpp -B backend_cpp/build -DOpenCV_DIR=C:/opencv/build
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_desktop.ps1
& '.\desktop\dist\TT Trajectory Engine\TT Trajectory Engine.exe'
~~~

Use a Visual Studio developer environment and appropriate generator/CUDA setup.
The script assumes the native build is configured, builds
**tt_intelligence_candidate**, builds the **.next-desktop** standalone frontend,
installs desktop dependencies and packages Electron, three engines and native
DLLs. The target name does not enable experimental intelligence.

Packaging refuses existing output directories. Close the old app and retain/move
its release first, or set **TT_DESKTOP_OUTPUT_DIR** to a new direct child of
**desktop/dist**. Launch the new EXE explicitly. The release manifest records
backend/model hashes, frontend identity and runtime dependencies. Source Electron
can run through **npm.cmd --prefix desktop start** after resources are built.

### Developer browser and native CLI

The browser interface remains a developer option:

~~~powershell
npm.cmd --prefix frontend run dev
~~~

Its HTTP default is **127.0.0.1:3000**, with non-desktop WebSocket default
**ws://127.0.0.1:9002**. Start the native backend in a separate terminal:

~~~powershell
cmake --build backend_cpp/build --config Release --target tt_trajectory_engine -j 4
.\backend_cpp\build\Release\tt_trajectory_engine.exe --sim --video 'C:\path\game.mov'
~~~

Alternative invocations, **one backend at a time**:

~~~powershell
.\backend_cpp\build\Release\tt_trajectory_engine.exe --live
.\backend_cpp\build\Release\tt_trajectory_engine.exe --sim --video 'C:\path\game.mov' --ball-backend blurball
~~~

| Option | Meaning |
|---|---|
| --sim / --live | Recording or camera; conflicting/repeated source modes fail |
| --video PATH | Recording only; rejected in live mode |
| --ball-backend yolo or blurball | Default or experimental detector |
| Positional PATH.engine | Explicit ball engine, not a --model flag |
| --scene-engine PATH | Separate scene engine |
| --confidence VALUE | Strictly between 0 and 1; default 0.25 |
| --port NUMBER | Native WebSocket port, 1024–65535; default 9002 |
| --data-root PATH | Explicit local data root |
| --max-frames NUMBER | Positive analysis limit for scoped runs |
| --experimental-intelligence | Opt into unfinished trajectory/calibration/event research |

Camera selection/authorization uses the UI, not a --camera option. Prefer an
explicit video path; legacy simulation defaults only try local game_1 locations
and fail without footage. A custom backend port needs a matching client; it does
not change the browser default. The older
[PowerShell service launcher](tools/launch_services.ps1) is a developer tool,
with identity/health/restart checks, rather than the desktop entry.

## Training data and review

### Recording useful material

The GUI includes **Recording checklist, gameplay steps & warnings**. Use stable
framing, the full table/net and both players, focus/exposure that preserve the
ball, clear near/far examples and ordinary rallies. Include bright backgrounds,
compression, blur, occlusion and difficult placements. Dimming footage or adding
a green tint has not established an accuracy improvement. The app does not
certify focus/exposure or configure true sensor FPS.

- **Record training clip:** unannotated images at up to five samples/second.
- **Record temporal bursts:** five consecutive captured frames once per second.
- **Capture difficult examples:** experimental heuristics using weak observations,
  disagreement, ROI quality and calibration problems.

Explicit captures have a 60-second limit and bounded disk queues with saved,
dropped and error reporting. These are sampled images, not continuous match
video. Preserve original full recordings for event timing, complete rallies and
eventual highlights. Do not pad incomplete bursts with repeated frames.

For negatives, physically remove the ball while players move normally. A hidden
or uncertain ball is not confirmed absence. Use iPhone 1080p/60 where available
and the PC camera's genuine supported rate; upsampling does not create 60 FPS.
Preserve original orientation, timing and untouched source files.

### Assisted review

In **Training review**, use **Ball only · faster**, correct a ball box, choose
visibility and **Save & next**. Enter saves/advances; arrow keys navigate.
Adjacent boxes/model outputs are editable suggestions, never automatically
approved truth. Blur-streak endpoints are separate targets from box extents.
Ball-only saves preserve scene/geometry labels; unknown classes remain unknown.

Optional full-scene review includes people/table boxes, table/net landmarks,
blur/visibility and event timestamps. Drafts persist across navigation; desktop
drafts survive app/port restarts. Revision checks and immutable annotation
history prevent silent overwrites.

**Full original no-ball interval** prepares all frames in a chosen interval and
requires explicit original-interval/no-hidden-ball confirmation. Sparse
negatives cannot establish two minutes of absence. Video Project 6 contains real
balls and must not be declared negative wholesale.

### Session splits and readiness

Keep whole gameplay sessions together, including simultaneous iPhone/PC footage.
Folder names do not establish device or test independence. The original ten
iPhone clips and both Video Project 6 review batches belong to the same existing
training session. World Cup additions are mixed-match compilations: audit overlap
and lineage before match training, and do not substitute them for PC-camera or
independent venue evaluation.

The latest strict snapshot, **12 October 2026**, reports:

| Scope | Eligible visible frames | Verified absent frames |
|---|---:|---:|
| Existing iPhone training | 128 | 293 |
| Actual PC-camera training | 0 | 0 |
| Independent validation, either device | 0 | 0 |

Fourteen captures contain 750 indexed images. Seven conflicting unique records
and two unknown-visibility labels are excluded; six identical reviewed copies
are deduplicated. There are zero new-A positive windows and **17 data/export
blockers**. Fitting remains disabled, and sealed labels were not consumed.
Earlier structural counts of 130 positives use a different audit. Registration
preserved all 868 files in the two confirmed capture trees.

Collect paired sessions **A: training**, **B: validation**, **C: sealed test**.
Training minimums are 2,000 reviewed visible frames, 1,000 verified absences and
500 continuity-audited positive three-frame windows from A, with both devices.
For B, each device needs 1,000 visible references, ten complete reviewed rallies
and ≥120 seconds of original reviewed absence. Collect corresponding independent
C coverage and keep it sealed until final model/settings freeze. These minimums
do not guarantee accuracy.

See [capture/review instructions](TRAINING_LIVE.md),
[next actions](TRAINING_NEXT_ACTIONS_2026_10_12.md),
[focused review queue](TRAINING_REVIEW_QUEUE_2026_10_12.md) and
[strict audit](artifacts/yolo_paired_training_plan_2026_10_12_v2.json).

## YOLO improvement and hybrid roadmap

Improve standalone YOLO before training a hybrid. The implemented runner compares
five 12-epoch pilots: reviewed data at 640, unfrozen backbone, 960 input, reviewed
extent targets and camera-oriented augmentation. Shared controls include AdamW,
learning rate 0.0001, seed 42, AMP disabled and common batch 16/8/4 within the
approximately 7 GiB allocation budget. Sampling is 50% match, 25% iPhone and 25%
PC-camera, with separate domain validation. Final selected configurations use
seeds 42/43/44, up to 60 epochs and patience 12.

In a compatible ML environment, use **new output paths**:

~~~powershell
python -B python_tools/prepare_yolo_paired_training.py artifacts/paired_yolo_EXPORT
python -B python_tools/yolo_ball_experiments.py plan artifacts/paired_yolo_PLAN.json --prepared artifacts/paired_yolo_EXPORT --batch 16
~~~

Only after the plan passes readiness:

~~~powershell
python -B python_tools/yolo_ball_experiments.py train artifacts/paired_yolo_PLAN.json data_only640 artifacts/paired_yolo_A
~~~

Preparation/planning does not train or deploy. Missing data, changed inputs/runtime,
conflicting duplicates and invalid splits block fitting. Sealed C labels stay
unopened. Full recipes, augmentation and memory controls are in the
[experiment workflow](python_tools/YOLO_BALL_EXPERIMENTS.md).

After YOLO passes validation, compare it with causal BlurBall on identical frames.
A proposed learned selector chooses **an original YOLO position, an original
BlurBall position, or neither**. It must preserve standalone quality and show
complementary benefit. Averaging raw scores or blending conflicting positions is
not the plan. A supervised temporal student and separate distillation experiment
follow only after validated teachers.

Export candidates separately:

~~~powershell
python python_tools/export_yolo_onnx.py --model 'C:\path\best.pt' --output_dir artifacts/candidate_export --fp16
~~~

The exporter requires an actual checkpoint and refuses existing outputs. Use
**--kind scene** for the separate scene model. Native shape/preprocessing/
coordinate/reduced-precision parity must pass before manual promotion. Keep
original engines, manifests and rollback evidence.

### Release requirements — targets, not achieved results

| Area | Gate |
|---|---|
| Detection | Precision ≥98%, recall ≥95%, separately for iPhone and actual PC camera |
| Localization | Strict <4 px at 1080p-equivalent resolution; also report 8/15 px |
| Confirmed absence | False detections on ≤1% of verified absent frames and ≤1 false-track episode/min |
| Difficult subsets | Separate bright/small-ball/blur/partial-occlusion results; ≥100 visible labels per supported subset |
| Recovery | Accepted recovery precision ≥95%, bounded causal lifetime and no absence regression |
| Uncertainty | Nominal 95% regions achieve 92–98% coverage, reporting error and region size |
| Geometry/events | Independent landmark median ≤3 px/p95 ≤6 px at 1080p equivalent; supported event precision ≥95%, recall ≥80%, one-to-one 50 ms matching |
| Live | Sustained 60 FPS analysis where the camera supplies it; p95 capture-to-display ≤100 ms; 30-minute soak |
| Promotion | No baseline precision/recall/absence regression, native parity, frozen independent testing and rollback |

Predicted points are not correct detections. Reports include counts, misses,
wrong locations, abstentions and uncertainty intervals. Threshold selection on
test footage or reusing one session cannot establish independence. **120 FPS is
a later optimization target**, not a current end-to-end guarantee.

## Evaluation and performance

### Detector results

These use **different footage/tolerances** and cannot be pooled into one accuracy
number. Unknown annotations are excluded, not assumed absent.

| Evaluation | Precision | Recall | Scope |
|---|---:|---:|---|
| Deployed YOLO, test_1 | 87.6% | 86.5% | 15 source-pixel matching; 880 visible/5 absent references |
| Deployed YOLO, test_2 | 90.8% | 84.1% | 15 source-pixel matching; 773 visible/6 absent references |
| YOLO, official 272-frame development clip | 4.41% | 4.44% | Strict <4 source px; 270 visible/2 absent |
| Causal BlurBall, same clip | 97.69% | 94.07% | Past/current only; includes warmup misses |
| Lookahead BlurBall, same clip | 98.48% | 96.30% | Recording-only lookahead |
| Hybrid v1, same development comparison | 100% | 1.85% | Only five correct accepted detections; severe abstention |
| Experimental hybrid v2, same comparison | 97.69% | 94.07% | Matches BlurBall without complementary recall gain |

On **284 reviewed absent venue frames**, false detections were **42 YOLO /
71 causal BlurBall / 75 hybrid v2**. This correlated, partly hard-mined subset is
not a whole-video false-positive rate, but the hybrid fails non-regression.
No hybrid was promoted. Official checkpoint training lineage is unknown;
test_1/test_2 and the comparison clip are exposed development/regression evidence,
not newly blind tests.

Scene detection on 779 test_2 frames achieved 99.3% person precision/98.5% recall
at score 0.4 and IoU 0.5. Table boxes matched 779/779; refined mask overlap averaged
79.7%. Box detection is not geometric pose accuracy, and one sequence does not
establish venue generalization. People labels include officials.

Sources: [deployed evaluation](VERIFICATION.md),
[BlurBall evaluation](BLURBALL_EVALUATION.md),
[hybrid v2 report](HYBRID_V2_IMPLEMENTATION_REPORT.md).

### Physics and intelligence evidence

- The sandbox passed a 36-case launch sweep. Halving its timestep moved the
  second landing by 8.2 mm; an analytical restitution check differed by less than
  0.0015 m/s. This tests simulation, not video contacts.
- Initial measured-point nominal-95% uncertainty coverage was 56.26%; a later
  noise-selected configuration reached 89.12%, still below the gate and with a
  different state population. Defaults were not promoted.
- Matched five-frame recovery v2 reached 83.33% accepted-recovery precision,
  below 95%, and worsened confirmed-absence behavior.
- Causal quadratic forecasting reduced mean error but worsened combined RMSE:
  10.863 px versus constant velocity's 10.245 px. Abstentions, outlier regressions
  and missing independent physical references prevent promotion.

See [intelligence experiments](INTELLIGENCE_ROADMAP.md) and
[forecast results](BLURBALL_IMPLEMENTATION.md).

### Offline versus live performance

Historical offline decode/inference benchmarks exclude parts of the present
multi-model desktop workload. They cannot establish live capture-to-display
latency. An older two-model file-playback session measured about 70.5 analyzed FPS
and 23.4 preview FPS, not live 60/120 FPS acceptance.

A separate historical camera smoke acquired **21 actual 640×480 frames** and
released capture. It verifies acquisition/release only. Representative live
gameplay accuracy, current capture-to-display timing and a 30-minute soak remain
unverified.

<!-- BEGIN RESULTS SECTION -->
### Archived benchmark snapshot

These frozen numbers are historical, not current release acceptance.
[python_tools/results.json](python_tools/results.json) records the original
engine/configuration and commit.

| Historical test | Result | Qualification |
|---|---|---|
| Bounce logic on GT game_1 positions | Precision 94.47%, recall 90.31%; TP/FP/FN 205/12/22 | Tuning; no detector; ±3 frames/±25 ms matching |
| Bounce logic on GT test_1 positions | Precision 84.38%, recall 87.10%; TP/FP/FN 27/5/4 | Development; no detector; same matching |
| Old ball-engine offline pipeline, five ×16,800-frame passes | Median of pass medians 8.08 ms; range 7.93–8.09 ms | RTX 5050; decode/preprocess/inference/tracking/physics; excludes present scene/preview/transport/render |
| One recorded pass in that configuration | p50 8.04 ms, p95 10.02 ms, p99 11.70 ms; 66.82% within 8.33 ms | Does not establish sustained 120 FPS, current model timing or live latency |

**python_tools/make_results.py** is a legacy **hard-coded historical generator**,
not a benchmark runner. It rewrites this marked section and results.json with
frozen numbers; do not use it to claim fresh measurements.
<!-- END RESULTS SECTION -->

## Verification

These are reproduction commands, not checks rerun for this README update.
Required dependencies, models and built executables must exist first.

~~~powershell
# Desktop process/security/draft checks; no camera authorization
node --test desktop/tests/*.test.cjs

# Frontend data-flow and physics checks
node frontend/tests/telemetry_source.test.cjs
node frontend/tests/flight_physics.test.cjs
npm.cmd --prefix frontend run build

# Native targets, after building them through CMake
.\backend_cpp\build\Release\test_letterbox.exe
.\backend_cpp\build\Release\test_tracking.exe
.\backend_cpp\build\Release\test_physics_engine.exe
.\backend_cpp\build\Release\test_observed_spin.exe
.\backend_cpp\build\Release\test_analysis_journal.exe

# Dataset/trainer controls in the required Python environment
python -B -m unittest discover -s python_tools -p 'test_yolo*py' -v
~~~

**node backend_cpp/tests/test_complete_recording.cjs** exercises an isolated
360-frame fixture and indexed replay, requiring its local models/fixture.
Packaged smoke reproduction is in [DESKTOP_APP.md](DESKTOP_APP.md). Integration
tests use fixed test ports and should run sequentially; browser/CDP checks need
an isolated debugging profile. **test_live_camera.cjs explicitly opens camera 0**,
so do not include it in unattended checks.

Existing packaged evidence covers fixture analysis, source/model switching,
painted frame/coordinate agreement, genuine renderer saves, Unicode/space paths,
invalid/canceled source preservation, drafts and normal owned-service cleanup.
Picker API-boundary checks differ from manual OS interaction acceptance. See
[desktop verification](DESKTOP_VERIFICATION.md) and the ledger.

## Data and repository layout

Inside this project the release uses existing project data. Moved portable
releases use **%APPDATA%/TT Trajectory Engine**; **TT_DESKTOP_DATA_ROOT** can override
the desktop root. Local data opens the active directory.

~~~text
backend_cpp/                 Native inference, tracking, geometry, recording and tests
desktop/                     Electron shell, IPC, service ownership and packaging
frontend/                    Next.js views, review APIs and tests
python_tools/                Import, training, export, metrics and research tools
python_tools/dataset/
  raw_videos/                Canonical local originals; optional split/device subfolders
  raw_markup/                Dataset annotations
models/                      Local engines, manifests and upstream notices
training_data/live/          Captures, reviewed labels, session registry/source mapping
artifacts/analyses/          Versioned saved recording analyses
artifacts/desktop/           Desktop drafts, separate from approved labels
artifacts/                   Verification, isolated experiments and manifests
tools/                       Build, launcher, inventory and maintenance utilities
logs/                        Runtime logs in the active data directory
~~~

Large media, models, captures and builds are Git-ignored. Some linked artifacts
remain local and may be absent from a fresh checkout. Keep untouched originals
or a verified external archive: training completion alone does not make them
disposable. Hashes, review, sequential decoding, independent evaluation and cache
provenance can still depend on those bytes. **source_locations.json** reconnects
moved originals by verified identity.

Budget footage, extraction, checkpoints, caches and retained releases before
long runs. Remove generated files only when reproducible and not the only copy
of weights, labels, lineage or rollback evidence. The
[cleanup report](REPOSITORY_CLEANUP_2026_10_11.md) records the earlier preservation
audit; its free-space figures are historical.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing engine/DLL or incompatible GPU | Check model resources, release manifest and toolchain; keep the complete release folder. Engines from other GPU/runtime combinations are not automatically reusable. |
| Port 3000/9002 occupied | Use desktop dynamic ports. For developer servers, identify the owner and stop only the intended project process; preserve unrelated services. |
| CSS/JavaScript 400 or stale assets | Stop the developer frontend before rebuilding .next, then restart/reload. Desktop uses .next-desktop; source edits do not rebuild a package. |
| Same-origin review required | Use the app's own review origin and update an outdated package; preserve Host/Origin/revision checks. See the origin-fix report. |
| Older UI after building | Close the single-instance app and open the new EXE directly; batch launch prefers retained paired packages. |
| Ceiling lights detected or bright/blurred balls missed | Retain clean footage, review positives/confirmed negatives and evaluate the domain. Raising scores or tinting a preview does not prove improved accuracy. |
| Table box but no calibration | Grey/occluded/foreshortened geometry may fail refinement. A box alone is insufficient; do not fabricate a mapping. |
| Spin unresolved / 3D height unavailable | Expected when evidence is insufficient; sandbox settings are not measurements. |
| Seek waits / partial analysis | Inspect progress and logs. Native VFR/MOV timing is a known acceptance item. |
| Training disabled | Resolve session/device coverage, duplicate/visibility conflicts, reviewed intervals and checked-export blockers; do not bypass a failed plan. |
| Absence preparation unavailable / disk full | Supply the optional Python decoder environment and full-frame storage budget; keep existing labels intact. |

## Documentation and research

| Topic | Reference |
|---|---|
| Completion and remaining acceptance | [Requirement ledger](REQUIREMENTS_LEDGER.md) |
| Desktop lifecycle, build and runtime evidence | [Desktop guide](DESKTOP_APP.md), [verification](DESKTOP_VERIFICATION.md), [precision release](PRECISION_RELEASE_REPORT.md) |
| Deployed detector, physics and recording checks | [Verification](VERIFICATION.md), [venue failures](DETECTION_FAILURE_REPORT.md) |
| Capture and review | [Training guide](TRAINING_LIVE.md), [origin fix](TRAINING_REVIEW_ORIGIN_FIX.md) |
| Current data blockers | [Next actions](TRAINING_NEXT_ACTIONS_2026_10_12.md), [focused review queue](TRAINING_REVIEW_QUEUE_2026_10_12.md) |
| Standalone YOLO/native parity | [Experiment workflow](python_tools/YOLO_BALL_EXPERIMENTS.md), [implementation audit](YOLO_UPGRADE_IMPLEMENTATION_2026_10_11.md) |
| BlurBall and causal policy | [Implementation](BLURBALL_IMPLEMENTATION.md), [evaluation](BLURBALL_EVALUATION.md) |
| Hybrid and temporal student | [Hybrid v2](HYBRID_V2_IMPLEMENTATION_REPORT.md), [student](TEMPORAL_STUDENT_PROTOTYPE.md), [next experiments](YOLO_HYBRID_NEXT_EXPERIMENTS.md) |
| Twelve advanced capabilities | [Intelligence roadmap](INTELLIGENCE_ROADMAP.md), [high-precision plan](HIGH_PRECISION_IMPLEMENTATION_PLAN.md) |
| Papers and applicability | [Research catalog](RESEARCH_APPLICATIONS.md) |
| Media preservation and cleanup | [Cleanup audit](REPOSITORY_CLEANUP_2026_10_11.md) |

The research catalog accounts for the supplied PDFs and twelve paper URLs,
including access/identity limitations. Published CNN, temporal tracking and
robotic trajectory methods inform experiments; their accuracy/timing does not
automatically transfer to this application.

Remaining milestones include independent table/net landmarks and camera
intrinsics, calibrated uncertainty, reliable rally/player association, validated
spin/physical flight fits, player-relative tactical heatmaps, strike-seeded
counterfactual replay, actual highlight export/live buffering and adaptive
inference scheduling. Partial prototypes are documented explicitly and are not
counted as completed high-precision features.
