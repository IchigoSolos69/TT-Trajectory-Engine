# Detector and footage update — 25 September 2026

For the subsequent experimental intelligence work, see [INTELLIGENCE_ROADMAP.md](INTELLIGENCE_ROADMAP.md). Runtime correctness checks pass, but initial image-plane covariance coverage fails the requested scientific acceptance target. The advanced path therefore remains opt-in; this document's deployed detector results must not be interpreted as validation of the new tracker or event graph.

## What is deployed

- `models/yolov11_ttgames.engine`: ball detector trained with corrected game_1 labels and additional game_2 footage. SHA-256: `2f0b1c61d2b8e9ad9cc08d56ccce467ec24be3bc87d63e28c4896055b1a8efd9`.
- `models/scene.engine`: separate two-class person/table detector trained from OpenTTGames segmentation masks. SHA-256: `5eb0ecbfe0cddb8631cc89fbf6e7a2145d2a4a6b10595034afd4ab8b1a454cee`.
- Both use FP16 internal computation, FP32 I/O, and static 640×640 inputs. The ball output has 5 channels; the scene output has 6. Engines are local GPU-specific build artifacts and are ignored by Git.
- Original ball engine/ONNX backups: `artifacts/detector_improvement/original_deployment/`.

## Native ball detection results

These are C++ TensorRT predictions on every source frame, scored only where annotations establish ball presence or absence. A true positive requires the top detection to be within 15 pixels of the annotated centre. A wrongly localized positive frame counts as both a false positive and a false negative. The deployment threshold stays at **0.25**; scores were not raised artificially.

| Footage | Precision, original → deployed | Recall, original → deployed | F1, original → deployed |
|---|---:|---:|---:|
| test_1 | 75.7% → **87.6%** | 61.7% → **86.5%** | 68.0% → **87.0%** |
| test_2 | 72.0% → **90.8%** | 55.2% → **84.1%** | 62.5% → **87.3%** |

On test_1, mean model confidence among correctly localized detections increased from 0.388 to 0.654. Confidence is still a model score, not a calibrated probability of correctness.

test_1 contains 880 positive and 5 explicitly negative scored frames; test_2 contains 773 positive and 6 negative frames. Unannotated frames are excluded, not assumed negative. Consequently these precision figures do not establish false-positive rates across whole videos. test_1 was already a development set. test_2 stayed outside training but has now been evaluated across candidates; neither result should be presented as a newly sealed blind benchmark.

The first single-game candidate regressed on test_1 and was rejected. The deployed model was selected by chronological validation while training on game_1 and game_2; no test_* sequence was used for training. Unknown labels and partial missing-coordinate annotations were excluded, and explicit absent-ball labels were corrected. An interrupted training run resumed its saved optimizer/EMA state with fewer Windows data-loader workers after a shared-memory allocation failure.

Evidence: [machine-readable comparison](artifacts/verification_summary.json), [training split audit](artifacts/multigame/data_audit.json), [test_1 scores](artifacts/multigame/test1.json), [test_2 scores](artifacts/multigame/test2.json).

## Scene detector and table alignment

On 779 held-out test_2 annotated frames, native person detection achieved **99.3% precision / 98.5% recall** at confidence 0.4 and box IoU 0.5. Table boxes matched all 779 annotated tables under that criterion. These are results for one dataset sequence, not universal detection accuracy.

The separate model identifies a table region; color/contour refinement inside that region estimates its visible quadrilateral. Mean overlap with the annotated table mask was **79.7%**. This refinement supports blue/green surfaces and can fail on other colors, heavy occlusion, or unusual views. Ambiguous surfaces return no calibration; the old fabricated perspective grid is removed. Long-axis orientation is inferred from visible edge lengths and remains approximate under strong foreshortening.

People labels include officials. The UI therefore labels them “PERSON / PLAYER”; it does not invent player identities. Evidence: [scene evaluation](artifacts/scene_detector/test2.json).

## Footage and playback verification

Simulation reads only the selected dataset file. Live mode opens only the selected camera after authorization. No camera/file cross-fallback or synthetic footage is permitted. The source frame and its annotations travel in the same message and are committed together after JPEG decoding. The socket worker retains bounded pending output, handles partial writes, and replaces an unstarted old packet with the newest one.

Playback uses source timestamps. Pause preserves the exact image/detection pair; restart resets tracking. Rates are available from 0.25× to 4×. When inference cannot keep up, replay samples source frames and reports the skipped count. **Every displayed frame is analyzed, but accelerated replay does not analyze every source frame.** Physics receives the actual sampled source interval.

| Requested replay rate | Measured rate in one native playback session |
|---:|---:|
| 0.25× | 0.256× |
| 1× | 0.989× |
| 2× | 2.016× |
| 4× | 3.919× |

At 1× in that session, analysis throughput was about 70.5 FPS and preview delivery about 23.4 FPS. These measurements include the new scene/preview work and are **not** the historical multi-pass GPU-only benchmarks. A 120 FPS file does not imply 120 FPS live processing. The UI separately reports source, analysis, and preview rates.

An independently decoded dataset frame exactly matched its received JPEG after applying the same resize/encode procedure: mean absolute pixel error **0.0**, frame 922. Browser checks exercised all views, pause/resume, restart, and playback-rate acknowledgment without uncaught JavaScript exceptions.

Camera 0 supplied **21 actual 640×480 preview frames** in a separate hardware smoke test and was released after STOP_CAMERA. Only timing/detection metadata was saved. That session verifies live acquisition and release, **not live ball/player accuracy**: no representative labeled camera footage was available.

Evidence: [source routing](artifacts/detector_improvement/source_routing.json), [playback measurements](artifacts/playback_runtime.json), [frame alignment](artifacts/preview_alignment_verification.json), [browser trace](artifacts/browser_runtime.json), [live metadata](artifacts/live_camera_runtime.json).

## Physics and spatial views

The sandbox now integrates repeated table contacts, including a second bounce when the trajectory actually lands on the table. It includes the 20 mm ball radius, swept contact timing, restitution, tangential friction/spin transfer, finite table/net bounds, and floor termination. It no longer disables table collision after the first bounce or fabricates post-net drop points.

A 36-case launch sweep passed. Halving the integrator step moved the second landing point by **8.2 mm** in the repeated-bounce test. The restitution check agrees with an analytical drop to within **0.0015 m/s**. Native strike/bounce tests also pass, including a 15 FPS timing check that prevents doubled speed estimates.

2D/3D views use estimated table-plane coordinates rather than treating normalized image coordinates as metres. The 3D view has regulation markings, table legs, a mesh net, and perspective/top/side controls. Single-camera homography cannot reconstruct the airborne ball’s true 3D position. Measured telemetry no longer includes simulated height or speed-derived spin. The video bounce detector remains experimental. The sandbox checks do not establish bounce-detection accuracy on video.

## September 25: strike speeds, rotation and local recordings

The latest strike speed and last 100 events retain source timestamps and frame IDs across preview throttling. Values estimate table-plane motion; direction reversals can still be detection errors. Restart/source changes clear history. The footage, physics input and spatial projection use the same raw ball observation. Both views share coordinates without spring lag or fabricated height. JPEG decoding precedes telemetry publication; canvas drawing occurs at the React layout commit. Browser checks showed identical frame-16 coordinates (0.1341 m, 0.6082 m), with zero image/spatial frame-ID mismatches across 120 render samples. This verifies presentation consistency, not absolute calibration or physical 3D accuracy.

Observed rotation fits optical flow of visible white-ball surface features. It requires a resolved ball of at least 24 pixels, six features, robust inliers and multiple consistent samples. Only camera-axis rotation is estimated; full spin, topspin/backspin and temporal aliasing are unresolved. A synthetic 16.667 RPM marked sphere measured 16.466 RPM. Blank/tiny balls and time gaps correctly return unavailable. Real-camera spin accuracy has not been measured.

**Open recording…** opens the local Windows picker in simulation mode. Files are decoded before replacing the source; invalid selections preserve playback, and live mode rejects files. Backend loading/reset/isolation tests passed. A subsequent native-picker interaction was followed by an observed source change from game_1.mp4 (source 2) to test_2.mp4 (source 6), with the dialog closed and no error. The automation click reported unavailable coordinate geometry, so the exact final-action attribution is uncertain; this is an observed successful source change, not a deterministic automated native-dialog test. See [picker evidence](artifacts/native_picker_selection.json). The earlier approval/usage-limit interruption no longer prevents running the rebuilt app.

Evidence: [native build](artifacts/spatial_native_build.log), [runtime checks](artifacts/recording_selection_runtime.json), [browser alignment](artifacts/browser_spatial.json), [2D](artifacts/spatial_radar.png), [3D](artifacts/spatial_3d.png). Next.js production build, native physics/rotation tests, source/decode lifecycle tests and the 36-case sandbox sweep passed.

The local tracker uses ByteTrack-style two-stage IoU association and constant-velocity prediction, not a full Kalman implementation. Regression tests cover motion consistency, weak-observation association, occlusion recovery, class isolation, and expiry. Weak detections can maintain an identity internally but cannot bypass the displayed confidence threshold.

## Reproduce checks

The final nine native/JavaScript checks all exited successfully: [captured outputs](artifacts/final_checks.json). The native targets and Next.js production build also completed successfully.

From the repository root, with the built executables and both deployed models:

```powershell
backend_cpp/build/Release/test_letterbox.exe
backend_cpp/build/Release/test_scene_geometry.exe
backend_cpp/build/Release/test_tracking.exe
backend_cpp/build/Release/test_physics_engine.exe
backend_cpp/build/Release/test_bounce_matcher.exe
node frontend/tests/flight_physics.test.cjs
node frontend/tests/telemetry_source.test.cjs
node backend_cpp/tests/test_source_routing.cjs
node backend_cpp/tests/test_playback.cjs
```

`test_live_camera.cjs` explicitly opens camera 0, saves metadata only, and stops capture. Browser verification requires the local frontend/backend and an isolated Chrome debugging profile on port 9227. Do not run multiple backend integration tests simultaneously: they use port 9002.

To reproduce training/export, use `train_multigame.py`, `train_scene_detector.py`, and `export_yolo_onnx.py` (`--kind scene` for the scene model). Checkpoints, engines, and full prediction CSVs remain in the local `artifacts/` directories. Promotion requires native evaluation and export parity checks; the 32-frame final parity sample differed by at most **0.843 pixels** and **0.00222 confidence** between PyTorch and TensorRT.

## September 29 integration checkpoint

This checkpoint supersedes the old playback-skipping behavior described above. It does not sign off the entire action plan.

- The C++ recording worker writes consecutive per-frame records and an index independently of presentation. A 360-frame real-footage fixture with spaces and `café` in its filename passed pause, seek-ahead, frame-step, historical-event, invalid-file preservation, source-write-lock, and process-restart cache checks. Evidence: `artifacts/complete_recording_runtime.json`.
- Cached playback measured approximately 0.229×, 1.041×, 2.064× and 3.741× for requested 0.25×, 1×, 2× and 4× on this short fixture. All 360 analytic records remained accounted for; no analytical frames were skipped. These are presentation checks, not live inference benchmarks. Stored per-frame processing timings describe the original analysis, not replay throughput. Evidence: `artifacts/playback_runtime.json`.
- Native source-routing checks passed, including simulation camera-command rejection and live standby without camera authorization. The frontend lifecycle test passed source/decode/reconnect checks. No new browser render traces or deterministic native-picker test were run in this checkpoint; computer-use tools were unavailable in this session.
- Launcher health checks, compatible reuse, explicit restart, unrelated-port preservation and isolated-test cleanup passed. Port inspection errors no longer masquerade as free ports. Evidence: `artifacts/launcher_runtime.json`.
- Five-consecutive-frame recorder tests passed, including frame-gap rejection, explicit start, duration, clean frame cloning, queue draining and write-error reporting. These used synthetic images; live camera capture was not authorized or exercised.
- Extended review API tests passed landmark bounds, visibility/event validation, recorded event timestamps, immutable prior revisions and existing origin/path/stale-write guards. Evidence: `artifacts/training_review_api.json`. Visual review usability still needs a rendered browser check.
- Temporal capture audit tests passed unknown exclusion, frame-gap exclusion and cross-split duplicate rejection. No real live dataset is available yet.
- Native production/candidate builds and the frontend production build passed. Next.js emitted a webpack cache snapshot warning but completed compile, type checking and route generation.

Remaining release gates include full-length recording/disk-fault tests, browser alignment under rapid seeking, deterministic Windows picker selection, a 30-minute soak, reviewed live datasets, uncertainty/event/calibration accuracy, and the remaining advanced analytics described in REQUIREMENTS_LEDGER.md. No candidate was promoted solely because it compiled or improved recall.

## October 8: live training recording guide

The live Visualizer's capture panel now contains an expandable Recording checklist, gameplay steps & warnings guide. It provides four optional setup checks, an ordered six-step collection routine, post-capture annotation/split steps, and warnings about sampled images, camera-property verification, 60-second/disconnect termination, incomplete bursts and unreviewed suggestions. Compact warnings and mode descriptions remain visible outside the expanded guide. Checklist state is keyed to the camera/source session and is explicitly a human reminder, not an image-quality score or recording permission gate.

The production frontend build passed compile, lint/type checking and route generation. The initial non-elevated build failed with EPERM in the existing administrator-owned .next output tree; building with access to that tree passed. No global filesystem or Git trust settings were changed.

`node frontend/tests/training_capture_guide.test.cjs` passed execution/server-render checks against the production components: disabled capture before camera authorization, all three mode commands, active stop behavior, failure/drop warnings, checklist toggles, and capture/annotation/split guidance. Evidence: `artifacts/training_capture_guide_checks.json`. No real camera was opened and no training data or model was changed. These are component checks, not desktop browser visual evidence.

Desktop inspection was attempted using the computer-use skill. Chrome launch returned `Computer Use app approval timed out`; the rendered desktop check remains pending. No UI-permission bypass or alternate desktop automation was used.
