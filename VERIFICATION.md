# Detector and footage update — 25 September 2026

For the October 9 ceiling-ball/duplicate-leg follow-up, see [DETECTION_FAILURE_REPORT.md](DETECTION_FAILURE_REPORT.md). Native person-fragment suppression and recording scene synchronization pass regression/runtime/browser checks. The reported ball false positive, grey-table calibration and venue model training remain unfinished; the detector weights and threshold are unchanged.

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

## October 8: stale frontend build and missing favicon

The reported CSS and page JavaScript failures were reproduced with HTTP 400. The verified project-owned frontend PID 16380 was serving HTML/build ID `pjGtZfdbHr_sMCdcBVCmO`, while the rebuilt disk output had ID `o2PP_Ph5x46CqX2CQJuDS` and different CSS/page chunk names. Its `/api/health` still returned HTTP 200: endpoint responsiveness did not detect the stale process. The full asset responses were Next.js error HTML, not usable CSS or JavaScript.

Only that frontend was stopped after checking its exact project command path and creation timestamp. Production build passed compilation, lint/type checking and static generation; the updated frontend was started on 127.0.0.1:3000. The backend and camera were not restarted or authorized. A table/ball favicon was added at `/favicon.ico`.

The launcher probe now checks the page's referenced CSS/JavaScript with bounded HTTP requests, verifies MIME types, and drains all requests before reporting failure. `node tools/test_frontend_health.cjs` passed real HTTP fixtures for healthy assets, 400/404 missing chunks, HTML masquerading as JavaScript, unrelated identity, and a clean nonzero CLI failure. Frontend probes exit naturally; forced early process exit is retained only for the existing bounded WebSocket probe.

`node tools/test_frontend_assets.cjs` passed against the restarted production server: both `/` and `/training-review` contain the current disk build ID, every referenced CSS/JavaScript asset returns HTTP 200 and hashes identically to its current disk file, and `/favicon.ico` returns HTTP 200 with matching ICO bytes. Evidence: [HTTP acceptance report](artifacts/frontend_asset_runtime.json). The launcher frontend probe also passed with nine referenced assets checked. This is HTTP asset/build integrity evidence, not browser hydration/render verification or sign-off of the larger project plan.

Rebuilding beneath a running `next start` can repeat the mismatch. Stop the frontend before building, restart afterward, and reload an existing browser tab with Ctrl+Shift+R to replace its previous build URLs. The reported `content-script-start.js` is absent from application source and the served page asset list; its extension-style connection errors are not attributed to application logic without browser provenance evidence.

## October 8: original iPhone footage, lighting comparison and review import

Ten original MOV files from `C:\Users\Adi\Downloads\recordings` were audited. The user confirmed one venue/session, so none was assigned an independent train/validation/test split. Sequential OpenCV decoding counted every yielded frame and source timestamp; six five-frame bursts per clip produced 300 clean JPEG95 frames. Original source hashes, codec, dimensions, average FPS, interval ranges and decoder-count discrepancies are retained in [import lineage](artifacts/phone_recordings_import.json). All imported labels remain unreviewed. Originals were not edited/transcoded.

All decoded frames have 1920×1080 dimensions. IMG_7936–IMG_7939 average approximately 29.976 FPS; the other six average approximately 59.85–59.91 FPS. IMG_7936 reports 1101 frames but yields 1099 in a sequential scan. The first import stopped rather than fabricating the two final samples. A subsequent timestamp check caught random-seek disagreement in IMG_7944; the importer now samples via sequential decoding. Incomplete outputs were preserved under dot-prefixed failure directories and excluded from the review list. Native C++ MOV count/seek/PTS acceptance remains unfinished; these observations are from Python OpenCV 5.0 and do not establish native decoder parity. HDR metadata remains unknown.

Nine sequential source frames from IMG_7944 were tested with the unchanged deployed ball TensorRT engine, using raw, bounded gamma, CLAHE and an adaptive gamma/CLAHE candidate. All 36 top scores stayed below 0.25. The raw frame-5202 score was approximately 0.043; gamma approximately 0.062; CLAHE approximately 0.018. The initial random-seek experiment corresponded to different actual frames and is superseded for frame identity by [sequential comparison](artifacts/phone_lighting_sequential/report.json). These are unlabeled model-response experiments, not accuracy/precision/recall measurements, native C++ parity, multi-pass GPU benchmarks or live latency measurements. No candidate/filter/threshold was deployed. The actual screenshot-like frame occurs at sequential frame 5205 in this decoder; repeated comparison at that frame reproduces the earlier approximately 0.092 raw response.

Authoritative scene geometry code also restricts tabletop contours to saturated blue/green pixels inside the scene-model ROI. A grey table can therefore be detected yet fail calibration. No rectangle/default geometry was substituted. Learned landmarks, manual calibration and independent geometric validation remain required.

The review API accepts imported phone captures under the existing local path/origin/revision guards, lists original filenames, and exposes source/session diagnostics. Model preparation and temporal audit reject known-session overlap across splits. Fully occluded annotations, and partial visibility without an observable ball box, are excluded from detector datasets even when the surrounding scene was marked reviewed.

Production frontend compile, lint/type checks and route generation passed. Synthetic phone-import checks passed exact frame-image correspondence, sequential indexing, original preservation, unreviewed defaults, overwrite refusal and safe resume. Existing live-data and temporal-audit suites passed with same-session leakage and occlusion guards. The isolated review API test passed imported phone metadata/image handling and existing security/revision tests. Real HTTP checks confirmed all ten clips and 300 unreviewed frames, plus current dashboard/review build assets and favicon. Evidence: [phone review runtime](artifacts/phone_review_runtime.json), [review API](artifacts/training_review_api.json), [asset integrity](artifacts/frontend_asset_runtime.json). Browser annotation usability, reviewed ground truth, independent session evaluation and native MOV replay timing are not signed off.

## October 8: dimmed-footage false ball and focused review

Sequential TensorRT inference on IMG_7943 source frames 1680–1690 reproduced the reported ceiling-feature false positive. At frame 1685 the raw score is 0.375732, center (1440, 290.625) px, matching the screenshot's approximately 38% label. Nearby top proposals switch between two ceiling regions. The screenshot's approximate user-marked location near (1092, 582) px remains an unreviewed hint, never ground truth. An original-image context crop shows a small blurred feature at that location. Raw/gamma/CLAHE/adaptive inputs and a six-pass fixed overlapping 960×640 grid did not yield an accepted proposal near it. The grid is independent of the screenshot hint. This is evidence of this failure and failed recovery attempts, not a precision/recall evaluation or native parity test. No model/filter/threshold was changed. Evidence: [diagnostic report](artifacts/phone_dim_ball_diagnosis/report.json), [context](artifacts/phone_dim_ball_diagnosis/hint_context.png), [comparison](artifacts/phone_dim_ball_diagnosis/proposal_comparison.jpg). These offline multi-pass diagnostic experiments are not GPU throughput benchmarks or live hardware latency measurements.

The importer now accepts one recording and a bounded source-frame interval, with a distinct capture sequence, selection reasons and a readable note. Fifteen original-resolution JPEG95 images from source frames 1680–1694 were published as `phone_c343f0cdb72df492_2`, preserving source PTS/hash/session lineage and all-unreviewed defaults. Prior captures/annotations were not overwritten. Synthetic checks passed exact frame/image correspondence, invalid interval rejection, settings-mismatch refusal, source preservation and preservation of prior reviewed annotations. Evidence: [targeted import](artifacts/phone_dim_failure_import.json).

Training review now renders Fit, 100% and 200% with a bounded scroll viewport. Production component tests passed normalized annotation equivalence at all zoom levels and scroll offsets, source/native dimensions, overlay alignment, failed/unloaded/busy guards and source/zoom/cancel invalidation. Evidence: [component checks](artifacts/review_frame_image_checks.json). The frontend production build passed compile, lint/type checking and route generation; isolated HTTP API tests passed capture-note exposure and existing path/origin/revision safeguards. Current dashboard/review CSS/JavaScript match the disk build and the favicon returns HTTP 200.

An isolated in-app browser tab selected the targeted capture, navigated to its sixth image (source 1685), and verified native dimensions approximately 1920×1080 at 100%, approximately 3840×2160 at 200%, and working bounded horizontal/vertical scrolling. The enlarged original image visibly exposes the blurred ball. The user's existing review tab had an unsaved draft and was left intact; no reviewed labels were saved by automation. Evidence: [browser checks](artifacts/phone_failure_review_browser.json), [review screenshot](artifacts/phone_failure_review_zoom.jpg). This establishes review selection/zoom/render usability; it does not establish annotation correctness or detector accuracy. Candidate training and promotion remain unfinished pending reviewed data and independent session evaluation.

## October 9: simpler class-scoped annotation

Ball-only review is now the default. Click placement replaces the existing ball box and retains its source-normalized size; drag still sets the visible extent. Save & next, Enter, arrow navigation, explicit absent/hidden choices, a bounded previous-reviewed-box draft and a wrapping review queue reduce repeated work. Scene boxes and optional landmarks/events remain available in full mode. Drafts recover through browser-local storage and never become training labels until explicitly saved. Previous-box reuse does not estimate ball motion and does not approve a frame automatically.

The API stores per-class review flags and preserves earlier scene boxes/landmarks/events on ball-only updates. Empty or contradictory visible/unknown/fully-occluded ball labels are rejected in the new review flow. Python preparation selects only reviewed classes for each model, supports `--kind ball`, stores a version-2 per-model audit and excludes inconsistent legacy visibility labels. Ball-only annotations are eligible for audited temporal bursts; unchecked scene classes remain excluded from scene training. Known-session separation, cross-split duplicates, unknown-frame exclusion and source/revision guards remain active.

Six dataset tests, the temporal-burst test, review helper tests and native-dimension/scroll coordinate component tests passed. An isolated HTTP fixture passed ball-only class flags, explicit absence, contradictory visibility rejection, and preservation of saved scene and landmark/event data. A production browser fixture verified Enter-to-save/advance, one-click box replacement, reuse remaining a draft, recovery in a fresh page load, skipping, distinct absent/hidden states, optional scene controls and arrow navigation. Its persisted annotations were checked by Python: ball-only positives/absent frames are eligible, hidden frames are excluded, and all unchecked scene classes remain ineligible. Fixture files are separate under `artifacts/easy_review_fixture`; that temporary server was stopped. Evidence: [browser workflow](artifacts/easy_ball_review_browser.json), [API](artifacts/training_review_api.json), [readiness/persistence audit](artifacts/easy_review_data_audit.json).

The real IMG_7944 capture was inspected read-only in a separate browser tab; the older tab with unsaved edits was not reloaded. Thirty saved reviews exist; 19 ball-positive frames pass detector-label consistency checks. Source frames 1–5 are tagged visible with no box, source frame 19951 has a box tagged fully occluded, and the five final sampled frames have empty boxes with explicitly unknown visibility. These eleven are excluded pending clarification; unknown visibility does not become a negative even in a legacy full review. No original annotation was rewritten. The browser queue finds these states and wraps to earlier conflicts. This is label readiness, not proof of annotation correctness or accuracy. The updated frontend production build passed compile, type/lint checks and route generation; referenced assets match the current disk build. [Rendered workflow](artifacts/easy_ball_review.jpg). No model was trained/promoted and no camera was opened. Independent later sessions and model acceptance gates remain unfinished.

## Local desktop conversion — 2026-10-09

The portable Windows application is built under `desktop/dist/TT Trajectory Engine/`. See `DESKTOP_VERIFICATION.md` and `artifacts/desktop_packaged_smoke.json` for native-window runtime checks with developer runtime directories excluded, 360/360-frame recording analysis, frame/coordinate agreement across views, persistent drafts and service cleanup. Detector models are unchanged. Camera capture, manual OS picker selection and prolonged soak are not claimed by this conversion.

## BlurBall challenger — 2026-10-09

See `BLURBALL_EVALUATION.md`. Both downloaded clips completed native analysis but had almost no usable table calibration. An isolated RGB/affine three-frame evaluator reproduces native YOLO accepted/missing results on all 299 frames, with score differences at JSON rounding scale. On official match22 rally003 (272 frames, strict <4 source-pixel tolerance), BlurBall offline precision/recall are 98.48%/96.30%; causal latest-frame selection is 97.69%/94.07%; current raw top YOLO is 4.41%/4.44%. Conventional recall includes misplaced/missing visible points. This is one published test clip, only two absent labels, and no independent venue/webcam accuracy acceptance.

ONNX/TensorRT FP32 numerical parity passes on 18 real windows/51 components; maximum centroid difference 0.000209 source px. Offline 18-pass TensorRT forward median 9.85 ms excludes the rest of the pipeline; single-session live latency is unmeasured. Three focused geometry/tracking/scoring checks pass. Source/checkpoint hashes, per-frame predictions, comparisons and JSON reports are retained in `artifacts/blurball_research`. No application integration, default model promotion, reviewed-label rewrite, table calibration fix, physical 3D/spin inference or completion of the advanced plan is claimed.

## October 9: native integration and precision release

The earlier challenger-only scope is superseded by native/application integration, while default promotion, local fine-tuning, physical analytics and whole-plan closure remain unfinished. [BLURBALL_IMPLEMENTATION.md](BLURBALL_IMPLEMENTATION.md) and [PRECISION_RELEASE_REPORT.md](PRECISION_RELEASE_REPORT.md) record the exact contracts, results and limits.

Final native build and contact/gap/source-time forecast tests pass. The final main/controller report accounts for all272 recording frames, exact selected centroids, scene-frame identity, compatible cache reuse, one-/two-frame source completion without fabricated context and live standby file rejection with no camera opened. The published model's identical-input native numerical parity passes; the separate OpenCV-version preprocessing difference remains documented. The previous YOLO/scene weights and original phone review SHA-256 are unchanged.

Final packaged tests pass with a clean runtime PATH: real renderer-origin annotation save, actual pointer streak/save/reload, source/model switching, all-frame analysis, analysis/frame identity across views, C++-only forecast rendering, invalid/canceled source preservation and normal close cleanup. Root inspected the final rendered screenshots. The verified build occupies the standard application directory and the previous package is retained for rollback. [Packaged evidence](artifacts/blurball_desktop_smoke_final.json), [delivery](artifacts/blurball_release_delivery.json).

Native causal replay of the original IMG_7943 prefix decoded1694 frames and selected390 observations. All15 frames1680–1694 have accepted hypotheses, and the frame1685 marker visually follows the ball streak rather than the ceiling. No reviewed reference labels exist for that window, so accuracy/recall/localization metrics remain null. Originals/annotations were not altered. [Diagnostic scope](artifacts/blurball_research/phone_7943_causal/summary.json).

Ten forecast-evaluator integrity tests pass. Both complete272-frame lookahead and causal journals are scored against exact independently supplied source coordinates. Causal paired MAE improves but RMSE and per-horizon p95 regress versus constant velocity; the forecast stays off by default. Neither a learned forecaster nor calibrated prediction coverage is claimed. Live hardware latency, grey-table geometry, native MOV true timestamps/seek acceptance, independent session data and broad model promotion remain outstanding.

## Reviewed CNN dataset adapter — 2026-10-09

The strict [BlurBall dataset adapter](python_tools/blurball_reviewed_dataset.py) is implemented and checked against the inspected upstream affine and RGB normalization. It transforms original reviewed endpoints into output-map center, half-length and undirected degree-axis exactly once. Its explicitly versioned continuous binary capsule target avoids the upstream fractional-positive blank-map case while retaining the actual reviewed endpoint extent; this is an intentional target-policy change, not bitwise reproduction of upstream streak rasterization. Sharp-point targets retain the inspected radius-three behavior. Loss selection is mandatory: paper-described quality focal or upstream default MSE; neither starts training here.

[The final combined run](artifacts/blurball_dataset_adapter_tests.log) passes 31 tests: 11 adapter geometry/loader tests, four exporter regressions and 16 separately authored integrity tests. Tests check chronological RGB/target channels and NumPy/PyTorch batch shapes, source/output/inverse geometry, reversal invariance, tiny positives, explicit absence and exclusion of unknown/occluded/partial targets. Export rows and mutable labels/indices/images are reaudited; truncated or reordered indices, stale timestamps/source IDs and session/original-recording/decoded-image leakage fail closed. Two independently reproduced completeness defects were fixed before the final passing run; earlier failure logs are retained. The mandatory adapter audit is stronger than the older standalone exporter.

[Synthetic target previews](artifacts/blurball_dataset_adapter_v1/adapter_audit.json) were generated separately from real captures, with root visual inspection of the tiny-positive example. Equivalent PyTorch normalization matches bitwise; the full upstream torchvision training environment is not yet available. No real annotations, session assignments, source videos or model weights changed. This establishes structural training handoff readiness only. Real reviewed consecutive windows, independent sessions, adequate negative/contact diversity, a reproducible trainer and model acceptance measurements remain required.

## October 9: no-ball interval review and hybrid data foundation

See [HYBRID_PRECISION_IMPLEMENTATION.md](HYBRID_PRECISION_IMPLEMENTATION.md) for source-audited 9,164-frame import/causal replay, 300 unreviewed uniform images, explicit bulk ball-absence review and training provenance. Production API/page-handler checks, combined dataset tests, journal/mining integrity, frontend build and real packaged Chromium interaction pass. The application does not automatically label the supplied no-ball scene: 1,254 accepted model hypotheses remain suspected failures, with accuracy null. Twenty-four diverse nonoverlapping five-frame bursts are selected for review; coarse temporal/spatial diversity is not embedding clustering.

The separate updated package opens review without starting inference or a camera. Source changes, genuine renderer-origin save, selected-frame synchronization, drafts and clean shutdown pass on isolated fixtures. The original source and deployed YOLO/scene/BlurBall engines remain unchanged. Green-channel preprocessing has geometry/channel/clipping tests and an offline comparison option; no filter or hybrid is enabled by default. Independent sessions, temporal continuity audit, real trainer completion and all earlier acceptance failures remain unfinished.

## October 9: paired evaluation and actual backward-path evidence

Native source/model/frame-bound candidate selection passes 25 contract checks;
the journal scorer passes 25 integrity, reference and runtime-provenance tests.
The full 272-frame official development comparison is scored at strict <4 source
pixels. Causal BlurBall precision/recall are 97.69%/94.07%; the first conservative
selector achieves 100% precision on only five accepted predictions and 1.85%
recall. Its no-regression gate fails. No default changes or thresholds fitted on
this reused clip follow from these results.

The full no-ball paired replay accounts for all 9,164 frames, actual increasing
PTS and zero lookahead. Selected hypothesis counts are 2,557 YOLO, 1,254 BlurBall
and 15 experimental; unreviewed accuracy stays null. Historical evaluator binary
hashes are explicitly recorded-only after rebuild; Unicode/space and 500-frame
progress checks verify the archived subsequent executable. The initial apparent
stall was a stale directory-size misreading: the interrupted journal contained
7,204 consecutive indexed frames. Its corrected partial evidence retains the
initial snapshot and log.

Nine training-readiness tests pass; the real audit finds 19 structurally eligible
ball positives, zero verified negatives and zero reviewed CNN targets/windows,
all in one session. Actual network MSE/direct-dictionary quality-focal forward,
backward and in-memory optimizer updates pass on isolated artificial positive
and absent windows. All output-connected gradients are finite; only the exact
source-derived unused final fusion outputs are disconnected. The full upstream
Hydra runner remains unverified, and no candidate checkpoint is saved. See
[the current report](HYBRID_PRECISION_IMPLEMENTATION.md) for exact artifacts and
remaining data/calibration/physical/live dependencies.

## October 10: completed mixed-recording reviews

The user explicitly confirmed that positive boxes in Video Project 6.mp4 mark
actual balls. Its earlier no-ball description does not override saved labels.
Root visually inspected uniform image 211 / source 6520 and found a real ball
beside the net. No positive/absent label was changed by the audit.

The fresh [readiness report](artifacts/real_training_readiness_2026_10_10_v1.json)
supersedes the earlier zero-negative inventory: 750 saved images / 732 distinct
frames, 129 eligible detector positives, 284 reviewed absences, zero visible CNN
targets and 172 structural all-negative windows. All captures remain one session
with no frozen independent partitions. Seven overlapping positive labels differ
in box geometry, rather than presence/absence, and are excluded.

The [reviewed-negative subset scorer](python_tools/score_reviewed_negative_subset.py)
uses the previously audited complete native replay, actual source-frame/PTS,
source/model/journal hashes and saved review/image/history snapshots. Of exactly
284 unique reviewed absent frames, YOLO selects false balls on 42 and causal
BlurBall on 71. The experimental selector selects none here but retains its
failed positive-recall gate; it is not promoted. Uniform versus hard-mined counts
and selection bias are explicit in [the report](artifacts/reviewed_negative_subset_2026_10_10_v2.json).
Positive/unknown/unsampled frames are not declared absent. Historical runtime
bytes and JPEG-to-original-pixel binding are not independently reverified.

[Forty-two tests pass](artifacts/reviewed_negative_subset_tests_2026_10_10_v1.log):
25 existing paired integrity/scoring checks and 17 subset checks cover exclusion,
overlap/conflict handling, exact PTS, stale snapshots, explicit class review and
absence-history integrity. This is a sampled diagnostic, not global accuracy,
independent live acceptance, model fitting or a new deployment.

The [next-step handoff](NEXT_TRAINING_STEP.md) identifies IMG_7944 saved images
16–20 / source 14960–14964 for a small visible-CNN-label check, followed by real
positive-window diversity, temporal continuity and independent session splits.
Five targets are not an adequate training set. Camera capture was not authorized
or opened, existing services were not restarted, and user labels/models/splits
were preserved.

## October 10: real reviewed CNN targets and optimizer handoff

The user saved IMG_7944 images 16–20 / source 14960–14964 as reviewed center
targets. Independent qualitative inspection finds a real moving ball at the
annotated points; pixel localization error is not independently measured. The
[fresh v2 audit](artifacts/real_training_readiness_2026_10_10_v2.json) returns
`errors=[]`, 130 detector positives, 293 reviewed absences, eight visible CNN
targets (six points/two streaks) and 178 structural windows. All recordings remain
one unassigned session; synthetic fixture split files are not real held-out data.

The new [single-session staging command/reader](python_tools/stage_reviewed_training.py)
snapshots source/image/review/index/adapter identity, removes duplicate windows,
and excludes unknown/partial/occluded/conflicting CNN targets independently of
detector-box eligibility. Its [manifest](artifacts/reviewed_training_stage_2026_10_10_v1/manifest.json)
contains three fully visible windows and 175 absent windows. `ready_for_fitting`
remains false. Seventeen [staging tests](artifacts/reviewed_training_stage_2026_10_10_v1/tests.log)
and eight [smoke guard tests](artifacts/reviewed_training_smoke_guard_tests_2026_10_10_v1.log)
pass, including stale hashes, source/session/image overlap, nonempty positive
channels and exactly blank absent channels. No unknown frame becomes a negative.

The [actual reviewed-data quality-focal check](artifacts/blurball_reviewed_training_smoke_qfl_2026_10_10_v1/report.json)
passes two in-memory optimizer steps on visible frames 14960–14962 and absent
frames 1–3. Inputs are `[1,9,288,512]`, targets `[1,3,288,512]`. Each step has finite
loss, 212 connected parameter tensors with finite gradients and 206 changed
parameter tensors. Only the exact 39 inspected unused final fusion tensors are
disconnected. The positive channels retain 27/29/29 target pixels; absent channels
retain zero. Loss values are diagnostic, not accuracy. Original checkpoint and
complete stage are reverified after the optimizer. No checkpoint is saved.

The shared optimizer helper retains the [passing synthetic MSE check](artifacts/blurball_training_smoke_mse_2026_10_10_v3/report.json).
Current YOLO, scene, BlurBall engines and the upstream checkpoint retain their
previous hashes. No application build/runtime change, camera capture, source
editing, label overwrite, split assignment, default promotion, live performance
measurement or model-accuracy improvement is claimed. Independent data,
representative positive-window coverage, continuity and full candidate training
remain separate acceptance work. See [the updated checklist](NEXT_TRAINING_STEP.md).
