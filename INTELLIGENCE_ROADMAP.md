# Intelligence implementation and acceptance gates

This is an experimental implementation, not a claim that all twelve requested features are complete. C++ owns the trajectory, calibration, covariance, rule events, fitted stroke measurements, hard-example selection and model provenance. Next.js draws the resulting telemetry and sends explicit controls/review annotations.

## Running the experimental path

Build the `tt_intelligence_candidate` CMake target, then run it from the repository root with `--sim --experimental-intelligence`. The dashboard uses port 9002; close the existing backend before running a replacement. `--port 9102` is available for isolated native tests. Camera mode requires `--live --experimental-intelligence` and separate dashboard authorization before acquisition. Source modes never fall back into one another. The normal launcher does not opt into experimental intelligence.

The dashboard candidate uses the currently deployed detector weights. It does not silently replace weights or record the camera. A separate five-frame neural recovery candidate has been trained and evaluated offline; it is not enabled in the dashboard pipeline. The candidate target shares production sources and libraries, but has a separate executable so testing does not require overwriting a running engine.

## Feature status

| Requested feature | Implemented | Remaining before acceptance |
|---|---|---|
| Uncertainty-aware reconstruction | Shared `TrajectorySample`, image-plane Kalman covariance, projected planar covariance, measured/predicted/reacquired/rejected/lost states, bounded prediction, ellipse/corridor rendering | Calibrate noise on validation footage; meet independent localization and coverage gates. Height is unobserved; the 3D view shows a planar corridor, not volumetric reconstruction. |
| Persistent table digital twin | Corner-order alignment, temporal corner filtering, movement invalidation, reacquisition epochs, dropout expiry, edge-support diagnostic | Train/validate table landmarks and net/center-line constraints; solve orientation under foreshortening; assess independent reprojection and world-position error. |
| Rally event graph | Bounded C++ event history, links, candidate vs rule-supported status, source frame/time, exact-frame recording seek | Measure event precision/recall/timing against labels; serve/edge events, complete flight summaries and match outcomes are not implemented. |
| Active learning | Explicit difficult-example capture, bounded worker, perceptual/reason deduplication, raw frames, model hashes, local review UI, reviewed annotation export format | Real-camera capture/review validation, stronger diversity selection, table keypoint editing and full promotion dashboard. Live-domain improvement requires new reviewed recordings. |
| Stroke intelligence | Separate weighted incoming/outgoing fits, estimated planar speed, heading change, fit residual, table-end association | Validate strike timing and fit windows; reliable identities, reaction interval, bounce depth, target placement and full spin remain unavailable. |
| Counterfactual replay | Existing standalone physics sandbox remains available | A C++ replay seeded from a real validated strike, parameter isolation, recorded comparison, Monte Carlo uncertainty and replay UI are not implemented. |
| Highlights/reconstruction | Rule-supported high estimated speed produces a clip-candidate interval and reason | Full rally tagging, actual video cutting/export, live circular buffer and metadata export are not implemented. |
| Temporal recovery model | Five-frame CNN trained on game_1/game_2 crops, FP32 TensorRT export, native numerical parity, causal C++ detector-driven ROI replay with four-frame recovery limit | Causal recovery precision regresses; improve hard negatives and validate independent footage, natural occlusions and live clips before integrating/promoting. Neural output is a recovery hypothesis, not proof of visible ball presence. |
| Blur-aware preprocessing | Ball ROI sharpness, contrast, elongation and orientation diagnostics; sharpness affects measurement noise | Blur/visibility labels, validated blur bins, exposure-aware motion cues and comparison of optional ROI restoration remain. Sharpness is not a measured blur probability. |
| Physics consistency | Covariance-aware innovation gate, image displacement limit, reason codes and explicit rejected observations | The gate is a motion-consistency prototype, not a validated aerodynamic constraint solver. Event-aware acceleration/contact gates remain. |
| Tactical heatmaps | Ambiguous contact candidates are excluded explicitly | Player-relative aggregates, uncertainty weights, sample counts and tactical dashboard are not implemented. |
| Adaptive scheduler | Existing scene and preview cadence retained | State-aware GPU scheduling, temporal recovery dispatch, individual stage timing and throughput/latency comparisons are not implemented. Optimize after accuracy gates pass. |

## Measurement conventions

- `MEASURED` means a detector observation updated the filter, not exact ground truth. Raw detector centers and filtered centers remain distinguishable.
- Table coordinates use width 1.525 m and length 2.740 m. Homography projects onto that plane; an airborne image point generally does not project to its true horizontal location.
- The single-camera pipeline does not observe height. Z, full spin, reaction time and unsupported player identities are null rather than invented.
- Kalman covariance is propagated on actual source time. Missing predictions expire after eight source frames or 0.15 s; source-frame skips count toward the limit. Strong coherent observations outside the gate start a new segment instead of drawing a teleporting connection.
- The rendered ellipse uses the two-dimensional chi-square 95% contour of the model covariance. Its empirical coverage is separately measured and currently fails the intended target. It must not be described as a calibrated 95% guarantee.
- Table quality is a heuristic score, not a probability. Lock requires both repeated geometry and sufficient quality. Camera movement invalidates the mapping until a consistent new candidate is established. Geometry beyond the supported projection range is withheld.
- Four fitted corners do not provide an independent reprojection test. The payload therefore reports `reprojection_error_px: null`, corner stability residual and image-edge support. Net and center-line landmarks remain `UNOBSERVED`.
- Contact events are candidates. `RULE_SUPPORTED` strikes have multiple supporting cues but have not passed an event-accuracy benchmark. Table end A/B is not a verified human identity.
- Review edits require explicit class labels, preserve frame/clip drafts, and use revision checks to prevent silently overwriting another review. Saved predictions are not training truth until reviewed.

## Recorded verification

Native tests cover state transitions, covariance expansion/contraction, prediction expiry after skipped frames, coherent reacquisition, table ordering/movement/dropout, malformed geometry, weighted velocity fits, event deduplication after seek, hard-example deduplication and recorder error/expiry behavior.

The real TensorRT candidate test exercises calibration lock, bounded trajectory history, source-frame agreement, exact-frame seek/pause and invalid seek preservation on port 9102. The browser test uses a recorded native candidate packet to check rendering, and an isolated fixture directory to check actual annotation clicks, navigation and saved labels. It is not a live-camera accuracy test. The review API test checks frame confinement, same-origin writes, normalized labels, optimistic revision conflicts and reviewed/unreviewed persistence.

Evidence:

- [Native runtime](artifacts/intelligence_runtime.json)
- [Browser/CDP results](artifacts/browser_intelligence.json)
- [Review API results](artifacts/training_review_api.json)
- [Development localization and coverage](artifacts/trajectory_test1_recovery_metrics.json)
- [Verification of earlier detector and UI work](VERIFICATION.md)

The initial development replay on test_1 produced measured-point RMSE **17.28 px** and **56.26%** coverage of nominal 95% ellipses (631 measured annotated positions). Predicted-point RMSE was **40.04 px** (34 positions), with **41.18%** coverage. Those failures are preserved in the report. Four annotated positive frames were lost; all five explicitly absent annotated frames still had a track. Unknown annotations were excluded. There is no demonstrated accuracy gain from the experimental tracker yet.

That replay uses saved native detector centers and approximated 12-pixel boxes for ROI quality because the older CSV did not include box sizes. New prediction dumps retain actual box sizes. These results test image-plane covariance only, not world-coordinate or 3D accuracy. test_1 is a development sequence; test_2 has also been inspected and must not be relabeled as a sealed blind benchmark.

### Subsequent noise-selection experiment

Nine predefined acceleration/measurement-noise combinations were compared on game_1 chronological validation frames 70999–88599. The lowest measured-point RMSE selected acceleration sigma 40000 px/s² and measurement scale 1, recorded before running the test_1 regression. On test_1 that candidate produced 13.48 px measured-point RMSE and 89.12% nominal-95% coverage, versus 17.28 px and 56.26% for the default. State populations differ (781 vs 631 measured points), so this is not a paired accuracy gain. Reacquisition RMSE worsened to 32.75 px and all five explicitly absent frames still had tracks. Defaults remain unchanged and the calibration gate still fails. See [selection record](artifacts/trajectory_noise_selection.json) and [regression metrics](artifacts/trajectory_test1_selected_metrics.json).

### Temporal neural recovery: trained, exported, not promoted

The first candidate uses five RGB crops (15×128×128 input, 192-pixel source ROI) and emits a 32×32 heatmap plus a presence score. Training has 2,387 crops from the first 20,000 frames of game_1/game_2; validation has 921 crops from game_1 frames 70999–79416. No test_* sequence or live recording entered training. Epoch 11 of 12 was selected on validation. Training includes blur, lighting and synthetic current-frame masking. The score is not a calibrated visibility probability.

At score 0.95, prior-annotation-conditioned ROI validation gives 93.70% precision / 88.28% recall. Masking the current-frame ball gives 90.42% / 83.36%; this is synthetic occlusion and the ROI uses a previous annotation. Repeating only the current frame produces no accepted detections; that distribution-shift ablation is not a separately trained single-frame baseline. See [training audit](artifacts/temporal_data_audit.json) and [ROI evaluation](artifacts/temporal_recovery_report.json).

The FP32 TensorRT export passes a 32-input native/PyTorch comparison: all heatmap peak locations agree, maximum presence-score difference is 0.0000972 and maximum logit difference is 0.00754. This establishes export parity only. See [native parity](artifacts/temporal_parity.json).

A stricter C++ replay now positions crops solely from the last two accepted detector observations. It requires five consecutive source frames, bypasses neural recovery for strong detections, expires after four misses or 50 ms, and never lets inferred points prolong the detector anchor lifetime. Unknown annotations are used neither to center crops nor to decide dispatch; only the separate scoring script reads labels.

On all 17,601 game_1 validation source frames (2,451 annotated positives, 16 explicit negatives), this policy invoked recovery 731 times. At 15-pixel localization tolerance:

| Pipeline | Precision | Recall | F1 |
|---|---:|---:|---:|
| Deployed detector alone, score ≥0.25 | 89.13% | 83.27% | 86.10% |
| Detector + temporal candidate, score ≥0.95 | 88.29% | 85.52% | 86.88% |

Recovery added 55 correct localizations and 29 false positives; four more explicitly absent frames acquired false hypotheses. Its accepted, annotated recovery subset has only 65.48% precision. **The precision/negative gates fail; no promotion or live-accuracy improvement is claimed.** Many validation frames are unlabeled, so whole-video false-positive rates remain unknown. See [causal native evaluation](artifacts/temporal_pipeline_validation.json).

The next training iteration needs detector-conditioned hard negatives and missing-ball windows, followed by the same fixed causal policy and independent sequence checks. Actual live-camera improvement still requires reviewed representative recordings.

## Reproduction

From the repository root, after building the corresponding native targets:

```powershell
backend_cpp/build/Release/test_trajectory_intelligence.exe
backend_cpp/build/Release/test_rally_learning.exe
backend_cpp/build/Release/test_training_recorder.exe
backend_cpp/build/Release/test_temporal_window.exe
node backend_cpp/tests/test_intelligence_runtime.cjs
node frontend/tests/training_review_api.cjs
```

The browser variant adds `--browser` to the last command and requires the isolated test Chrome debugging endpoint on 9227. Tests create their own candidate backend/review server and stop those children. Do not run competing tests on 9102 or 3100 simultaneously.

```powershell
backend_cpp/build/Release/eval_trajectory_uncertainty.exe artifacts/multigame/test1.csv python_tools/dataset/raw_videos/test_1.mp4 artifacts/trajectory_replay.jsonl
python python_tools/score_trajectory_uncertainty.py artifacts/trajectory_replay.jsonl python_tools/dataset/raw_markup/test_1/ball_markup.json artifacts/trajectory_replay_metrics.json
```

`inject-misses` is an optional evaluator mode that hides five detector observations periodically. This is a synthetic dropout stress test, not a real occlusion benchmark. `sweep` compares explicitly recorded noise configurations; use chronological validation data for selection, not a test sequence.

The local recovery candidate can be reproduced with `python_tools/temporal_recovery.py` (`prepare`, `train`, `validate-export` in a new artifact directory), followed by TensorRT FP32 export. To replay the existing candidate without changing deployed models:

```powershell
backend_cpp/build/Release/eval_temporal_pipeline.exe artifacts/multigame/trajectory_validation.csv python_tools/dataset/raw_videos/game_1.mp4 artifacts/temporal_recovery_v1/temporal.engine artifacts/temporal_pipeline_validation.csv
python python_tools/score_temporal_pipeline.py artifacts/temporal_pipeline_validation.csv python_tools/dataset/raw_markup/game_1/ball_markup.json artifacts/temporal_recovery_v1/temporal.engine artifacts/temporal_pipeline_validation.json
python python_tools/test_temporal_metrics.py
```

The one-pass offline replay recorded median recovery GPU inference of 0.238 ms; that excludes decoding, ROI preparation, ball detection and the dashboard. It is neither a multi-pass throughput benchmark nor a single-session live-camera FPS result.

No new performance claim follows from these tests. Historical multi-pass GPU benchmarks and actual single-session camera throughput remain separate measurements. No automatic candidate promotion is enabled.

## September 29: second temporal candidate

Temporal v2 adds 185 causal detector-conditioned training cases (158 positive, 27 crop negatives) from the original game_1/game_2 training ranges. It was fine-tuned for eight epochs at learning rate 0.0003 from v1; the existing 921-crop validation set was unchanged. These remain chronological development splits within recordings, not the recording-disjoint blind evaluation required for release.

Native TensorRT parity passed all 32 argmax comparisons: maximum logit difference 0.01382, presence-score difference 0.0001274 and zero localization delta on identical tensors. See `artifacts/temporal_v2_parity.json`.

The matched end-to-end comparison uses exactly the deployed detector predictions in `artifacts/multigame/trajectory_validation.csv`, the same 17,601 source frames, four-frame/50 ms recovery budget, score 0.95 and 15 px tolerance as v1:

| Pipeline | Precision | Recall | Accepted recovery precision |
|---|---:|---:|---:|
| Deployed detector | 89.13% | 83.27% | — |
| Detector + v1 recovery | 88.29% | 85.52% | 65.48% |
| Detector + v2 recovery | 88.92% | 86.13% | 83.33% |

V2 added 70 correct localizations and 14 false positives on annotated eligible frames. Explicit-absence false positives increased from 2 to 6. Therefore it still fails the 95% recovery-precision gate, total precision non-regression and absence non-regression. It remains outside the deployed runtime. Live and independent-sequence accuracy remain unverified. Evidence: `artifacts/temporal_pipeline_v2_matched.json`; local training lineage and weights remain under `artifacts/temporal_recovery_v2/`.

An additional run in `artifacts/temporal_pipeline_v2_validation.json` used the older candidate detector predictions. Its baseline differs, so it must not be substituted for the matched v1/v2 comparison above.

The matched v2 replay measured 0.229 ms median recovery GPU inference. This is one offline replay excluding decode, crop generation, detector, transport and rendering; it is neither a multi-pass GPU throughput benchmark nor single-session live-camera latency.
