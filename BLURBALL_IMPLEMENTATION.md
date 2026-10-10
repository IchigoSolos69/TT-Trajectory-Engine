# BlurBall implementation — 2026-10-09

BlurBall is implemented as an explicit **experimental, opt-in C++ ball detector** alongside the existing YOLO detector. It uses the published pretrained checkpoint; no local recording has been automatically relabeled or used to train it. YOLO remains the default, and the separate scene model still supplies person/table proposals.

## Local application controls

The launcher and dashboard offer **Ball model → BlurBall · experimental**. Select a recording or a live session as usual. Changing the ball model stops only the application's own backend, starts a new session with the selected engine and uses a distinct saved-analysis identity. Switching back to YOLO is supported. Camera authorization remains a separate explicit control.

The verified package is installed under `desktop/dist/TT Trajectory Engine`; the previous release is retained under `desktop/dist/TT Trajectory Engine before precision`. The final packaged selector/origin/frame-identity/streak-review/forecast smoke passed with developer runtime directories excluded. Its screenshots were inspected after compositor synchronization: the saved streak and frame-30 forecast are visible. Source frontend production build and isolated standalone review tests passed. This scoped release does not close the broader requirement audit, physical analytics, live accuracy or the 30-minute soak. [Delivery manifest](artifacts/blurball_release_delivery.json), [final packaged checks](artifacts/blurball_desktop_smoke_final.json).

CLI selection is `--ball-backend blurball` with an explicit `.engine` path when required. Model resources and hashes are recorded in `models/blurball.manifest.json`; the upstream MIT source notice is retained in `models/BlurBall-LICENSE.txt`. Checkpoint redistribution terms remain a public-release dependency: the checkpoint contains no separate license metadata.

## Authoritative native representation

`BlurBallEngine` validates fixed FP32 device tensors `[1,9,288,512] → [1,3,288,512]`. Three same-sized consecutive BGR frames are converted to RGB, uniformly affine resized and ImageNet normalized in chronological CHW order. Empty/type/size or tensor-contract errors are explicit; there is no fallback to invented detections.

Thresholded heatmaps produce connected components, heatmap-weighted centers, component mass and PCA streak evidence. Component selection uses maximum mass within the upstream strict 300-source-pixel previous-position gate. This is a heuristic association rule, not a physical speed limit or calibrated uncertainty. The weighted center, rather than the streak bounding-box midpoint, feeds the C++ tracker and every rendered view.

The telemetry contract includes model SHA-256, backend, experimental state, uncalibrated score kind, time support and timestamp basis. `ball_blur` provides source-pixel center/endpoints, full endpoint-to-endpoint `extent_px`, undirected `axis_deg`, component mass and heatmap peak. The extent is **full length**, whereas the paper's `l` label is half-length. These values must not be mixed. Axis direction has a 180° ambiguity and exposure remains null.

The dashboard labels the backend experimental, shows its causal/lookahead policy and draws the source-coordinate streak. Its score is displayed as a heatmap peak; it is not a confidence percentage. The frontend does not compute another trajectory.

## Recording and live policies

| Operation | Source support and behavior |
|---|---|
| Complete recording analysis | Collect all eligible overlapping three-frame windows. A source frame can use its current/past/future channels, with at most two frames of lookahead. Emit source frames in order, including the trailing frames. Presentation controls do not change this analysis policy. |
| Causal live inference | Use only output channel 2 from `[t−2,t−1,t]`. The first two real frames are warmup; no future frames are used. Driver delivery at the camera's advertised sensor cadence remains unverified. |
| Fewer than three recording frames | Preserve and emit all real source frames with no temporal detection, rather than fabricate missing context. |
| Gaps/reset | Reset context on duplicate/discontinuous live IDs, explicit resets and resolution changes. Do not duplicate or interpolate frames to pad the model input. |
| Source identity | Keep original source-frame IDs and frame accounting. Do not delete exact duplicate source frames or renumber their successors. |

The persistent controller manifest includes engine/source/scene/runtime hashes, preprocessing revision and temporal policy, preventing incompatible cached YOLO analysis from being reused as BlurBall analysis. Current recording timestamps are nominal source-frame/FPS times, and live timestamps use the host capture clock. This does **not** resolve variable-frame-rate MOV true-PTS/seek integrity (`SYNC-MOV` in the ledger).

This detector is distinct from the earlier optional five-frame ROI **recovery** candidate. Integrating BlurBall does not promote that candidate or satisfy its accepted-recovery precision gate.

## Native evidence and limits

| Check | Result and scope |
|---|---|
| Identical-input numerical parity | Passed on 18 real windows: maximum native-versus-PyTorch logit error 0.00003815; centroid delta 0.0002044 source px; no unmatched component counts. [Report](artifacts/blurball_research/native_identical_input_parity.json). |
| Independently preprocessed Python/native comparison | The original strict comparison **failed**. Native OpenCV 4.14 and Python OpenCV 5.0 affine uint8 interpolation differ despite identical decoded BGR images; maximum resulting logit difference 2.7945 and centroid difference 1.3467 px across those windows. This is retained as a known version-specific preprocessing difference. [Failed comparison](artifacts/blurball_research/native_parity.json), [diagnostic](artifacts/blurball_research/native_preprocess_diagnostic.json). |
| Native tensor/input contract | Empty input, wrong image type and mixed dimensions rejected. [Manifest](artifacts/blurball_research/native_contract_checks/manifest.json). |
| Stream frame policy | Passed 272-frame recording accounting/image identity/all-window components and tail handling; 12 causal-frame checks, two-frame warmup, ID-gap/duplicate-ID/explicit/size resets and one-/two-frame clips. No camera opened. [Report](artifacts/blurball_research/native_stream_policy/stream_result.json). |
| One official labeled clip | Native heatmap evaluation reproduces offline precision 98.48%, recall 96.30%; causal precision 97.69%, recall 94.07%, at strict distance `<4` source pixels. Only 272 frames, 270 visible and two absent references; this is not venue/live acceptance. [Metrics](artifacts/blurball_research/native_official_22_003/metrics.json). |
| Native main/controller/cache integration | Passed 272 journal/index rows, tail/scene-frame identity, 264 measured-ball frames and matching selected centers (maximum 0.0000007071 source px versus the native evaluator). Normalized overlay JSON rounds coordinates to four decimals (maximum 0.0721 source px difference on this clip). Provenance survives all rows, compatible cache reuse passes, short one-/two-frame recordings remain complete with no padded detection, and live standby rejects a file while leaving the camera unopened. [Report](artifacts/blurball_research/native_main_integration.json). |
| Packaged application selector, cache, rendering, cleanup and renderer review save | Passed with a clean runtime PATH, zero renderer errors, 272 analyzed BlurBall frames and model/analysis/frame identity checked at paint. The native selector changed backend/cache identity, switching back released the experimental process, and normal close released owned ports/services. The real Chromium renderer saved/reloaded fixture annotations with HTTP 200; the latest smoke also verifies actual streak-pointer annotation/save and authoritative forecast rendering. [Report](artifacts/blurball_desktop_smoke.json). |
| Live capture-to-display accuracy and latency | Unmeasured; no camera capture performed for these checks. |

The initial independent-preprocessing failure is not hidden by the identical-input pass. Full cross-version pixel/logit parity remains unresolved, and broader native accuracy evaluation is required before default promotion. The official checkpoint's original training manifest is unavailable; the one official clip does not prove blind generalization.

The short-recording check exposed an OpenCV one-frame MJPG AVI seek failure: setting frame zero could report success while the next read failed. The controller now reopens the validated source with a fresh decoder before source replacement. The failed diagnostic is retained; the final one-/two-frame controller tests pass.

**Offline multi-pass native timing:** 270 windows on the RTX 5050 yielded median GPU inference 9.73 ms and complete adapter inference wall time 17.95 ms. That wall time includes adapter preprocessing/transfers/heatmap postprocessing, but excludes source acquisition, the scene model, downstream analytics, preview encoding, transport and rendering. These are non-exclusive offline stage measurements. **Single-session live capture-to-display timing remains unmeasured.** A 120 FPS end-to-end claim is unsupported.

## Quantities the model does not provide

BlurBall supplies image observations and streak evidence. It does not supply person identity, table calibration, physical height, strikes, spin RPM or a measured 3D trajectory. Existing table-plane projection and derived planar speeds retain their estimation limits. If calibration is missing, improved ball detection alone cannot justify physical analytics. Unknown height/spin/exposure remain unavailable; streak length is not converted to km/h from video FPS.

Current table refinement still struggles with grey tabletops and independent landmark calibration is unfinished. Contact-aware trajectory fitting, labeled event evaluation, empirical covariance coverage and reviewed independent venue positives/negatives remain separate gates. No reported single-clip improvement closes those items.

## Research reviewed for this integration

[RESEARCH_APPLICATIONS.md](RESEARCH_APPLICATIONS.md) accounts for both attached PDFs and all 12 supplied URLs. It identifies applicable work on multi-frame/blur observations, contact-specific flights, calibrated geometry and conditional spin. It also separates stereo/high-speed/event-camera results from what one ordinary camera supports.

Three IEEE IDs (11147615, 10572328 and 4281948) remain bibliographically unresolved because access/metadata retrieval was blocked. The SAGE spin paper and JCMSI rotation paper have limited abstract/excerpt access; their uninspected equations/constants were not implemented. Additional titles/DOIs/PDFs are needed for those specific methods. Paper results do not replace the application's own validation or permission to auto-approve annotations.

The broader plan and whole-chat audit remain unfinished. See [REQUIREMENTS_LEDGER.md](REQUIREMENTS_LEDGER.md), [BLURBALL_EVALUATION.md](BLURBALL_EVALUATION.md) and [TRAINING_REVIEW_ORIGIN_FIX.md](TRAINING_REVIEW_ORIGIN_FIX.md).

## Paper-informed image-plane flight baseline

A contact/motion-boundary-aware recent-observation quadratic fit is implemented in C++ as an **image-plane research baseline**, with source-pixel coordinates, explicit fit support, segment boundaries and a maximum 50 ms horizon. It is a position-only fit: the blur axis is displayed as evidence but is not yet a fitting constraint. This is not an LSTM/mixture network, physical 3D reconstruction or calibrated confidence envelope. Forecast points never become measured ball detections.

`python_tools/evaluate_imageplane_forecast.py` validates complete saved journals and scores exact next-one/two source frames against independent official CSV coordinates. It compares the stored polynomial, last-observed-position and last-two-observation constant velocity on the identical eligible targets, reports coverage/abstentions and excludes absent-ball labels from location error. It never interpolates GT or chooses resets/parameters from GT. Ten synthetic integrity/exact-polynomial tests pass in [the test log](artifacts/imageplane_forecast_tests.log), including causal-policy scoring and rejection of filtered-tracker inputs without separate fitting-observation provenance.

The completed packaged native journal contains all 272 source rows, 264 observed-ball frames, 192 available fits and 80 abstentions. Exact nominal source-frame targets at about 16.74 ms and 33.49 ms produce the following paired results. Error is Euclidean source-pixel position error, not physical distance:

| Source-frame horizon | Scored / visible target opportunities | Quadratic MAE / RMSE | Constant-velocity MAE / RMSE | Last-position MAE / RMSE |
|---|---:|---:|---:|---:|
| +1 | 191 / 269 (71.0%) | 2.586 / 4.765 px | 3.008 / 5.072 px | 15.760 / 17.731 px |
| +2 | 188 / 268 (70.1%) | 5.372 / 11.291 px | 6.466 / 11.527 px | 31.221 / 34.771 px |

Across all 379 paired targets, quadratic MAE/RMSE is 3.968/8.642 px, versus 4.723/8.881 px for constant velocity. The combined p95 is slightly worse for the quadratic fit: 17.309 versus 17.155 px. This is limited offline improvement in mean error on an eligible subset, with substantial abstention and no contact-labeled breakdown. It does not establish superiority on all error measures or all frames. [Full exact-frame metrics, sample pairs, hashes and provenance](artifacts/blurball_research/imageplane_forecast/metrics.json).

Recording BlurBall observations can already use the two future target frames, even when the fit's support-frame IDs are past-only. This makes the current recording experiment **noncausal**; it cannot demonstrate unseen-future or live forecasting skill. Contact ground truth and uncertainty coverage are unavailable; there is no promotion approval. A causal detector/held-out-session experiment and contact-specific reference labels are needed before interpreting forecasting quality, or training/promoting the neural predictor described in the papers.

**Causal forecast replay is now evaluated separately.** `backend_cpp/tests/eval_imageplane_forecast.cpp` decodes the recording chronologically and invokes the actual native stream with only channel 2 of `[t−2,t−1,t]`. It preserves two warmup rows without a detection, declares zero lookahead and supplies raw detector centers to the C++ flight estimator. No scene/contact reference or future label influences selection, fitting or resets; only observation gaps and motion discontinuity determine its segments. All 272 rows and metadata checks pass, with 260 observed-ball frames, 199 available fits and 73 unavailable fits.

| Causal source-frame horizon | Scored / visible target opportunities | Quadratic MAE / RMSE | Constant-velocity MAE / RMSE | Quadratic / constant-velocity p95 |
|---|---:|---:|---:|---:|
| +1 | 197 / 269 (73.2%) | 3.118 / 6.283 px | 3.389 / 6.033 px | 13.929 / 10.494 px |
| +2 | 194 / 268 (72.4%) | 6.492 / 14.062 px | 7.161 / 13.213 px | 29.617 / 26.176 px |

Across 391 paired targets, the quadratic fit lowers mean error to 4.792 px from 5.261 px, but **increases RMSE to 10.863 px from 10.245 px**. The per-horizon p95 errors also worsen; combined p95 is nearly equal (21.000 versus 21.024 px). These outlier regressions prevent a claim that the quadratic predictor is reliably better or ready for default use. The causal and lookahead experiments have different eligible subsets, so their aggregate errors are not a paired comparison of detector policies. [Causal exact-frame report, full accounting, sample pairs and hashes](artifacts/blurball_research/imageplane_forecast_causal/metrics.json).

This is a **causal offline source replay**, not a measured webcam session or live capture-to-display latency benchmark. It is still one labeled recording with no independent contact labels, calibrated forecast uncertainty or held-out venue/session acceptance. The method remains an opt-in research baseline; it neither trains a neural predictor nor approves model promotion.
