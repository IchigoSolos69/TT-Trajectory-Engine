# High-precision detection and forecasting implementation

Updated 2026-10-09. This plan connects the neural methods in [RESEARCH_APPLICATIONS.md](RESEARCH_APPLICATIONS.md) to implementation and acceptance work. Accuracy means agreement with reviewed reference labels; a larger displayed model score is not itself an improvement.

## Current concrete work

- C++ remains responsible for source identity, detection, geometry, contacts and forecasts. The BlurBall temporal detector is a separately identified candidate; the existing detector remains the default until its deployment gates pass.
- The short-horizon image-plane forecast is a contact-segmented quadratic baseline, disabled by default. It is not a trained neural forecaster, physical 3D reconstruction or measured future path. Contact candidates, discontinuities and missing observations must bound/reset its history.
- Training review supports an **optional blur-streak drag**. Endpoints remain normalized to the source image; source JPEG dimensions are checked by the API. The stored center and half-length are derived in original pixels. The axis is undirected, and a zero-length sharp-ball label has a null axis. Only explicit save marks a label reviewed.
- Streak annotation preserves existing ball/person/table boxes. A streak-only review does not become a YOLO empty-ball negative. Unknown/occluded frames remain excluded; confirmed absence is a separate explicit label.
- `python_tools/export_reviewed_streaks.py` audits/exports a versioned three-frame JSONL handoff with session grouping, source hashes, annotation revision hashes and source timing. It never changes original captures or annotations. The separate [reviewed BlurBall dataset adapter](python_tools/blurball_reviewed_dataset.py) is now implemented and checked on synthetic fixtures; real reviewed supervision and the complete training runner remain separate prerequisites.

Focused geometry, production page-handler/API, origin and source-coordinate checks verify this new review flow. The [final packaged Chromium smoke](artifacts/blurball_desktop_smoke_final.json) also passed an actual streak-pointer drag and explicit Save/reload on a fixture, preserving its existing boxes; [rendered streak evidence](artifacts/blurball_desktop_smoke_final.streak.png) is retained. The same run verified model-selector/cache identity, all 272 official source frames and authoritative C++ forecast rendering at frame 30. This verifies the application interaction; it does not supply reviewed venue ground truth, full model training or independent live accuracy.

The published BlurBall checkpoint now runs through the opt-in native C++ adapter. [BLURBALL_IMPLEMENTATION.md](BLURBALL_IMPLEMENTATION.md) records its source/model hashes, causal live channel policy and separate recording lookahead policy. Identical-input native/PyTorch parity passes, but independently preprocessed OpenCV-version parity remains unresolved. The checkpoint's original training manifest is unavailable, so the existing 272-frame official clip is development/regression evidence, not a demonstrated blind test.

## Original phone failure evidence and remaining supervision

The original source is `C:\Users\Adi\Downloads\recordings\IMG_7943.MOV`, SHA-256 `c343f0cdb72df492b3de2a69278e4d9c6884e9a22e80a5db69b903a2a9129e7f`: 2,114 reported source frames, 1920×1080 HEVC and 59.91214 average FPS. Its imported capture manifests preserve decoder source PTS, rather than playback wall time. Frame IDs below are **one-based source IDs**:

| User-reported failure | Imported source PTS | Exact review image and metadata |
|---|---:|---|
| Source frame 1685, ceiling-light candidate/missed-ball screenshot | 28,106.6667 ms | [capture metadata](training_data/live/phone_c343f0cdb72df492_2/capture.json); [frame_6.jpg](training_data/live/phone_c343f0cdb72df492_2/frame_6.jpg), burst 2/index 0 |
| Source frame 1848, ceiling candidate/duplicate-leg screenshot | 30,828.3333 ms | [capture metadata](training_data/live/phone_c343f0cdb72df492_3/capture.json); [frame_9.jpg](training_data/live/phone_c343f0cdb72df492_3/frame_9.jpg), burst 2/index 3 |

The three `phone_c343f0cdb72df492_1`, `_2` and `_3` folders contain 60 annotation entries, all currently `reviewed:false` with empty class lists and no reviewed streaks. These are **unknown labels**, not verified absent balls or furniture negatives. All belong to `iphone_same_venue_session_1`; they cannot populate independent train/validation/test splits. [The earlier frame-1685 diagnosis](artifacts/phone_dim_ball_diagnosis/report.json) uses an explicitly unreviewed approximate screenshot hint; its proposals are useful debugging evidence but do not establish localization error, precision or recall.

Native BlurBall now completed a chronological **1,694-frame causal prefix** of this 2,114-frame source, with zero lookahead and two real warmup rows: [manifest](artifacts/blurball_research/phone_7943_causal/manifest.json), [completion](artifacts/blurball_research/phone_7943_causal/completion.json), [journal](artifacts/blurball_research/phone_7943_causal/frames.jsonl). At frame 1685 its center is `(1087.952271, 578.530212)` source pixels; the [overlay](artifacts/blurball_research/phone_7943_causal/frame_1685.jpg) visually follows the ball region instead of the earlier ceiling-light candidate. This is a qualitative source-frame check, **not reviewed ground truth or an accuracy result**. Frame 1848 is outside that completed prefix; full-source/independently labeled phone acceptance remains unfinished. Original source files and annotations remain unmodified.

## Paper-to-method mapping

| Reference | Method worth applying | Implementation scope and dependency |
|---|---|---|
| BlurBall, Gossard et al., 2025 | HRNet with Squeeze-and-Excitation over three RGB frames; quality-focal heatmaps covering the whole visible streak; center/half-length/axis rather than a sharp point at the leading edge. | First neural training priority. Audited dataset adapter is implemented; obtain real reviewed independent data and verify the complete trainer before fine-tuning. Compare causal latest-frame output separately from offline outputs using future frames. |
| Li et al., *Fractals*, 2022 | Structured-output CNN localization followed by LSTM and mixture-density trajectory prediction; alternatives for free flight and rebound. | Research challenger after reliable labeled targets exist. Its 3D setup uses synchronized cameras and synthetic Bullet3 trajectories. Start with honest **image-plane** forecasts; do not borrow its physical landing-error claims for monocular video. |
| Zou et al., *Scientific Reports*, 2024 | Motion candidates and graph message passing for object selection/association. | Optional offline ambiguity experiment. Reported complete 1080p cost is far above the current live latency budget; no reproducible pretrained artifact was verified. Benchmark against bounded native association before adding it. |
| Gomez-Gonzalez et al., *Robotics*, 2019 | Lightweight CNN segmentation followed by cross-camera geometric agreement. | Modular observation/reprojection tests apply now. The multi-camera 3D consistency stage needs synchronized additional cameras; it cannot validate monocular height. |
| Lin et al., *Sensors*, 2020 | Separate ANN predictors before/after bounce with trajectory-coefficient targets. | Supports contact-specific fitting. A learned physical rebound model needs calibrated 3D training trajectories, friction/spin evidence and independent tests. Apparatus-specific restitution is not a production constant. |
| Liu and Ding, *Frontiers in Neurorobotics*, 2022 | Cross-layer detector features and learned landing prediction; target-preserving augmentation. | Review augmentation effects on tiny balls. Its laboratory/multiple-camera/high-speed-logo setup is not evidence for spin or live venue performance on 60 FPS phone recordings. |
| Ji et al., 2018; Myint et al., 2015 | Saliency/color/motion candidates within predicted ROIs. | Compare as supporting evidence, not automatic labels or a replacement selected merely for historical timing. Candidate generation must not reinforce a wrong ceiling-light track. |
| Wang et al., 2018/2019; Liu et al., 2012; Gossard et al., CVPRW 2024 | Physical flight comparisons, surface image registration, event-camera logo flow. | Conditional spin research only. Full methods are unavailable for the first two references; event-camera inputs require different hardware. Current small unmarked balls keep unresolved spin. |
| Nakazato et al., 2024 | Human eye-tracking experiment. | Context for readable displays; no CNN weights, detector upgrade or automatic tactical interpretation follows from it. |
| IEEE 11147615, 10572328, 4281948 | Identity/method unverified because access failed. | Obtain a title/DOI/PDF before making algorithm decisions. No title or technique is guessed. |

## Stage 1 — Audit the useful supervision

1. Keep the user's ten iPhone clips from the same venue/session in **one split**. Another independently recorded session is required for live validation and another independent test session for a blind live promotion result. Existing OpenTTGames development/regression footage is useful but does not replace live test data.
2. Review consecutive bursts containing a visible ball, sharp and long blurred appearances, ceiling-light/furniture negatives, partial visibility, player occlusion, serve and contact transitions. Do not label a hidden center by interpolation and do not accept a model's suggestion automatically.
3. Use one drag for visible streak ends and one click for a sharp ball. Confirm only the visible observation. Preserve partial visibility as its own evaluation subset; audit whether a truncated streak is an appropriate center target before including it in initial fine-tuning.
4. Preserve original frames, timestamps, dimensions, source/session identity and labels. Annotations must retain revisions. Training splits are assigned outside the review page; returning to a clip does not move it between splits.
5. Audit the export. It requires all three frames of a causal training window to have an explicitly reviewed visible streak or confirmed absence. Unknown/occluded frames break supervised windows; source gaps, burst boundaries and dimension changes also break them. Identical images across splits are rejected.

Example read-only audit:

```powershell
python python_tools/export_reviewed_streaks.py training_data/live artifacts/streak_splits.json
```

Export into a **new** directory only after the audit reports reviewed windows in independent train/validation/test groups:

```powershell
python python_tools/export_reviewed_streaks.py training_data/live artifacts/streak_splits.json --output artifacts/streak_training_v1
```

The split JSON is `{"train":["capture_id"],"val":["other_capture_id"],"test":["third_capture_id"]}`. These names are placeholders, not existing verified recordings. The exporter verifies each capture's actual session ID and rejects reuse of one session across groups. Counts alone do not establish adequate sample size.

## Stage 2 — Fine-tune a detector before learning more analytics

1. **Implemented:** the audited JSONL dataset/target adapter described below. It keeps raw endpoint provenance while generating explicit transformed targets. It never synthesizes unreviewed absence or guessed occluded-ball labels. Target-changing augmentation remains disabled pending independent validation.
2. Start from the existing inspected BlurBall checkpoint with a **versioned, explicitly chosen training configuration**. Its original training lineage is unavailable; do not assume inspected upstream split settings prove how those weights were trained. Add reviewed venue-training examples with sharp/blur/absence diversity. Keep independent validation and test footage entirely out of gradient updates and mining.
3. Test target-preserving exposure, contrast, compression, noise and motion-blur augmentation. Do not treat a brightness filter as a proven fix or shrink the ball with aggressive mosaic/crops until tiny-target recall has been measured. Evaluate raw and filtered inputs using the same labels and frozen threshold.
4. Use validation localization, precision/recall and absent-frame false positives to choose checkpoints and thresholds. Do not select by training loss or mAP alone. Evaluate severe blur, near-player occlusion, contacts and bright distractors independently.
5. Export the chosen candidate through the existing isolated ONNX/TensorRT process. Verify input/output parity and source-coordinate mapping at native resolution. Then repeat **native C++** causal evaluations, not merely Python evaluations.
6. Promote only after fixed comparison sets show no precision/recall regression, no additional absent-ball false positives and accepted recovery precision meets the action plan's 95% target. Otherwise retain an opt-in candidate with an identifiable manifest and rollback path.

### Adapter contract before any CNN fine-tuning

The authoritative inspected code is pinned in [the model manifest](models/blurball.manifest.json), with local files under `artifacts/blurball_research/source/src/`. The separate adapter implements this contract without mutating original annotations or patching the vendored research source. The following rules also govern any future trainer/augmentation changes.

1. **Read audited windows directly.** Consume the export's three chronological original JPEG paths, per-frame labels, dimensions, timestamps, source IDs, session/split and revision/image hashes. Revalidate them at load time. Yield a three-frame sequence with three corresponding targets; retain source provenance as a sidecar. The upstream `datasets/tabletennis.py` CSV loader expects case-sensitive `Frame,Visibility,X,Y,l,theta`, contiguous zero-based annotation indexing and generated PNG names. Feeding sparse source IDs such as 1685 directly into that loader, or joining separated bursts into one clip, would misalign or invent temporal context. A direct JSONL sequence adapter avoids that assumption.
2. **Transform both endpoints, once.** Apply the same source-to-output affine transform used for the image to the reviewed endpoints, then derive output-map midpoint, half-length and undirected angle. Preserve source endpoints/dimensions and original derived fields. Review `axis_rad` is radians; upstream line heatmaps require **degrees**. Review `half_length_px` is a source-pixel half-length; native `ball_blur.extent_px` is a **full** source-pixel length. Neither is automatically the output-map half-length. The current upstream `dataloaders/dataset_loader.py` transforms the center but passes length unchanged: blindly forwarding source labels is unsafe. Prevent a second center transform in the integration.
3. **Preserve observation semantics.** A reviewed sharp point has zero half-length and an unresolved/null axis in provenance; an angle-zero numeric placeholder is permissible only inside the zero-length target format. Confirmed absence generates a blank target. Unknown/occluded annotations generate no supervised target or window. Initially exclude truncated partial streaks unless their center-target convention has been reviewed; never replace them with guessed hidden endpoints.
4. **Test tiny targets before training.** In the inspected `utils/heatmap.py`, the binary line generator loops over `range(int(l))` for positive lengths, producing an empty map when `0 < l < 1` output pixel. A visible tiny streak must not become an absence target. Choose and version a tested target generator/rasterization policy. Tests must cover sharp, fractional, horizontal/vertical/diagonal and edge-clipped streaks, non-square dimensions, affine scale/rotation, absence, unknowns and inverse coordinate recovery. Render representative input/target pairs after every target-changing augmentation.
5. **Choose loss and splits explicitly.** The paper describes quality focal loss; inspected `configs/train_blur.yaml` selects MSE, while `configs/loss/hm_qfl.yaml` defines the quality-focal alternative. Record the actual loss, heatmap policy, resolution, normalization, output channel ordering, seed and augmentation settings. Validate unique dataset IDs and disjoint sessions instead of blindly copying the upstream split list. Preserve checkpoint and configuration hashes; do not claim a paper reproduction from the default entrypoint alone.
6. **Freeze evaluation before fitting.** Review this session for training/development, obtain independent validation and test sessions, then freeze labels/subsets, strict localization tolerance, thresholds-selection policy and baseline outputs. Audit full image hashes across splits. The exporter reporting windows in all splits only checks structural readiness; sample sufficiency, adapter target correctness and blind evaluation remain separate gates. Native causal validation, actual negative-frame diversity and unresolved cross-version preprocessing parity must be addressed before promotion.

### Implemented adapter, commands and evidence

[python_tools/blurball_reviewed_dataset.py](python_tools/blurball_reviewed_dataset.py) provides a read-only audit and PyTorch-compatible map-style dataset. It rebuilds the handoff from the original capture/index/annotations, compares every exported window and lineage hash, checks complete declared frame counts and globally increasing one-based source IDs/timestamps, and rejects stale annotation IDs/times. It checks whole-session and original-recording separation plus both byte-level and decoded-RGB duplicate images across splits. Split/export/source changes are detected again when loading samples. No frames are reindexed, padded or invented. A missing independent split or no reviewed windows causes an error.

The tensor path is chronological RGB CHW `[9,288,512]`, with the exact inspected zero-rotation/max-dimension affine, OpenCV `INTER_LINEAR`/black border and ImageNet normalization. All three output maps are `[3,288,512]` at scale key `0`. Fully visible reviewed streaks and explicit absence are eligible. Partial streaks remain excluded pending a truncation-label convention. A positive center cropped out of the model image is rejected, never converted to a blank negative; normalized endpoint `1` is an image boundary and a sharp center there is not a valid pixel.

`continuous_binary_capsule_v1` rasterizes the exact transformed segment with radius **2.5 output pixels**, preserving the reviewed center/half-length/undirected axis. Sharp zero-length points keep the upstream **3-pixel radius**. The positive target uses distance to the complete continuous segment rather than upstream `range(int(l))` sampling, so a fractional positive half-length cannot disappear. This is an explicitly versioned **target-policy change**, not bitwise parity with positive upstream targets or proof of better trained accuracy. Sharp-target parity and fractional-regression behavior are tested; original label extent is unchanged.

Read-only audit after reviewed-session export (paths below are the future real handoff, not existing accepted live data):

```powershell
python python_tools/blurball_reviewed_dataset.py audit artifacts/streak_training_v1 training_data/live artifacts/streak_splits.json --loss quality_focal
```

The loss is mandatory: `quality_focal` selects the paper-described beta-2 heatmap configuration; `mse` explicitly selects the upstream entrypoint's alternative. The audit records the chosen settings. It neither loads a checkpoint nor starts an optimizer.

Python interface, with `python_tools` on the import path:

```python
from blurball_reviewed_dataset import ReviewedBlurBallDataset

dataset = ReviewedBlurBallDataset(
    export_dir, captures_dir, split_file, "train",
    loss_choice="quality_focal", as_torch=True,
)
inputs, heatmaps = dataset[0]            # [9,288,512], {0: [3,288,512]}
inputs, heatmaps, provenance = dataset.sample(0)
```

`__getitem__` returns the upstream training tuple shape and works with PyTorch's `DataLoader`; `sample` additionally returns source geometry, transformed geometry, source-frame IDs/timestamps/hashes and session/label-method provenance. The constructor audits all three splits before exposing one. No stochastic augmentation or automatic pseudo-labeling is performed. It currently rechecks source metadata hashes on each sample for integrity; loading throughput has not been benchmarked or optimized for a large training run.

Evidence: **11 focused adapter tests + 4 existing streak regressions + 16 independently written integrity tests = 31 passed**, in [the combined runtime log](artifacts/blurball_dataset_adapter_tests.log). Independent math/input checks matched upstream affine construction and equivalent PyTorch normalization bitwise, including portrait/4:3/odd-size images, endpoint reversal, chronological channel order, geometry units and tiny-positive/absence cases. The installed runtime lacks usable `torchvision.transforms`, so these checks do **not** claim execution of the complete original torchvision/Hydra training runner. That environment and full-model forward/loss/backward smoke must pass before training.

Isolated synthetic fixtures and their [adapter audit](artifacts/blurball_dataset_adapter_v1/adapter_audit.json), [read-only CLI result](artifacts/blurball_dataset_adapter_readonly_audit.log), [sharp preview](artifacts/blurball_dataset_adapter_v1/preview_sharp.png), [tiny-positive preview](artifacts/blurball_dataset_adapter_v1/preview_tiny_positive.png), [diagonal preview](artifacts/blurball_dataset_adapter_v1/preview_diagonal.png) and [absence preview](artifacts/blurball_dataset_adapter_v1/preview_absent.png) are retained. All previews were inspected visually. They are **synthetic structural tests**, not real gameplay labels or evidence of sufficient training data. New fixture-only previews can be generated without reading the actual captures:

```powershell
python python_tools/blurball_reviewed_dataset.py synthetic-preview artifacts/NEW_synthetic_preview --loss quality_focal
```

The destination must not already exist. No local fine-tuning or weight promotion is started by these tools. The exact next training step is: review consecutive visible/absent examples from this session; collect and review independent validation/test sessions; freeze/audit/export their manifests; verify the complete trainer and target-policy configuration; then run one reproducible candidate fine-tuning experiment and the native causal promotion comparisons. Present phone labels and one-session footage do not meet those prerequisites.

## Stage 3 — Train a neural forecast only with valid future targets

1. Build a forecast evaluation dataset from reviewed, complete consecutive source intervals. Inputs end at time `t`; target frames are strictly later. Keep source timestamps. No target coordinates may be generated by the same forecast that is being evaluated.
2. Define separate short horizons, including the current baseline's 50 ms maximum. Match exact source times or document interpolation of **reference observations**; do not use playback wall time or frontend rendering cadence.
3. Evaluate held-out measured image-plane targets first. Split by complete recording/session and report contacts separately. Keep every future target unavailable to the causal input/model selection pipeline.
4. Compare last-position, constant-velocity, contact-segmented quadratic and uncertainty-propagating tracker baselines. Then test a compact LSTM or temporal feature model with a mixture-density head. Useful input features include position/time deltas, state, blur half-length and undirected-axis encodings; detector score remains an uncalibrated feature unless separately calibrated.
5. Model mixture components as alternative outcomes, not guaranteed event labels. Do not call a component “bounce” without labeled contact evidence. Do not invent physical height, velocity or RPM to complete a feature vector.
6. Evaluate ADE/FDE by horizon, contact timing, prediction expiration, coverage/region size, missing/absent behavior and native runtime. Require a meaningful improvement over the strongest baseline and the action plan's empirical coverage gate before enabling a neural predictor by default.

The earlier [272-row recording forecast regression](artifacts/blurball_research/imageplane_forecast/metrics.json) is **noncausal**: recording BlurBall observations may already contain the future target frames. Past-only fitting IDs do not remove this detector lookahead. Its eligible +1/+2-frame mean errors are 2.586/5.372 px versus constant velocity 3.008/6.466 px, with only 71.0%/70.1% target coverage; combined p95 is slightly worse, 17.309 versus 17.155 px. This experiment does not demonstrate unseen-future skill or supply neural ground truth.

A separate [complete native causal journal](artifacts/blurball_research/imageplane_forecast_causal/frames.jsonl) and [exact-frame evaluation](artifacts/blurball_research/imageplane_forecast_causal/metrics.json) are now available for the same official clip: zero detector lookahead, channel 2 only, both warmup rows retained, all 272 source rows, 260 observed frames and 199 fits/73 abstentions. The +1/+2-frame paired errors are quadratic MAE **3.118/6.492 px** versus constant velocity **3.389/7.161 px**, with target coverage **73.2%/72.4%**. Across 391 targets, quadratic MAE is lower (**4.792 versus 5.261 px**), but RMSE is worse (**10.863 versus 10.245 px**). This is a causal single-clip regression result, not an improvement on every error measure or blind venue/live acceptance. There is no contact ground truth or calibrated uncertainty coverage; the forecast remains off by default and unpromoted.

Independently reviewed held-out-session future targets and contact labels are still needed for neural forecasting. Imported phone PTS are available, but the application's native nominal-FPS MOV timestamps have not passed true-PTS/seek integrity; resolve that before variable-rate forecasting acceptance.

The *Fractals* LSTM/mixture method is a useful research design, not a ready-to-deploy physical predictor for this camera. Synthetic flights can supply augmentation or pretraining, but independently labeled real footage decides promotion.

## Stage 4 — Integrate defensible physical inference

- Obtain camera intrinsics and independent table/net landmarks. A stable table homography is still a plane mapping; it does not resolve an elevated ball's 3D position.
- Fit only identifiable flights, with contacts dividing segments. If geometry or observations cannot support physical height/spin, return null with a reason and retain synchronized image/plane views.
- Propagate observation/calibration uncertainty. Validate nominal 95% regions against sufficiently populated measured, predicted and reacquired reference subsets; report both localization error and region size.
- For spin, independently validate resolved surface-feature rotation. A blur axis estimates exposure motion, not ball rotation. Any trajectory-based spin fit must demonstrate identifiability and uncertainty against independent physical reference before displaying RPM.

## Release evidence

| Area | Required record |
|---|---|
| Review/export | Explicit save, clear/reload, sharp-point null axis, source dimensions, legacy boxes unchanged, exact-origin checks, unknown/occluded exclusion, session and duplicate gates. |
| Detection | Native/Python/export parity; source-indexed fixed labels and thresholds; separate precision/recall/localization and absent-ball false positives, with blur/lighting/contact subsets. |
| Forecast | Causal inputs, valid independent future targets, no crosses-contact fits, bounded missing-ball lifetime, ADE/FDE by horizon, coverage and region size, baseline comparison. |
| Physical analytics | Independent geometric reference, fitting-window/event provenance, explicit unknowns and spin observability. |
| Operations | Complete recording coverage unaffected by playback controls; native sustained run; end-to-end source-to-display latency separate from offline multi-pass GPU benchmark. |

Every candidate stores source/model hashes, settings, review lineage, evaluation protocol and result files. A feature that runs but fails these gates remains experimental. Missing independent live sessions and valid physical/forecast labels remain dependencies; this plan does not count them as passed tests.
