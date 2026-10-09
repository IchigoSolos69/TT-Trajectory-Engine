# BlurBall architecture and integration audit

Inspected official repository commit `2f0f5496f7ba4b5b1a36790749935121b2ce972d` on 2026-10-09. These are source findings, not claims of model accuracy or runtime performance. No deployed model or application code was changed during this audit.

## Model contract

- `src/configs/model/blurball.yaml:2–9` specifies three input frames, three output frames, 512 × 288 input and output, RGB differences disabled, scale 0 only.
- `src/models/blurball.py:328–350` concatenates the three color frames as channels; network input is `[N, 9, 288, 512]`. HRNet multiresolution branches use squeeze-and-excitation blocks (`:35–50`). The full-resolution output is `{0: logits}` with shape `[N, 3, 288, 512]` (`:414–425`, `:558–598`).
- The network has no separate blur-angle/length, visibility-probability or covariance head. Blur shape is encoded in the heatmap; the postprocessor derives weighted centroid, component orientation and PCA extent. `src/detectors/blurball_postprocessor.py:39–79` returns **half-length**, measured in heatmap pixels.
- `src/dataloaders/__init__.py:18–27` normalizes RGB pixels from [0,1] using ImageNet mean `[.485,.456,.406]` and standard deviation `[.229,.224,.225]`. `src/utils/utils.py:22–30` explicitly reads training/evaluation images as RGB. The dataset uses a consistent affine warp and its inverse (`src/dataloaders/dataset_loader.py:145–174`).

## Temporal causality and frame accounting

- Step 3 uses disjoint windows, emits all three frame outputs and may need two future frames before its first frame result is available. The stock video runner leaves a trailing 1–2 frames unprocessed when total length is not divisible by three (`src/runners/inference.py:105–126`). Videos shorter than three frames yield no predictions.
- Step 1 slides by one frame but still emits all three outputs, accumulating candidates for each source frame from overlapping windows (`:116–123`). An interior frame receives hypotheses from windows starting at t−2, t−1 and t; the last two use future frames. The published offline step-1 result must therefore not be described as causal live inference without buffering.
- A causal mode is possible: use `[t−2,t−1,t]` and consume only output channel 2 for frame t. This needs no future frame but differs from the published overlapping-window selection and requires independent accuracy/latency evaluation. The first two source frames must remain accounted for explicitly rather than silently disappear.
- Output candidates are accumulated, not averaged; `src/trackers/online_blur.py:11–35` selects the highest component score after a previous-position distance gate. It does not propagate uncertainty, recover occluded points or estimate physical trajectories.
- README states duplicate filtering is automatic. Actual `process_video(video_path, filter=False)` default and caller with no flag mean no duplicate filtering runs (`src/utils/preprocess.py:14,47–57`, `src/runners/inference.py:308`). If deduplication is added, retain the original frame IDs, PTS and duplicate-to-reference mapping; never shift analytics to a compressed sequence clock.

## Stock inference pitfalls to avoid

1. `src/runners/inference.py:90–105` feeds cv2 BGR to `ToPILImage` without converting to RGB. This differs from training/evaluation. Use the inspected RGB dataset preprocessing for a comparison.
2. Direct `Resize` in the video runner and inverse affine mapping agree for common 16:9 landscape input, but not generally. Preserve an exact forward/inverse transform for every source geometry.
3. The component score is a **sum** of heatmap probabilities, not a [0,1] detector confidence (`blurball_postprocessor.py:39`). The recommended 0.7 is a **pixel heatmap threshold**, not the existing YOLO detection threshold. Do not label component sums as confidence percentages.
4. The postprocessor inverse-transforms XY only, leaving angle/half-length in heatmap coordinates (`:119–126`). Transform blur endpoints into source pixels before showing them; an anisotropic transform changes both angle and length.
5. The postprocessor forces angle zero when `mu20 == mu02` even if `mu11 != 0` (`:45–50`), so diagonal components with equal axis moments can report the wrong orientation. Do not mistake this source convention for an independently measured blur vector.
6. `online_blur.py:12–15` uses removed NumPy 2 `np.Inf`; its `xy_pred` local is initialized to None and never assigned, so the motion-quality branch is never active. A minimal adapter can avoid these compatibility problems without replacing the production tracker.
7. The video runner's frame counter remains zero (`inference.py:58,104`) and exported preview is forced to 25 FPS (`:261`). Preserve source PTS/FPS and benchmark with synchronized GPU timing rather than these counters.
8. Stock checkpoint load is `torch.load` (`blurball_detector.py:76`). Prefer `weights_only=True` for the public checkpoint and record its hash; do not blindly install the old pinned training environment into the current CUDA/TensorRT environment.

## ONNX / TensorRT assessment

No ONNX/TensorRT exporter or native backend is supplied by the inspected repository. The model forward uses ordinary convolutions, batch normalization, ReLU, adds, SE global pooling/linear/sigmoid/multiply and nearest-neighbor resize. A fixed-shape export is plausible; a wrapper returning `model(x)[0]` avoids the dictionary output contract. This is an inference from inspected operators, **not verified export compatibility**. Export/parity tests must compare logits, component positions and source-frame mappings. Keep connected-component/PCA postprocessing on the CPU initially; it is not part of the Torch forward graph.

## What it contributes to this project

Useful candidates are multi-frame tiny-ball localization, blur-center labeling, streak orientation/extent evidence, and a supervised blur-aware dataset. It is a good isolated challenger or bounded recovery input to the existing C++ authoritative trajectory pipeline. It does not detect players/tables, calibrate cameras, reconstruct physical 3D, classify strikes or infer spin RPM by itself.

The streak axis is undirected (180° ambiguity); source-frame motion can orient it. Streak length is a loose pixel-speed cue only if actual exposure duration is known. Source FPS is not exposure duration. Heatmap spread and blur extent can inform an anisotropic measurement-noise model, but neither is calibrated uncertainty without empirical coverage tests. Use exposure/PTS-aware timestamps and avoid treating the blur-center location as an exact instantaneous position.

## Primary sources

- [Official source repository](https://github.com/cogsys-tuebingen/blurball)
- [Model config](https://github.com/cogsys-tuebingen/blurball/blob/main/src/configs/model/blurball.yaml)
- [Model forward](https://github.com/cogsys-tuebingen/blurball/blob/main/src/models/blurball.py)
- [Detector and postprocessor](https://github.com/cogsys-tuebingen/blurball/blob/main/src/detectors/blurball_postprocessor.py)
- [Video runner](https://github.com/cogsys-tuebingen/blurball/blob/main/src/runners/inference.py)
- [Dataset preprocessing](https://github.com/cogsys-tuebingen/blurball/blob/main/src/dataloaders/dataset_loader.py)
