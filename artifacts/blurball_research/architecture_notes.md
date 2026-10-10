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

## Existing native analyses of the user's downloaded clips

Read-only audit of complete saved analyses; these are proposal coverage, **not localization accuracy**:

| Saved source / analysis | Accounted frames | Positive ball proposal | Table box frames | Four-corner frames | Max speed / strikes |
|---|---:|---:|---:|---:|---|
| `001.mp4`, `1791526513809868_28444` | 193/193 | 112 (58.03%) | 3 | 1 | 0 km/h / 0 |
| `demo5.mp4`, `1791528592320735_39620` | 106/106 | 88 (83.02%) | 1 | 0 | 0 km/h / 0 |

Both completion manifests say COMPLETE and every `video_error` is empty. Both used deployed ball SHA256 `2f0b1c61d2b8e9ad9cc08d56ccce467ec24be3bc87d63e28c4896055b1a8efd9`, threshold 0.25 and experimental intelligence false. Thus these recordings decoded and generated many ball hypotheses, while physical analysis was unavailable due to almost entirely missing table calibration. Candidate score ranges were 0.2517–0.7891 and 0.2629–0.6152 respectively. Both final strikes blocks are empty. This does not establish whether any individual proposal was the true ball.

## Accessible official reference annotations and evaluator conventions

Read-only public WebDAV directory listing at `https://cloud.cs.uni-tuebingen.de/public.php/webdav/`, with public share token `C3pJEPKWQAkono7`, exposes match directories 00–25 plus small midpoint and endpoint annotation archives. The midpoint archive is 1,591,844 bytes, SHA256 `74ea49d3ce1a27f09edf1e3525348ac717e50f8f024f226baec941026237faf7`; it was inspected in memory without downloading the full dataset. It contains 465 entries and CSVs such as `all_csv_annotations/22_csv_003.csv`.

The source config lists matches 22–25 as test. `22/rallies_videos/003.mp4` is only 2,058,437 bytes, making a small official labeled comparison feasible. Its CSV contains 272 frames: 270 Visibility=1, two Visibility=0. Header is `Frame,Visibility,X,Y,theta,l`.

`src/datasets/tabletennis.py:16–46` loads X/Y directly and marks only Visibility=1 visible. There is no explicit unknown/occluded annotation class in that loader; do not claim such evaluation subsets from these two codes alone. Predicted positions are inverse-affine transformed into the source image before the evaluator receives them. Reference coordinates in the inspected CSV reach x=980.54; another match-22 CSV reaches x=1275.3, so the CSVs are not 512-pixel input coordinates. Check actual reference video dimensions rather than scaling labels by assumption.

`src/configs/runner/eval_blurball.yaml:14` sets localization threshold **4 source pixels**, with a strict `<` comparison in `src/utils/blur_evaluator.py:38–39`. The runner's default function argument `dist_thresh=10` is only used for a separate false-positive-list condition; it is not the evaluator's acceptance threshold. Official recall uses TP/(TP+FN), excluding incorrectly localized visible predictions counted as FP1. For this project's fixed detection gates, report recall TP/(TP+FN+FP1), with FP1 contributing a false positive and a missed true location, or display both conventions explicitly. Otherwise incorrectly localized predictions can falsely inflate reported recall. The official RMSE accumulates errors for all visible frames with a prediction, including misplaced proposals.

Two absent frames are insufficient to establish absent-ball false-positive safety; this labeled subset is exploratory evidence, not a full model promotion gate. CSV theta/l are consumed without length rescaling when generating targets (`dataset_loader.py:183–197`), while XY is affine transformed. Verify the annotation's blur-length convention independently before treating exported source-scaled blur length as directly comparable to CSV `l`.

The public frame extraction script uses `ffmpeg -i "$video" -start_number 0 -vsync 0 -an -y .../%05d.png`. It does not resize, convert FPS or remove duplicates. Source frame 0 corresponds to the first sequentially decoded frame. A CSV/video length discrepancy must remain a reported alignment failure rather than being repaired through an assumed cadence.

## Verified isolated ONNX / TensorRT export

The earlier export feasibility inference was subsequently tested using `python_tools/export_blurball_candidate.py`, without modifying any deployed model. See `artifacts/blurball_research/export/report.json` and `export_runtime.log`.

- Installed Torch 2.14.0+cu130, TensorRT 11.3.0.99, ONNX 1.22.0; NVIDIA GeForce RTX 5050 Laptop GPU.
- Checkpoint SHA256 `3545206c7155194ea654899d33579c88c9fd8e82c632cbdbae3b0c0ec3f2985f` loaded with `weights_only=True` and strict parameter matching.
- Fixed-shape opset-17 ONNX export and ONNX checker passed. TensorRT parser produced no errors, and a strongly typed FP32 engine with TF32 disabled built successfully in 27.49 seconds.
- Engine contract `[1,9,288,512]` FP32 → `[1,3,288,512]` FP32; SHA256 `48a3c3910b87552efd6cf2475fc70f1fd770db46d73fc30fe605b051d25faf0f`.
- Tested 18 real consecutive-frame windows from the user's original `001.mp4` and `demo5.mp4`: three overlapping windows around source frame 3, 40 and 80 per source. Maximum logit difference 0.0000391; maximum heatmap probability difference 0.00000174. All 51 matched connected components agreed, with zero unmatched components and maximum centroid difference 0.000209 source pixels.
- Offline GPU forward timing across those 18 passes: Torch median 25.38 ms, TensorRT FP32 median 9.85 ms and p95 10.02 ms. These exclude capture, RGB/affine preprocessing, transfer, component extraction, other models, telemetry and rendering. Existing application/research work may share the GPU; this is export experiment timing, not a controlled sustained benchmark or live latency result. No 120 FPS claim follows from these measurements.

Numerical export parity therefore passed this finite experiment. Detector accuracy, native C++ integration, prolonged stability, FP16 conversion and live performance remain separate acceptance work. The engine cannot replace the YOLO engine file directly: its temporal input and heatmap output contracts require a dedicated adapter and source-frame mapping.
