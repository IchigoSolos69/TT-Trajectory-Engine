# Causal temporal ball student prototype

Implementation date: 2026-10-10. This is executable research code; it has not
been trained to convergence, exported to TensorRT, measured for accuracy or
latency, added to the application, or promoted as a deployed detector.

The implementation reuses **actual locally installed Ultralytics YOLO11n
blocks 0 through 4** from `yolo11n.pt`: Conv, Conv, C3k2, Conv, C3k2. These
provide stride-4 and stride-8 spatial features. It does not describe a generic
handwritten convolution stack as YOLO, nor splice together unrelated saved
checkpoint tensors. The detection neck/head and pretrained box outputs are
not reused. New ordered temporal fusion, heatmap and visibility heads start
with random initialization.

## Network and causal data contract

`python_tools/temporal_ball_student.py` exposes `TemporalBallStudent`:

```text
t-2, t-1, t RGB frames [B,3,3,H,W], values 0..1
             │
shared pretrained YOLO11n early spatial encoder
             │
stride-4 / stride-8 features in explicit chronological order
             │
ordered concatenation + learned convolutional temporal fusion
             │
latest-frame-only heatmap logits [B,1,H,W]
latest-frame-only visibility logits [B,1]
```

Spatial dimensions must be divisible by 32. The reviewed adapter retains the
existing exact 512×288 max-dimension affine geometry and reverses its ImageNet
normalization to recover RGB 0..1 for the actual YOLO encoder. It uses the
**latest reviewed target**; historical-frame labels do not become the current
target. The heatmap is a localization support target, including a reviewed
blur capsule where present, rather than a calibrated position distribution.
Visibility logits/sigmoid outputs are not calibrated probabilities.

`CausalWindow` accepts one frame at a time and emits only complete t−2/t−1/t
windows. It resets on a source change, seek, frame-ID gap, non-increasing source
timestamp, excessive time gap or changed image geometry/device/dtype. Its
timestamp unit is source seconds and its explicit gap limit defaults to
100 ms. No lookahead frames are accepted. Pixel identity alone is not proof
of a scene cut: actual static frames can repeat. The strict reviewed staging
dataset independently excludes identical decoded review images.

In the initial smoke the spatial encoder is frozen, including BatchNorm running
statistics; only temporal fusion and the two heads update. An explicit
`freeze_encoder=False` option exists for a later controlled experiment. Frozen
weights are not evidence that venue adaptation has occurred.

## Supervision and optional distillation

The supervised loss combines per-sample positive/background-balanced quality
focal localization loss and binary visibility BCE. A reviewed absent frame has
a blank heatmap and visibility zero. A reviewed visible frame must retain a
nonempty target. Unknown/partial/hidden targets are not converted to negatives.

Optional teacher signals require detached probability tensors, a model SHA256,
one source identity and a consecutive source-frame triple per batch item,
identical geometry and causal offsets `[-2,-1,0]`. Future-frame/lookahead,
wrong-source, wrong-frame and wrong-geometry signals fail validation. Teacher
signals are accepted only as a separately weighted **training auxiliary**; a
validation/test call cannot use them. Ground truth remains reviewed supervision.
No real teacher was used in the initial optimizer smoke. This interface checks
alignment and tensor validity; it does not establish teacher accuracy or
license using teacher predictions as labels.

## Verified executable checks

CPU contract tests:

```powershell
python -B python_tools/test_temporal_ball_student.py
```

The final revision passed 15 tests. They exercise actual local pretrained blocks,
output geometry, history/order sensitivity, frozen encoder/BatchNorm integrity,
finite connected head gradients, positive/absent supervision, normalization,
current-frame target selection, gap/source/seek resets, no future dependency
in the stream adapter, aligned/distinct causal teacher guards, and rejection of
checkpoint mutation during model loading using a disposable checkpoint copy.

Evidence: `artifacts/temporal_ball_student_tests_2026_10_10_v3.log`.

Two real reviewed in-memory optimizer steps:

```powershell
python -B python_tools/smoke_temporal_ball_student.py `
  artifacts/reviewed_training_stage_2026_10_10_v1 `
  --checkpoint yolo11n.pt --device cpu `
  --output artifacts/temporal_ball_student_real_smoke_NEW
```

The strict `StagedReviewedDataset` audits and hashes original footage, JPEGs,
reviewed annotations, staging code and snapshot before loading and after the
optimizer. The diagnostic selects the first fully visible and first fully absent
triples deterministically. It saves only a report; candidate weights, source
media, labels, splits and deployed models remain unchanged. The final smoke
also checks prototype/smoke source hashes, the imported reviewed-example selector
source hash and all frozen encoder weights/stats. The original checkpoint digest
is captured before loading, compared immediately after loading and retained for
the final unchanged-checkpoint test.

Evidence: `artifacts/temporal_ball_student_real_smoke_2026_10_10_v3/report.json`.
The visible triple is source frames 14960–14962 in IMG_7944.MOV's reviewed
capture; the absent triple is source frames 1–3 in the mixed Video Project 6.mp4
capture. These are reviewed training-environment checks, not accuracy tests or
independent validation. The stage has 178 structural windows: only 3 fully
visible and 175 fully absent, all from the same unassigned session.

## Remaining work before a candidate can be evaluated

1. Expand reviewed causal positive windows and independently held-out sessions;
   audit source cuts, repeated/paused frames and camera timestamps.
2. Freeze train/validation/test session manifests and test settings. Neither
   adjacent images from the same session nor teacher suggestions become blind
   ground truth.
3. Train a bounded research candidate with learning curves and class-balanced
   sampling; compare frozen versus adapted encoder and measured augmentation.
4. Compare native causal YOLO, BlurBall, selector and student outputs on exactly
   the same frames. Include localization, precision/recall, false positives on
   confirmed absences, blur/occlusion subsets and inference/transport timing.
5. Only after passing the agreed gates, implement/export the student contract
   and verify TensorRT parity, source coordinates, streaming continuity and
   application integration. Keep the deployed detector unchanged until then.

Two optimizer steps demonstrate that the implementation can learn from the
reviewed tensors; they do not demonstrate improved detection or a useful model.
