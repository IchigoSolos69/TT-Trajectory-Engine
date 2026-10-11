# Improving YOLO and combining it with BlurBall — October 10, 2026

Implementation update: [the first native/model/runner milestone is tested](HYBRID_V2_IMPLEMENTATION_REPORT.md).
V2 fixes the old selector's development recall collapse but fails the venue
negative gate. The causal YOLO-feature student remains an unfitted prototype;
standalone fitting still awaits independent data. The evidence below records the
starting baseline, not the newest selector's results.

Both systems are convolutional neural networks. The useful combination is an
efficient box/scene detector, temporal dense ball localization and a trained
decision or shared-feature layer. Adding a component named "NN" alone does not
establish an accuracy gain. The steps below are experiments, not released models.

## Evidence that determines the design

- The current YOLO ball engine accepts one RGB frame at 640×640. A 1920×1080
  frame occupies 640×360 inside its letterbox. It was initialized from yolo11n
  and trained using game_1/game_2: 9,226 train and 3,377 validation images, about
  125 explicit negatives across these partitions. Training labels are fixed
  30-source-pixel squares around annotated centers. Domain coverage and target
  geometry are plausible improvement areas, not proven causes of every miss.
- BlurBall accepts three RGB frames at 512×288 and returns dense heatmaps. Its
  HRNet-based network preserves spatial features and fuses multiple resolutions.
  Its input is smaller than YOLO's actual 640×360 image; its advantage must not
  be described as simply higher input resolution.
- In the fixed official 272-frame development comparison at strict <4 source
  pixels, causal BlurBall precision/recall are 97.69%/94.07%, current YOLO
  4.41%/4.44%, and the first conservative selector 100%/1.85%. The selector's
  agreement/ambiguity rules discard useful BlurBall observations. It fails
  promotion; high precision with almost no recall is not a useful hybrid.
- The supplied screen recording uses different source frames for the models,
  and explicitly labels BlurBall **two-frame lookahead**. It is qualitative
  recording-mode evidence, not a fair causal/live accuracy benchmark. Both
  modes lack table calibration, so better detection alone cannot establish
  physical speeds or synchronized physical 3D reconstruction.

Sources: [native preprocessing](backend_cpp/src/tensorrt_engine.cpp),
[training/labels](python_tools/train_multigame.py),
[training split audit](artifacts/multigame/data_audit.json),
[BlurBall config](artifacts/blurball_research/source/src/configs/model/blurball.yaml),
[paired results](artifacts/paired_ball_official_22_003_v1/metrics_with_runtime_provenance.json),
[screen-recording inspection](artifacts/user_hybrid_video_2026_10_10_v1/visual_inspection.md).

## 1. Improve the standalone YOLO ball model

Keep unchanged YOLO and causal BlurBall as fixed baselines. Keep scene detection
separate; do not force a specialized ball checkpoint to learn all person/table
classes from unchecked labels.

| Experiment | Change | What it tests |
|---|---|---|
| Data only | Reviewed venue positives and representative/hard absent backgrounds, retaining match examples | Domain adaptation and background rejection |
| Target geometry | Compare the existing versioned center-square targets with reviewed ball/streak extent boxes | Whether annotation convention affects localization |
| Augmentation | Target-preserving exposure, compression and motion blur, with blur centers/extents adjusted consistently | Bright/blur robustness, without arbitrary center shifts |
| Backbone adaptation | Compare current frozen-layer fine-tuning with an explicit unfrozen candidate | Whether venue features need adaptation |
| Resolution | Full-frame 640 baseline versus higher-resolution candidates | Tiny-ball detail versus actual compute/memory cost |
| Crops | Bounded causal tracker-guided high-detail crops, retaining broad search and reacquisition | Detail without losing balls outside a stale ROI |
| Small-object head | Experimental finer-resolution feature head | Whether spatial feature preservation helps beyond data alone |

Run one change at a time before combining passing changes. Use the same frozen
held-out references and tolerance. Report ball size, brightness, blur, occlusion
and absent-frame subsets separately. Do not choose thresholds on test footage.

Higher input sizes or a new head require native work: the current exporter
rejects non-640 shapes, and native initialization/postprocessing assumes 640 and
8,400 predictions. Do not merely set a GUI option and claim the deployed engine
changed. Every exported candidate needs inspected I/O shapes, correct buffers,
coordinate mapping and native/PyTorch parity before application integration.

## 2. Build a hybrid that preserves the strongest branch

Use BlurBall as the initial primary ball-localization baseline. Improved YOLO
provides complementary observations, disagreement diagnostics and independent
fallback candidates. Scene YOLO supplies people/table observations separately.

The C++ decision layer must select an existing supported observation or abstain.
Weak or ambiguous YOLO evidence must not automatically veto valid BlurBall points.
Do not average raw YOLO scores with heatmap peaks or blend distant coordinates.
Strong contradictions can reduce trust, but rejection needs validated motion,
event, visibility and scene evidence, including contact-aware continuity.

After collecting matched reviewed cases, train a small reliability gate with
targets such as choose YOLO / choose BlurBall / neither. Inputs can include
within-model evidence, candidate agreement, causal motion residual, blur,
occlusion evidence and track age. Training and validation establish calibration;
unknown annotations never become a "neither" label. A neural gate is an optional
experiment, not inherently better than auditable rules.

Evaluate the hybrid against the best standalone model, including accepted-point
precision, conventional recall and false detections on confirmed absences. A
hybrid that merely abstains everywhere fails. Enforce causal three-frame support
for live evaluation, distinct from optional recording lookahead. One C++ selected
sample, covariance and provenance drive all views.

## 3. Train a single temporal student after the teachers pass

Proposed research architecture:

```text
Three consecutive RGB frames
        ↓
YOLO-style spatial encoder + causal temporal feature fusion
        ↓
Dense ball-position / visibility head
        + optional supervised blur-extent / orientation heads
        ↓
C++ tracking, uncertainty and contact-aware physics validation
```

This combines useful architectural properties; it is not a concatenation or
average of checkpoint weights. Use real reviewed targets as supervision. A
validated BlurBall or hybrid teacher can supply an auxiliary distillation signal
on training data. Teacher output is never independent test ground truth or an
automatically reviewed human label. Compare supervised-only against distilled
students, then check TensorRT parity and causal accuracy. A one-frame student
cannot be assumed to recover all information available to a temporal teacher.

## Current data and release limits

There are 130 eligible detector-positive and 293 reviewed absent frames, with
eight visible temporal targets and only three fully visible temporal windows.
All footage belongs to one unassigned real session. The staged real-data
optimizer checks prove the network can consume reviewed examples, not that a
representative candidate is trained or accurate.

More diverse positive windows, temporal-continuity checks and independently
recorded validation/test sessions remain required. Distinguish source video from
screen recordings with overlays; the new supplied screen recording was not
ingested into training. Preserve original models and rollback manifests. No
training, default change, hybrid promotion, performance claim or whole-plan
completion follows from this document.

Primary references: [BlurBall authors' repository](https://github.com/cogsys-tuebingen/BlurBall),
[Ultralytics augmentation guidance](https://docs.ultralytics.com/guides/yolo-data-augmentation/),
[Ultralytics distillation guidance](https://docs.ultralytics.com/guides/knowledge-distillation/).
