# Hybrid and temporal-model implementation — October 10, 2026

The first implementation milestone is executable and tested. It is research code,
not an improved trained detector release. The hybrid fixes its earlier recall
failure, but fails venue-negative acceptance and is **not promoted**.

## Implemented

- Native `ball_candidate_selection_v2.hpp` selects an existing causal BlurBall
  position even when YOLO is weak, ambiguous or disagrees. Comparable distant
  BlurBall components still cause abstention. High-score, unambiguous YOLO
  fallback is explicit and can be disabled. Original positions, scores and
  provenance remain intact; raw scores are not correctness probabilities.
- The paired native evaluator accepts `--selector-revision v2`; its default
  remains v1. Versioned configuration hashes and frame/model identities are
  independently recomputed by the Python journal auditor. The archived evaluator
  executable is supplied to scoring for verification of its actual bytes.
- `temporal_ball_student.py` reuses actual pretrained YOLO11n encoder blocks
  0–4, preserving stride-4/8 features. It fuses t−2/t−1/t features and returns
  a current-frame heatmap and visibility logits. New heads start untrained.
  Causal source/frame/time/geometry resets and conditional, aligned training-only
  teacher supervision are implemented. This is a new model architecture, not a
  merge of YOLO and BlurBall checkpoints.
- The standalone YOLO experiment runner prepares frozen 640, unfrozen 640 and
  higher-detail 960 candidates with isolated settings and provenance. Its actual
  reviewed-data plan refuses fitting until independent splits and checked
  preparation exist. Higher resolution also needs native export/parser support.
  Training uses new private train/validation copies, excludes test copies, and
  verifies targets after fitting. Source label caches are not inherited; AMP's
  automatic model-download check is disabled for these initial fixed experiments.

## Native accuracy evidence

Both native models ran on every actual frame of the labeled 272-frame development
clip and the 9,164-frame mixed venue recording. No camera, lookahead, image filter,
source mutation or analytical frame skipping was used.

| Model/policy | Development precision | Development recall | False detections / 284 reviewed absent venue frames |
|---|---:|---:|---:|
| Current YOLO | 4.41% | 4.44% | 42 |
| Causal BlurBall | 97.69% | 94.07% | 71 |
| Earlier hybrid v1, historical replay | 100% | 1.85% | 0 |
| Experimental primary hybrid v2 | 97.69% | 94.07% | 75 |

Development localization requires **strictly less than four source pixels**;
all 270 visible references remain in the full-recording recall denominator,
including two causal warmup frames. V2 matches BlurBall on this clip rather than
improving it. It still produces more reviewed-negative false detections than
either baseline. A no-regression gate therefore fails.

The net four extra venue false positives comprise six new BlurBall selections
that the standalone 300-pixel anchor heuristic had rejected, offset by two
ambiguity abstentions. No YOLO fallback was selected in the complete venue run;
disabling that fallback alone cannot fix this observed regression. Thresholds
were not retuned after observing these references.

The development clip has unknown checkpoint-training lineage. The venue subset
mixes uniform and hard-mined, correlated samples from one session; overlaps are
deduplicated and positives/unknowns are excluded from absence scoring. These
numbers are not newly blind accuracy, complete venue precision/recall, or a
whole-video absence rate. Unsampled venue frames remain unknown.

Evidence:

- [Development metrics](artifacts/paired_ball_official_22_003_primary_v2/metrics.json)
- [Venue reviewed-negative metrics](artifacts/hybrid_v2_implementation_2026_10_10_v1/reviewed_negative_metrics.json)
- [Complete venue native journal](artifacts/paired_ball_mixed_venue_primary_v2/diagnostic_summary.json)
- [Archived evaluator and logs](artifacts/hybrid_v2_implementation_2026_10_10_v1/)

## Training implementation checks

The temporal student passes 15 CPU contract/provenance tests and two real reviewed
optimizer steps: ball-present source frames 14960–14962 and explicitly absent
frames 1–3. All 12 trainable fusion/head tensors receive finite gradients and
change. The frozen encoder and BatchNorm statistics, original checkpoint,
reviewed staging snapshot and imported helper/source hashes are reverified.

These checks establish the executable optimization path, not convergence,
accuracy or a saved usable student. No actual teacher or saved student checkpoint
was used. Current staging contains only three fully visible windows and 175
fully absent windows, all from one unassigned session.

Native verification passes 25 legacy selector, 24 v2 selector and nine CLI
checks. Python journal/review regression verification passes 52 tests.
The hardened standalone YOLO runner passes 15 CPU control/provenance tests,
including stale-label-cache isolation and rejection of changed private targets.

The replays are offline sequential native evaluations. Their stage timings are
not a repeatable multi-pass GPU benchmark or single-session live capture-to-display
latency measurement. No live performance claim follows from these runs.

## Next fitting and integration gates

Keep existing completed annotations. Add representative consecutive positive
windows and independently recorded validation/test sessions; resolve conflicting
duplicate box reviews without erasing valid positives. Freeze audited splits and
prepare data, then run the isolated YOLO variants and a class-balanced temporal
student candidate. Evaluate on matched references, confirmed absences and blur/
occlusion subsets before export and native parity testing.

Application defaults and deployed models remain unchanged. Reliability-gate
training, full student fitting/distillation, TensorRT student export, application
integration and the broader measurement/calibration/event plan remain unfinished.

Instructions: [YOLO experiments](python_tools/YOLO_BALL_EXPERIMENTS.md),
[temporal student](TEMPORAL_STUDENT_PROTOTYPE.md),
[requirement ledger](REQUIREMENTS_LEDGER.md).
