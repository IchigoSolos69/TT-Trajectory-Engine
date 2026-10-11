# Next training actions — October 12, 2026

Improve standalone YOLO first, then evaluate its complementarity with causal
BlurBall. The experiment runner is implemented; adequate reviewed camera data
and independent validation remain the immediate dependencies.

## Completed today

The user confirmed that both reviewed Video Project 6 batches came entirely
from the original iPhone gameplay session. They are now registered as
`iphone_same_venue_session_1`, existing training data, device `iphone`.
No label, capture manifest, frame index, review image or annotation revision was
rewritten. All 868 files in those two capture trees passed before/after SHA
checks. Both assignments were saved together with an exact registry backup.

Evidence: `artifacts/phone_review_registration_2026_10_12_applied.json`.
Thirteen isolated registration safety tests passed. No fitting, deployment or
accuracy improvement followed from changing the device metadata.

The fresh [strict readiness plan](artifacts/yolo_paired_training_plan_2026_10_12_v2.json)
is bound to the updated registry. It verifies original-source resolution without
errors and leaves fitting disabled with 17 data/export blockers.

| Current eligible camera data | Visible | Verified absent |
|---|---:|---:|
| Existing iPhone training | 128 | 293 |
| PC-camera training | 0 | 0 |
| Independent iPhone validation | 0 | 0 |
| Independent PC-camera validation | 0 | 0 |

There are zero eligible positive windows from new session A, zero reviewed
validation rallies and zero full reviewed validation absence intervals. The audit
excludes seven conflicting unique records and two unknown-visibility records,
deduplicates six identical reviewed image copies, and does not read sealed-test
labels. These counts describe dataset readiness, not detector accuracy.

## Your next collection and review steps

1. Resolve the seven conflicting duplicate positive annotations in Video
   Project 6 and the two IMG_7944 boxes whose visibility remains unknown. Keep
   all completed labels; review only the identified disagreements/unknowns.
   The [focused review checklist](TRAINING_REVIEW_QUEUE_2026_10_12.md) identifies
   both batch counters for source frames 7297–7299 and 8383–8386, and IMG_7944
   review counters 11–12 (source frames 9974–9975).
2. Capture three genuinely separate gameplay sessions with **both** the iPhone
   and the actual PC camera: A for training, B for validation, C for sealed
   final testing. Keep all clips and simultaneous recordings from each session
   together. Existing iPhone footage remains existing training data even when
   its files are in validation/seal folders. World Cup compilations cannot
   substitute for an actual PC-camera session or independent camera testing.
3. Use fixed framing with the complete table visible, stable focus and exposure,
   and unchanged original video files. Retain bright-background conditions,
   small balls, fast rallies, blur, occlusion and different placements. Also
   record more than two minutes per device/session with the ball physically
   removed while players move. Hidden or uncertain balls are not negatives.
   iPhone 1080p/60 is the intended capture setting; use the PC camera's genuine
   supported rate and preserve source timing. Upsampling a 30 FPS source does
   not establish 60 FPS capture. Edited Video Project 6 is currently 30 FPS.
4. In the app's Training Review, assign the actual session/device, review ball
   boxes and precise center/streak targets, and explicitly save. People and
   table labels are not required for ball-only fitting. Use the burst strip to
   inspect adjacent frames and confirm continuity only after checking for cuts,
   repeated images and gaps. Reused boxes/model suggestions remain drafts.
5. Use Full original no-ball interval only for intervals inspected throughout
   and known to contain no visible, hidden or uncertain ball. Do not mark the
   whole Video Project 6 negative: it contains actual balls. Review a second
   pass over 10% of labels and resolve center differences above two pixels.

The agreed first-cycle camera targets are:

| Partition | Required reviewed material |
|---|---|
| Training | 2,000 visible-ball frames and 1,000 verified absences across both devices; 500 positive three-frame windows from new session A, with continuity and precise targets |
| Validation B | Per device: 1,000 visible frames, ten complete reviewed rallies and at least 120 seconds of verified original absence |
| Sealed C | Corresponding per-device visible/rally/absence coverage, collected separately; automated labels/results stay unread until models and thresholds are frozen |

These are readiness minimums, not a promise of the requested accuracy. Current
saved labels remain useful even though they do not meet all these requirements.

## What runs after the dataset is ready

Prepare a fresh checked export and immutable plan from current source locations.
Run five controlled 12-epoch YOLO11 pilots: reviewed data at 640 with frozen
layers, unfrozen backbone, 960 input, reviewed label extents, and camera-oriented
augmentation. Keep comparisons on the same data/seed/common batch and evaluate
match, iPhone and PC-camera footage separately. Select improvements on
validation, then run the final configuration with seeds 42/43/44.

Use the implemented [experiment workflow](python_tools/YOLO_BALL_EXPERIMENTS.md).
Do not run the train command on a blocked plan. Run candidates sequentially and
budget private data copies/checkpoints; C: has approximately 13 GB free after
cleanup, while D: has approximately 45 GB free. No files have been moved to D:.

When standalone YOLO passes validation, compare it with BlurBall on identical
causal frames. Train the learned selector to choose YOLO, BlurBall or neither,
preserving the selected position. Retain a hybrid only if complementary
detections improve results without worsening absence performance. A temporal
student remains a subsequent latency/accuracy experiment.

Freeze the chosen model, thresholds and settings before independent C testing.
Release requires at least 98% precision and 95% recall separately on each camera
domain, strict <4 px at 1080p equivalent, the agreed false-detection gates,
native export/preprocessing parity, and real live latency/soak evidence.
Predictions do not count as detected balls. The current 960 export still has a
coordinate-parity failure to resolve before native promotion.

No models were trained or promoted today. Earlier model results and frozen plans
remain historical evidence; changing registry metadata invalidates their input
snapshots. Current readiness is reported separately by the fresh plan.
