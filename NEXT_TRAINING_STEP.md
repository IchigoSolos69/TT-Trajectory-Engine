# Training handoff after the completed reviews — October 10, 2026

Your two **Video Project 6.mp4** review batches and the five **IMG_7944.MOV**
temporal point targets are saved. You confirmed that
positive boxes mark actual balls. Preserve those boxes; this recording contains
both ball-present and ball-absent examples. You do not need to redo both batches.

## What is ready

| Audited scope | Current result |
|---|---:|
| Saved images, all 14 captures | 750 |
| Distinct source frames | 732 |
| Eligible detector-positive frames | 130 |
| Eligible reviewed absent frames | 293 |
| Reviewed visible CNN/streak targets | 8: six points, two streaks |
| Reviewed absent CNN targets | 293 |
| Structural three-frame CNN windows | 178: three positive, 175 absent |
| Independent sessions | 1, still unassigned |

These are structural label counts, not measured model accuracy. Unknown frames
and seven duplicate frames with differing positive boxes are excluded. Unchecked
person/table classes do not become scene-training negatives. All original labels
are preserved.

## Completed next step

IMG_7944 saved images **16–20**, source **14960–14964**, now contain explicitly
reviewed center points. They form the three visible windows. Additional reviewed
targets at 4987–4988 are streaks; 19951 is a point. They do not yet form additional
fully reviewed visible windows. Center points supervise position, not measured
blur length or direction. Existing boxes remain useful for YOLO.

The strict [real-data stage](artifacts/reviewed_training_stage_2026_10_10_v1/manifest.json)
preserves all data under the original **unassigned** session, snapshots source,
image, annotation and adapter hashes, and excludes unknown/conflicting targets.
The dataset reader returns chronological three-frame RGB inputs and reviewed
heatmaps without inventing validation/test assignments. Seventeen staging tests
and eight optimizer-selection/target guard tests pass.

The actual BlurBall network passes [two real-data optimizer-step checks](artifacts/blurball_reviewed_training_smoke_qfl_2026_10_10_v1/report.json),
one on the reviewed visible window 14960–14962 and one on reviewed absent frames
1–3. Both have finite loss and gradients; 212 connected parameter tensors receive
finite gradients and 206 tensors change in memory. Positive targets remain
nonempty in each channel and absence targets exactly blank. The original
checkpoint and complete staged snapshot are reverified afterward. No candidate
checkpoint is saved or deployed, and no accuracy improvement is claimed.

## Data and training sequence

1. The fresh read-only audit and real-data handoff are complete. Review additional
   representative positive consecutive windows; only three are currently usable.
   Keep hidden, partial and uncertain observations accurate. Use a fresh report
   and stage directory after further label changes; prior snapshots become stale.
2. Inspect consecutive bursts for cuts, repeated images and capture interruption.
   Consecutive frame IDs alone do not establish continuity.
3. Capture ordinary rallies and explicit ball-absent scenes in **two separate later
   sessions** for validation and test. Keep this existing session together for
   training when assigning splits; do not divide nearby frames between partitions.
4. Freeze session/split manifests and check duplicate-image leakage. Review the
   sampled labels in each partition, with positives and verified absences.
5. Prepare separate ball-detector and temporal-CNN candidate data. Preserve match
   training examples; unknown scene classes stay excluded. Do not train a CNN on
   only absent windows or invent reviewed streaks from rectangles.
6. Fine-tune candidates into new directories. Compare unchanged and candidate
   models on fixed match, bright-background, blur, occlusion and absent subsets.
7. Export passing candidates and check native TensorRT preprocessing/output
   parity before considering deployment. Failed candidates remain experimental.

The current trainer enforces at least 30 reviewed frames and five positive/five
absent examples per independent partition. These are minimum input safeguards,
not evidence of adequate training diversity or guaranteed accuracy.

## What the completed labels already reveal

On **284 unique reviewed absent frames only**, the existing paired replay yields
42 YOLO false detections and 71 causal BlurBall false detections. These examples
can teach background rejection. The mined batch is enriched for detector failures;
this is not the whole video's error rate or global precision/recall. The hybrid
selector accepts zero false balls here but previously retains only 1.85% of visible
balls on the official development clip, so it still fails promotion.

Only two in-memory real-data optimizer checks ran, not a complete candidate
fine-tuning run. No label rewrite, split assignment or deployed-model change
occurred. Independent live accuracy and the complete advanced-feature plan remain
unfinished.

Delivery: [hash-bound handoff report](artifacts/reviewed_training_handoff_delivery_2026_10_10_v1.json).
It rechecks saved review metadata, unchanged engines/checkpoint/backend and the
passing test/optimizer evidence. Candidate fitting readiness remains false.

Evidence: [current readiness](artifacts/real_training_readiness_2026_10_10_v2.json),
[reviewed-negative comparison](artifacts/reviewed_negative_subset_2026_10_10_v2.json),
[next positive burst](artifacts/positive_gameplay_next_review_20261010.json),
[duplicate detail](artifacts/no_ball_review_conflicts_20261010.json).
