# Training handoff after the completed reviews — October 10, 2026

October 12 update: both Video Project 6 review batches are now registered as
existing iPhone training from the original session, following the user's device
confirmation. All 868 capture-tree files were preserved and reverified. The fresh
strict audit counts **128 eligible visible frames and 293 absences**, zero PC
camera/independent validation examples, and zero positive windows from new A.
Seven duplicate disagreements and two unknown-visibility labels remain excluded;
17 data/export blockers keep fitting disabled. Sealed labels were not consumed.
See [current collection and training steps](TRAINING_NEXT_ACTIONS_2026_10_12.md),
[the focused review queue](TRAINING_REVIEW_QUEUE_2026_10_12.md), and
`artifacts/yolo_paired_training_plan_2026_10_12_v2.json`.
Earlier structural counts and unassigned-session snapshots below remain
historical evidence; they are not the current paired-training readiness count.

October 11 update: the approved paired-camera YOLO runner now supersedes the
older minimum-input fitting safeguards below. See
[current implementation and data audit](YOLO_UPGRADE_IMPLEMENTATION_2026_10_11.md)
and [the executable workflow](python_tools/YOLO_BALL_EXPERIMENTS.md).
The moved sources resolve by unchanged SHA; clip-level folder splits
are development comparisons until original session/device identities and
independent camera validation are established. Completed reviews are retained.

The added World Cup clips are mixed-match highlight compilations, according to
the user's clarification. No current PC/webcam recordings exist. They can support
development and, after original-source/overlap auditing and review, match training;
they cannot replace independent camera validation or sealed test sessions. The
latest recursive inventory contains 29 videos (14/10/5 by folder), with 11 moved
originals reconnected to existing reviews. Extra session/match subfolders are
optional; source/session metadata carries the grouping. Existing iPhone clips
remain one training session regardless of where their files were moved.
The canonical recording root is now `python_tools/dataset/raw_videos`; see
`artifacts/repo_cleanup_2026_10_11/relocated_recordings_audit.json` and
[the storage cleanup report](REPOSITORY_CLEANUP_2026_10_11.md).
The earlier `artifacts/yolo_paired_training_plan_2026_10_11_v5.json` resolved
assigned originals but retained 15 data/export blockers. Regenerate a fresh
frozen plan before fitting: relocation changed the source resolver manifest,
and source-path compatibility changes also invalidate previous code hashes.
No training or split promotion resulted from cleanup. Existing models,
750 review images and 42 capture/index/annotation files pass preservation checks.
The paired workflow requires 2,000 reviewed visible
frames, 1,000 verified absences and 500 audited positive windows for training,
plus independent per-device validation coverage. The app's readiness panel shows
the missing data and keeps fitting disabled.

For easier absence review, import and assign the unchanged original, then use
the complete-interval preparation control in Training Review. Inspect every
prepared frame and the original interval before explicitly confirming that the
ball is absent and not hidden or uncertain. This saves a source-bound review
sidecar without changing earlier annotations. Do not confirm all of Video
Project 6: actual balls occur in some of its reviewed frames.

The next implementation milestone is now executable: a BlurBall-primary native
selector, a causal model using actual YOLO11n features and temporal heatmaps, and
an isolated YOLO experiment runner. The selector repairs the development recall
failure but fails venue-negative acceptance; the student passes real optimizer
checks but is not fitted. See [current implementation evidence](HYBRID_V2_IMPLEMENTATION_REPORT.md).
Completed reviews remain intact; no new accuracy claim or default change follows.

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

The older reviewed-data adapter/smoke safeguard enforced at least 30 reviewed
frames and five positive/five absent examples per independent partition. Those
historical minimums are not the active paired-training readiness gate above,
nor evidence of adequate training diversity or guaranteed accuracy.

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
