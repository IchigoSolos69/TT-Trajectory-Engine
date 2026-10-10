# Improving detection in the camera environment

The provided room screenshots show false person detections on furniture and an
unstable ball signal. They are diagnostic examples, not a new accuracy benchmark:
the screenshots already contain HUD text and detection overlays. They have not
been added as ground truth or used to train the deployed models.

## Collect clean examples

The live Visualizer now includes **Recording checklist, gameplay steps & warnings** below its recording controls. Expand it for an optional four-item setup checklist, an ordered collection routine (verified absence → clear ball examples → normal rallies → temporal bursts → difficult cases), and post-capture labeling/split instructions. The checklist records your observations only; it does not certify image quality or block recording. Recheck the setup after changing the camera or session.

Keep every clip from the same capture session together when assigning train/validation/test splits. Include both players and the full table/net, check saved near/far ball images, and collect clear examples alongside blur/occlusion cases. Hidden balls are not automatically absent-ball negatives. The visible mode cards distinguish sparse detector images, five-frame temporal bursts and selected difficult examples. These controls do not save continuous match video; keep a separate original recording for full-rally/event evaluation or highlights.

The collection routine is practical guidance, not a measured optimum for this camera. Its diversity, complete-label and held-out-evaluation principles follow [Ultralytics dataset guidance](https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results). Camera property behavior depends on the hardware, driver and capture backend, as described by [OpenCV's video I/O documentation](https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html). The current app does not configure or verify exposure, focus, resolution or sensor FPS.

1. Launch live mode and authorize the selected camera.
2. In Visualizer, click **Record training clip**. The backend saves original camera
   images without annotations at up to 5 samples/second. Each recording stops
   after 60 seconds, or when you stop recording, stop the camera, switch cameras,
   or disconnect every dashboard client.
3. Record separate clips of a stationary ball at different distances, slow/fast
   movement, partial occlusion, people moving, and the room with no ball/person.
   Include the actual table-tennis setup when available. A desk is not a
   table-tennis-table training positive.
4. Reserve separate later recordings, ideally with different lighting or camera
   placement, for validation and testing. Do not split adjacent frames randomly.

Files stay under `training_data/live/live_<timestamp>_<sequence>/` and are ignored
by Git. No upload occurs. `capture.json` records the camera and source resolution;
`frames.jsonl` maps each JPEG to its source frame ID and elapsed time. These are
sampled image sequences for detector training, not a full-frame-rate video for
measuring bounce timing. A bounded disk-writing queue reports skipped samples.
Disk failures appear in the UI and backend log, and previously saved JPEGs remain
usable. Do not judge model improvement by whether recording itself works.

## Review labels before training

For temporal training, choose **Record temporal bursts**: five consecutive captured frames once per second, with the same explicit start/stop and 60-second limit. Each row records burst ID, index, intended length, source frame and elapsed time. Capture gaps or a full disk queue can leave incomplete bursts; these must not be padded with duplicated or interpolated frames. No camera was opened for the September 29 recorder tests; they use synthetic frames.

The review page also supports eight table/net landmarks, visibility and blur tags, and event labels at the frame's recorded timestamp. Geometry and events have separate reviewed flags; omitted landmarks and unknown tags remain unknown. The API stores immutable prior versions in `annotation_history/`. These labels prepare future landmark/event training; they do not imply that a landmark model or event classifier has been trained.

Audit reviewed temporal bursts after assigning whole recordings to train/val/test:

```powershell
python python_tools/audit_temporal_captures.py training_data/live live_splits.json artifacts/live_temporal_audit.json
```

The audit rejects cross-split duplicate images and recording reuse, excludes incomplete/nonconsecutive/unreviewed bursts, and records image hashes and timing. A nonempty audit is a data-integrity check, not sufficient sample size or live-accuracy acceptance. Actual reviewed room and table-rally captures are still required.

The dashboard's **Review saved frames** link opens `/training-review`. **Ball only · faster** is the default: click the ball, correct its visible extent if needed, then use **Save & next** or Enter. A click creates an initial 12-pixel box or reuses the current/previous reviewed box size; it replaces the ball box rather than adding duplicates. You do not need people, table geometry or event labels for ball-detector training. **Full scene & advanced** exposes separate scene boxes and optional geometry/events.

**Reuse previous ball box** copies a reviewed box from the adjacent source frame in the same burst into a draft. Move/check it before saving; no motion is inferred and no boxes are automatically approved. **Not in frame** explicitly confirms absence. **Hidden / unsure** excludes the frame from ball-detector training, instead of treating occlusion as absence. **Skip / keep unreviewed** advances without approving the selected review scope. Arrow keys navigate frames, and **Next needing review** wraps to earlier unchecked/conflicting labels.

Drafts are stored locally in this browser and recovered on a fresh page load. They remain outside training until explicitly saved. If browser storage fails, the UI warns you to save before leaving. Conflicting server revisions are rejected instead of overwritten. Ball-only saves preserve previously saved scene boxes, landmarks and events; unchecked scene classes cannot become scene-model negatives. Existing old-page drafts must be saved before reloading to upgrade to this version.

With the experimental backend enabled, **Capture difficult examples** selects frames for low confidence, missing expected balls, motion disagreement, competing detections, low ROI sharpness, uncertain person candidates or unavailable calibration. It saves at most five frames per second, rejects similar reason/location/image descriptors and stops after 60 seconds even if nothing qualifies. This is a heuristic selector, not a trained uncertainty sampler; it cannot know by itself that furniture is a false person. Model hashes and selection reasons are stored beside suggestions. Use the ordinary recording control to collect representative positive and negative coverage too.

The UI saves the same annotation format accepted by the preparation script. The command below is an alternative initializer, not a required step after using the UI.

```powershell
python python_tools/prepare_live_training.py init training_data/live/<clip>
```

This creates `annotations.json` with every frame marked `reviewed: false`.
For every class you mark reviewed, record **all** its visible instances as
`[left, top, width, height]` boxes normalized to the source image. New annotations
carry explicit `reviewed_classes` flags. An unchecked class's empty array means
unknown, not absent. Reviewed ball negatives also require an explicit absent tag.
Keep unclear or unreviewed frames excluded. Predictions are suggestions, never
ground truth. Legacy full-review annotations remain readable, but a visible or
partly visible ball with no box is excluded from ball training.

```json
{
  "frame_1.jpg": {
    "reviewed": true,
    "reviewed_classes": {"ball": true, "person": false, "table": false},
    "ball": [[0.45, 0.55, 0.04, 0.05]],
    "person": [],
    "table": []
  }
}
```

Create `training_data/live/splits.json` with distinct recording IDs:

```json
{"train":["live_training_clip"],"val":["live_validation_clip"],"test":["live_test_clip"]}
```

```powershell
python python_tools/prepare_live_training.py prepare training_data/live/splits.json training_data/prepared_v1
python python_tools/train_live_candidates.py training_data/prepared_v1 training_data/candidates_v1
```

Preparation rejects overlapping recordings, duplicate images across splits,
missing reviewed class labels, invalid boxes, and accidental test-match imports.
It mixes reviewed live examples with original chronological match training data,
retains match validation, and excludes the live test recordings from fitting.
Training requires at least 30 reviewed frames in each independent split; this is
only a minimum input check, not a sufficient dataset-size recommendation. Include
both ball-positive and verified absent frames, plus real people examples.

For the simpler ball-only workflow, prepare and train only the ball candidate:

```powershell
python python_tools/prepare_live_training.py prepare training_data/live/splits.json training_data/prepared_ball_v1 --kind ball
python python_tools/train_live_candidates.py training_data/prepared_ball_v1 training_data/candidates_ball_v1 --kind ball
```

The version-2 preparation audit records separate counts and exclusions per model.
Scene training requires both person and table classes to be reviewed; ball-only
labels never create a scene dataset. Temporal-burst audits require ball review
only. The existing independent-session, duplicate and held-out test safeguards
apply to every preparation kind.

Candidate weights are saved separately. Before replacing either deployed engine:

- Compare baseline/candidate ball precision, recall, localization error, and false
  detections on the same reviewed live test recordings, at unchanged thresholds.
- Compare person/table detection and false person labels on empty-room frames.
- Re-run the match detector evaluations and retain separate live/dataset reports.
- Export to TensorRT, check native C++ results and PyTorch parity, then promote only
  a candidate that passes the agreed regression limits. Model confidence is a
  score; larger values by themselves do not establish better accuracy.

## Current status

Recording and data preparation are implemented. No model has been newly trained or
promoted for these room examples yet. The October 8 iPhone import adds ten original
camera recordings from one venue/session as 300 initially unreviewed frames,
plus 15 focused failure frames from IMG_7943. Reviewed labels
and independent later sessions are still required. The separately trained temporal
research candidate uses match footage only; it does not establish venue accuracy.
See `INTELLIGENCE_ROADMAP.md` for its failed promotion gates.

On October 9, IMG_7944 has 30 saved full reviews. The readiness audit accepts 19
ball-positive frames for ball-detector preparation. Five empty-ball entries
are tagged visible, five have unknown visibility, and another box is tagged fully occluded. Those eleven need
clarification and are excluded from detector preparation without changing the
saved annotations. This is a consistency audit, not proof that the other boxes
are accurate ground truth. See `artifacts/easy_review_data_audit.json`.

## Imported iPhone recordings and lighting experiments

Open [Training review](http://localhost:3000/training-review), choose an original
filename such as `IMG_7944.MOV`, and review all visible ball/person/table boxes.
Mark uncertain frames unreviewed. A fully hidden ball is not an absent-ball
negative; reviewed frames tagged occluded, or partial with no observable ball box,
are excluded by the preparation/audit tools. Label table/net landmarks only when
visible. Save labels explicitly; the import never supplies ground truth.

The local importer preserves originals, fingerprints them and extracts evenly
spaced bursts of five consecutive frames into `training_data/live/phone_*`.
It counts decoded frames and records source PTS. It publishes a capture only
after its manifest and images are complete. Failed partial imports remain under
dot-prefixed directories and are not listed by the review API. `--resume` verifies
the source/session and reuses completed imports without overwriting annotations.

```powershell
python python_tools/import_phone_recordings.py "C:\Users\Adi\Downloads\recordings" training_data/live --session iphone_same_venue_session_1 --bursts 6 --resume --report artifacts/phone_recordings_import.json
```

All ten clips belong to the same known session. No train/validation/test split was
assigned. Session-aware guards now reject splitting these clips across partitions.
The first four clips are about 30 FPS, the remaining six about 60 FPS. All decoded
images are 1920×1080. `IMG_7936.MOV` reports two more frames than the decoder yields;
random frame seeking in `IMG_7944.MOV` also disagrees with sequential source timing.
Phone imports use sequential timestamps; native recording replay/velocity timing
still needs dedicated MOV acceptance tests. HDR/color metadata is not established
by the current OpenCV audit; do not assume an appropriate HDR tone mapping.

`compare_lighting_filters.py` tests raw, bounded luminance gamma, CLAHE and an
adaptive combined candidate on identical decoded frames with the deployed
TensorRT engine. The authoritative experiment uses sequential decoding:

```powershell
python python_tools/compare_lighting_filters.py "C:\Users\Adi\Downloads\recordings\IMG_7944.MOV" models/yolov11_ttgames.engine artifacts/new_lighting_comparison
```

In the nine-frame sequential comparison, all variants remained below the current
0.25 threshold. These are unlabeled responses, not precision/recall measurements.
No preprocessing candidate was integrated into the deployed detector and no
confidence threshold/model was changed. Improved venue coverage, carefully
reviewed small-ball labels and controlled raw/filtered/ROI comparisons come next.
CLAHE is an experimental local contrast method, not proof of detection accuracy;
see [OpenCV's documentation](https://docs.opencv.org/4.13.0/d2/d74/tutorial_js_histogram_equalization.html).

## Review the dimmed recording failure

Choose **IMG_7943.MOV · Ceiling-light false positive and missed ball near frame
1685** in Training review. Its 15 frames cover source frames 1680–1694 in three
adjacent five-frame bursts. Frame 1685 is the sixth image. All start unreviewed,
retain the original source hash/timestamps and belong to `iphone_same_venue_session_1`.
These targeted frames cannot be treated as a new independent validation session.

Use **100%** for native pixels or **200%** for inspection, scrolling within the
image. Drag the real ball's visible extent; a click's initial 12-pixel box is only
a starting point. Ball-only review is sufficient for ball training; use full-scene
review when you also want to train person/table detection. The ceiling light
is background, so do not give it a ball box. Set visibility and blur tags only
when you can judge them. Leave uncertain positions unreviewed and distinguish
occlusion from verified absence. Save reviewed labels explicitly. Zoom changes
presentation; saved boxes remain normalized to the original image.

The deployed detector reproduced the approximately 38% ceiling-feature score at
sequential frame 1685. Gamma, local contrast and a fixed overlapping crop grid did
not recover the user-marked ball in this experiment. Dimming alone is therefore
not a verified solution to this failure. No filter or confidence threshold was
promoted. Use reviewed small/blurred balls and lamp/wall backgrounds in the next
training comparison; preserve originals rather than darkening them destructively.

To import another reported failure without overwriting earlier annotations:

```powershell
python python_tools/import_phone_recordings.py "C:\Users\Adi\Downloads\recordings\IMG_7943.MOV" training_data/live --session iphone_same_venue_session_1 --bursts 3 --start-frame 1680 --capture-sequence 2 --note "Ceiling-light false positive and missed ball near frame 1685" --resume --report artifacts/phone_dim_failure_import.json
```

Use a new capture sequence for a different target interval. Sampling remains
sequential, and an out-of-range request is rejected rather than silently shortened.

## No-ball footage without hundreds of repeated markings

The local launcher has a **Training review** button. Open it without starting a
camera session, then select **Video Project 6.mp4**. There are two distinct captures:
300 uniform images (`phone_452627e98a3c2652_1`) and 120 selected suspected-failure
images (`phone_452627e98a3c2652_2`). Both now have saved reviews and belong to the earlier
`iphone_same_venue_session_1`. The 420 saved images cover 407 distinct source frames;
13 selected frames overlap the uniform capture. Account for these duplicates when
building a training manifest instead of treating them as independent examples.
The user confirmed on October 10 that actual balls are present in the positive
frames. This is mixed footage: do not bulk-label the whole recording absent.
The two batches contain 124 positive image files, 290 absent files and six unknown
files before deduplication/conflict exclusion. Seven duplicate positive labels
have different box geometry and stay excluded pending reconciliation. Existing
reviews do not need to be repeated wholesale.

Expand **No-ball interval · review saved images together**, choose source-frame
bounds and preview. Inspect the selected images and use the individual frame
controls for uncertainty or existing ball evidence. Save pending drafts first.
When the recording interval is known to contain no ball, tick the explicit
confirmation and save. This approves only saved images within that interval.
Unsampled footage, person/table annotations and model proposals remain unreviewed.
Hidden/uncertain balls are not equivalent to an absent-ball negative. Confirmation
is refused when visible, partial or occluded ball evidence conflicts with absence.
Original revisions, source/image hashes and the interval review are preserved.

The source is about 5 minutes 5 seconds at 1080p/30 FPS, not six minutes at 60 FPS.
The complete sequential decode found 9,164 frames with monotonic decoder PTS.
Editing/cut/duplicate history is unknown. Inspect consecutive bursts for continuity
before temporal training. Keep all recordings from this session in one split.
Use a separate later session for independent validation/test, including real ball
rallies and ordinary no-ball scenes. Negative-only training cannot teach the model
where the ball is; combine these examples with reviewed positive gameplay.

The selected capture was generated from the complete causal native BlurBall
diagnostic, using temporal/spatial diversity rather than saving every adjacent
failure. Its selection reasons and source/model/journal hashes remain attached;
suggested failures never become labels automatically. The reviewed training
export validates explicit absence provenance and excludes unchecked scene classes.
See [the implementation report](HYBRID_PRECISION_IMPLEMENTATION.md) for native,
API, dataset and packaged-GUI evidence and the remaining model gates.

To reproduce the selected import into its existing capture without overwriting
reviews, use the exact selection and `--resume`:

```powershell
python python_tools/import_phone_recordings.py "C:\Users\Adi\Downloads\Video Project 6.mp4" training_data/live --session iphone_same_venue_session_1 --review-selection artifacts/no_ball_hard_negative_selection.json --capture-sequence 2 --capture-origin unspecified --resume --report artifacts/no_ball_hard_negative_resume.json
```

Do not combine `--review-selection` with `--start-frame` or a separate `--bursts`
request. A stale source/session/journal or changed selector is rejected. This
command imports/verifies review images; it does not train or promote a model.

## Check saved-label readiness before training

Run the read-only audit before preparing a candidate. Without a frozen split
manifest, it reports current labels under `unassigned`; it never chooses splits.
Use a new output filename each time, or omit `--output` to print the report.

```powershell
python -B python_tools/audit_training_readiness.py training_data/live --output artifacts/new_training_readiness.json
```

Once independent whole-session partitions are recorded explicitly:

```powershell
python -B python_tools/audit_training_readiness.py training_data/live --splits training_data/live/splits.json --output artifacts/new_frozen_training_readiness.json
```

The audit reuses reviewed-label and absence-provenance validators, checks original
recording hashes, and deduplicates both recording/frame identity and decoded RGB
images. Conflicting reviewed duplicates are excluded and reported. Cross-partition
source/image duplicates and sessions are reported as leakage. Hidden, uncertain
and unreviewed balls never become verified negatives. Numeric consecutive frame
IDs produce structural CNN window counts only; cut/duplicate continuity and a
real network/loss/backward smoke remain separate requirements. A minimum count
passing is not proof that labels are correct, a dataset is sufficient, a candidate
is accurate or a model may be promoted.

Current evidence is [the updated October 10 readiness report](artifacts/real_training_readiness_2026_10_10_v2.json):
750 saved images cover 732 distinct source frames in one session. After reviewed
duplicate exclusion, 130 ball-positive images and 293 verified absent images are
structurally eligible. There are eight reviewed visible CNN targets (six points,
two streaks), 293 absent targets and 178 structural three-frame windows: three
visible and 175 absent. These counts do not prove temporal continuity or adequate
training diversity. No independent session partitions are assigned.
The earlier `real_training_readiness_v3.json` zero-negative inventory is historical.
Original labels and model files were not changed by either audit.

The five **IMG_7944.MOV** images **16–20** / source **14960–14964** are now saved
as manually reviewed center targets. They form the three visible windows. Existing
boxes remain useful for detector training; they were not automatically converted
to approved endpoints. No measured streak extent is inferred from center points.
See [the current handoff checklist](NEXT_TRAINING_STEP.md) for remaining positive
window diversity, continuity, independent capture, split and promotion requirements.

## Stage current real reviews without inventing held-out splits

The new staging command creates a hash-bound, read-only single-session snapshot.
It deliberately keeps `partition=unassigned` and `ready_for_fitting=false`.
Unknown/occluded/partial targets are excluded; reviewed positive/absent windows
are deduplicated. Existing user labels, sources and split assignments are preserved.

```powershell
python -B python_tools/stage_reviewed_training.py training_data/live --session iphone_same_venue_session_1 --output artifacts/new_reviewed_training_stage
python -B python_tools/smoke_reviewed_blurball_training.py artifacts/new_reviewed_training_stage --loss quality_focal --output artifacts/new_real_data_optimizer_smoke
```

Use new output paths. A stale review, source, image or adapter snapshot fails
closed. The smoke verifies one real fully visible triplet and one real fully
absent triplet, without saving weights. Its loss choice is explicit; actual
fine-tuning and validation are separate tasks. [The completed real-data check](artifacts/blurball_reviewed_training_smoke_qfl_2026_10_10_v1/report.json)
passes; original checkpoint and staged data remain unchanged. Seventeen staging
and eight smoke guard tests pass. The existing synthetic MSE smoke also passes
after the optimizer helper was shared with the real-data path.

## Verify the CNN optimization path without touching real labels

The isolated smoke creates new artificial adapter fixtures and checks the actual
BlurBall network, selected loss, backward gradients and two in-memory Adam steps.
It saves evidence, not model weights. Use a fresh output directory for each run.

```powershell
python -B python_tools/smoke_blurball_training.py --loss mse --output artifacts/new_cnn_mse_smoke
python -B python_tools/smoke_blurball_training.py --loss quality_focal --output artifacts/new_cnn_qfl_smoke
```

The quality-focal path calls the inspected loss with dictionaries; the upstream
HeatmapLoss tensor/dictionary mismatch is not silently swallowed. Exactly the
final unused auxiliary fusion outputs are accounted for in gradient checks. Every
output-connected gradient must be present and finite, parameters must change and
the source checkpoint must remain unchanged. Existing passing reports are
[MSE](artifacts/blurball_training_smoke_mse_v2/report.json) and
[quality-focal](artifacts/blurball_training_smoke_qfl_v1/report.json). These do not
verify the full upstream Hydra runner or supply missing real training targets.
