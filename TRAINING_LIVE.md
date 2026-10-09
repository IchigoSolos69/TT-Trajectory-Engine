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
a starting point. Check every visible person and the table too. The ceiling light
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
