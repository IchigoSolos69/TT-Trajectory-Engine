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

The dashboard's **Review saved frames** link opens `/training-review`. Select a local capture, draw ball/person/table boxes, correct model suggestions, then choose **Save reviewed labels**. A ball click creates an initial 12-pixel box; drag or correct its extent when necessary. Check all three classes, including explicit absence. **Exclude / mark unreviewed** keeps ambiguous frames out of training. Draft edits survive frame/clip navigation but must be saved before leaving the page. Conflicting edits from another review tab are rejected instead of overwritten. Files remain local.

With the experimental backend enabled, **Capture difficult examples** selects frames for low confidence, missing expected balls, motion disagreement, competing detections, low ROI sharpness, uncertain person candidates or unavailable calibration. It saves at most five frames per second, rejects similar reason/location/image descriptors and stops after 60 seconds even if nothing qualifies. This is a heuristic selector, not a trained uncertainty sampler; it cannot know by itself that furniture is a false person. Model hashes and selection reasons are stored beside suggestions. Use the ordinary recording control to collect representative positive and negative coverage too.

The UI saves the same annotation format accepted by the preparation script. The command below is an alternative initializer, not a required step after using the UI.

```powershell
python python_tools/prepare_live_training.py init training_data/live/<clip>
```

This creates `annotations.json` with every frame marked `reviewed: false`.
For each reviewed frame, record **all** visible balls, people and table-tennis
tables as `[left, top, width, height]` boxes normalized to the source image.
Explicit empty arrays mean the class was checked and is absent. Keep unclear or
unreviewed frames excluded. Predictions are suggestions, never ground truth.

```json
{
  "frame_1.jpg": {
    "reviewed": true,
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
promoted for these room examples yet. Clean recordings and reviewed labels are
still required. As of the latest workspace inspection, no `training_data/live`
capture directory exists. The separately trained temporal-recovery research
candidate uses match footage only; it does not establish an improvement in this
room. See `INTELLIGENCE_ROADMAP.md` for its failed promotion gates.
