# Paired-camera YOLO ball experiments

The runner is experimental. It neither deploys a checkpoint nor turns model
suggestions into labels. Production plans consume the separate
`training_data/live/training_sessions.json` registry: existing/A is train, B is
validation, and C is sealed test. C capture, image, index and annotation files
are **not opened or hashed** by planning, preparation or fitting.

| Stage / runner name | Change |
|---|---|
| A / `data_only640` | 640 input, first ten layers frozen, historical 30-source-pixel center-square targets |
| B / `unfreeze640` | A with the backbone unfrozen |
| C / `high_detail960` | A with 960 input |
| D / `extent640` | Selected passing 640 freeze schedule; reviewed extent targets |
| E / `camera_aug640` | Selected passing 640 geometry/freeze recipe; camera-safe augmentation |
| `final_seed42`, `final_seed43`, `final_seed44` | Selected 640/960 recipe, 60 epochs, patience 12, three seeds |

Pilot settings are 12 epochs, AdamW, learning rate 0.0001, cosine schedule,
final learning-rate ratio 0.1, disabled early stopping, seed 42 and one worker. AMP, image
caching and plotting are disabled. D/E/final require an explicit recipe JSON
containing `freeze` (0 or 10), `geometry` (`center_square30` or `reviewed_extent`)
and `augmentation` (`common_preset` or `camera_safe`). D/E use `--best640-recipe`;
finals accept `--selected-recipe` with `imgsz` 640 or 960 (640 default), so a
successful 960 pilot is not silently reduced to 640. The legacy `final640_seed*`
names are retained as aliases; their actual input size always comes from the
saved selected recipe. These options record a comparison decision rather than
inferring a winner.
The production plan also freezes installed Ultralytics, PyTorch, NumPy, OpenCV
and Pillow distribution versions and the actual installed Ultralytics
`cfg/default.yaml` bytes. Unlisted inherited defaults therefore remain bound to
that unchanged file. Runtime/default drift invalidates the saved plan; planning
does not import heavy model packages, download weights or upgrade dependencies.

Every comparison uses a common batch chosen before fitting: 16, eight or four.
Camera-effect random seeds use original source/frame identity and epoch rather
than the private output folder, preserving repeatability when output folders change.
The actual trainer checks peak allocated/reserved CUDA memory against 7 GiB and
disables the installed Ultralytics first-epoch batch-halving retry. If a variant
exceeds memory, rerun all compared variants at one smaller common batch with
new plans/output folders. Native export, coordinate parity and live timing remain
separate gates, including for 960 input.

`yolo_ultralytics_hooks.py` supplies the `PairedCameraTrainer` passed directly to
`YOLO.train(trainer=...)`. Its dataloader uses `BalancedDomainSampler`: each
deterministic epoch contains 50% match, 25% iPhone and 25% PC camera examples.
Each camera contributes visible/absent observations in a 2:1 ratio. Epoch size
is rounded to a multiple of 24; reviewed groups cycle with replacement when
needed. This is actual sample selection, not a manifest-only statement.

Each epoch validates match, iPhone and PC camera separately. The custom
validator maps predictions to original source pixels and calls
`ball_release_metrics.score_frames(..., tolerance1080_px=4)` at fixed score
threshold 0.25. The minimum of the three domain F1 values controls checkpoint
selection/early stopping. Large match sets cannot swamp venue metrics.
`domain_ball_validation.jsonl` retains domain counts, precision, recall,
localization and selection fitness. Independent release approval remains required.

Camera augmentation uses gamma 0.8–1.25, brightness multiplier 0.85–1.15,
contrast 0.8–1.2, JPEG quality 50–95 and optional centered 3/5/7-pixel motion
kernels. Extent targets expand symmetrically; center-square targets retain their
centers. Boundary-truncating blur is skipped. Mosaic, mixup, copy-paste and
spatial jitter are disabled. The real dataset transform runs before tensor
formatting; mosaic closure is disabled so Ultralytics cannot replace it later.

Camera training data must supply at least 2,000 reviewed visible frames, 1,000
verified absences and 500 new-A continuity-audited positive three-frame windows,
with both devices represented. Each B device needs 1,000 visible references,
ten reviewed rallies and complete verified-absence intervals totaling at least
120 seconds after overlap deduplication. Unknown/hidden labels remain excluded. Existing venue
data contributes reviewed frame totals but not new A windows. C readiness remains not
audited until final sealed testing.

Separate `continuity_audit.json` records bind reviewed source-frame ranges to
the exact source identity, index SHA and image SHAs. Positive windows require
three reviewed box-positive frames and a precise reviewed latest center/streak.
Consecutive IDs alone do not establish
continuity. Sparse bursts cannot add up to a complete two-minute absence
interval. Reviewed rally identifiers use `geometry.rally_id` and
`geometry.rally_reviewed`. Duplicate images and conflicting labels are checked.
Conflicting source frames or decoded images are excluded as a group with explicit
blockers; no conflicting label wins by read order. Missing originals retain
reviewed-image progress counts but remain a distinct provenance blocker and
cannot permit fitting.
Moved recordings can be resolved through `training_data/live/source_locations.json`
by their unchanged original SHA. Both the sidecar and selected original bytes are
hashed into the plan. Filename similarity and folder names cannot establish a
source identity or a new session/split. Original capture manifests remain intact.

After assigning actual session/device metadata in Training Review:

```powershell
python -B python_tools/prepare_yolo_paired_training.py artifacts/paired_yolo_EXPORT
python -B python_tools/yolo_ball_experiments.py plan artifacts/paired_yolo_PLAN.json --prepared artifacts/paired_yolo_EXPORT --batch 16
```

An export can be saved while readiness remains blocked. A production plan
independently verifies every image, target, metadata record and list against
human review. Its `strict_data_blockers` prohibit fitting when data is missing.
When ready, run each pilot into a separate new directory:

```powershell
python -B python_tools/yolo_ball_experiments.py train artifacts/paired_yolo_PLAN.json data_only640 artifacts/paired_yolo_A
```

The trainer receives fresh private image/label copies. Source label caches are
not inherited. Source/model/code/private-input hashes are checked before and
after fitting. Geometry changes affect only private train targets; validation
keeps reviewed source centers. Errors are recorded and re-raised. A saved
checkpoint hash never implies accepted accuracy or deployment.

CPU control and installed-API hook checks:

```powershell
python -B -m unittest discover -s python_tools -p 'test_yolo*py' -v
```

Legacy fixture plans explicitly use `fixture_only=True` and cannot invoke real
neural training. Tests use synthetic temporary reviews or fake trainers. Hook
tests build an actual installed Ultralytics dataset/dataloader, check source
coordinate scaling against the release metrics and exercise memory/batch guards
without an optimizer. These prove connected controls, not model accuracy.
Current recordings lack the independent paired data needed for full fitting.

Complete original absence intervals are a separate review task. First import
the untouched recording and assign its existing session/device. Select a source
frame range which you have watched in the original video. Preparation sequentially
decodes the entire original, reconciles all decoded and reported frames, and
retains every selected frame at full resolution. It creates unreviewed suggestions
only. The review screen must explicitly confirm the entire original interval,
ball absence, and no hidden/unsure ball. Do not confirm intervals containing a
visible, hidden or uncertain ball. In particular, Video Project 6 contains actual
balls according to the user's correction; its filename or earlier description
cannot make it a negative clip.

The app uses this training-only CPU worker; standalone preparation is also
available:

```powershell
python -B python_tools/prepare_absence_interval.py --probe
python -B python_tools/prepare_absence_interval.py --captures training_data/live --clip EXISTING_CAPTURE_ID --start-source-frame 1 --end-source-frame 7201 --interval-id NEW_INTERVAL_ID
```

Source bounds are one-based inclusive decoded frame IDs. They are not a duration
claim: the worker uses original source PTS, never metadata FPS. The selected end
uses the next decoded frame's PTS when available. At original EOF the unknown tail
duration contributes zero. A nominal two-minute file can therefore have slightly
less than 120 seconds of audited coverage. Missing/nonmonotonic timestamps, gaps
over 100 ms, changed source bytes, frame-count mismatch and disk errors fail closed.
Each image write retains a 2 GiB free-space reserve.

`absence_intervals/ID/request.json` binds the session/device/role before extraction.
`manifest.json` binds original/capture SHA, sequential decoder/settings, all-source
PTS index and selected image index. Separate explicit `review.json` binds those
hashes plus the interval and review attestations. Existing annotations/manifests
remain intact. Any prior explicit visible, partly visible, hidden or unsure review
within the selected source range blocks absence confirmation. Unreviewed model
suggestions remain unknown.

The planner independently validates the confirmed manifest, original bytes,
source/selected indices and every selected image before creating empty ball labels.
All confirmed rows retain their source/frame identity, observed duration, review
SHA and interval ID in prepared metadata. Exact duplicates are deduplicated and
conflicting labels are excluded. For the 120-second B-device gate, the planner unions
actual source-frame PTS durations from complete confirmed original intervals.
Overlap counts once; inconsistent timing for the same original frame blocks readiness.
Sparse saved bursts contribute zero. Whole original hashes are bound to one
session/role/device before label inspection; conflicting reimports are excluded
even when sampled frames are disjoint. Sealed C is skipped using request/session metadata before
opening capture, annotation, prepared manifest or images.
Known sealed original SHA bindings also prevent decoding a development alias.
If a C assignment lacks its registry-only source identity, fitting receives an
explicit provenance blocker instead of assuming independence. Capture-directory
and evidence-file aliases/junctions are rejected before source evidence reads.

The app passes `--owner-pid` to the worker. A Windows SYNCHRONIZE process handle
stops work when its owning frontend dies, even if that PID is reused. Explicit
cancellation writes a `cancel.json` marker checked between decoded frames. Failed
or canceled partial extractions never publish a complete reviewable manifest.
Partial files remain identifiable by progress status and are never training labels.
An app-created job must additionally finish successfully with prepared progress
and no cancellation marker; a late job/log failure cannot credit a manifest. A
standalone CLI extraction has no app job file and retains the same source/review
and prepared-progress requirements.
No new optimizer run is authorized by interval preparation alone; all existing
paired-data gates still apply.

CPU acceptance checks include actual MP4 decoding, a 1,802-frame source with a
two 60-second intervals established by actual source PTS (120 seconds after union), review-only export, source
and index/image tamper rejection, sealed-C exclusion, disk failure and an actual
Windows owner-process death:

```powershell
python -B -m unittest discover -s python_tools -p test_absence_intervals.py -v
```
