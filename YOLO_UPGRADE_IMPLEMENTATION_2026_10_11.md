# YOLO improvement implementation — October 11, 2026

The standalone YOLO diagnosis, paired-camera training controls, native tensor
contracts and local review workflow are implemented. Candidate fitting and
model promotion remain unfinished because the required independent, reviewed
camera data is not ready. Existing engines and completed annotations are preserved.

## Recording folders and existing reviews

The Downloads folder audit found 23 videos: nine in training, eight in validation
and six in seal. Eleven relocated original sources match stored SHA-256 identities
and now resolve through `training_data/live/source_locations.json`. Original
capture metadata and annotation files were not rewritten.

The ten IMG_7936–IMG_7945 originals still belong to the previously reviewed
`iphone_same_venue_session_1`. Their actual current folder distribution is four,
three and three; moving them cannot create independent sessions. Twelve imports
of those originals, including targeted failure bursts, are explicitly registered
as existing iPhone training data. The separate folder comparison manifest preserves
the requested clip distribution as development comparisons. The user clarified
that the additional recordings are official World Cup gameplay from the event
website, and that there are no current PC/webcam recordings. Twelve additional
sources are recorded as user-declared public match footage in the development
manifest. They require their original match identities and reviewed targets;
they cannot establish the actual PC-camera domain. Previously exposed clips
cannot become blind tests.

The strict assigned-camera audit currently accepts 18 visible frames and nine
absent frames from 330 saved frames, with 301 unreviewed and two unknown. This
count does not discard the separate mixed-recording annotations: those are retained
and still require source/device/session reconciliation. Seven conflicting duplicate
positive boxes remain excluded until human review. Video Project 6 contains actual
balls in some reviewed frames and must never be labeled negative as a whole.

Evidence: `artifacts/yolo_upgrade_2026_10_10_v1/download_folder_audit.json`,
`existing_session_assignment.json`, `development_folder_split_v2.json`, and
`artifacts/yolo_paired_training_plan_2026_10_11_v4.json`.

## Implemented training controls

The experiment runner uses the actual installed Ultralytics dataset, dataloader,
trainer and validator interfaces. It supports the five controlled pilots, final
seeds 42/43/44, explicit 640/960 recipes, fixed common batches, a 7 GiB training
memory guard, private fitting copies, and source/model/code/default/runtime hashes.
Sampling is 50% match, 25% iPhone and 25% PC camera, with a 2:1 positive/absence
ratio inside each camera domain. Checkpoint fitness uses the weakest domain's
source-coordinate F1. Camera effects are applied to private training inputs;
original recordings are unchanged.

Reviewed-data checks require the approved 2,000 visible frames, 1,000 absences,
500 audited new-session positive windows and independent per-device validation
coverage. Session C is skipped before opening capture indexes, images or labels.
The CLI refuses a not-ready plan before loading a model or creating optimizer
output. Sixty-two training and interval tests pass, exercising real installed
interfaces and fixture data. The final v4 CLI refusal is also verified against
current data, with no optimizer directory created. This establishes connected
controls, not trained accuracy. See
`python_tools/YOLO_BALL_EXPERIMENTS.md` for commands.

## Complete original-video absence review

Training Review now has an explicit preparation and review workflow for a
continuous ball-absent interval. It decodes the entire unchanged original on the
CPU, reconciles its frame coverage, retains every selected frame and uses source
timestamps for duration. Preparation alone creates no reviewed labels. You must
inspect the complete interval and explicitly confirm absence and that no ball is
hidden or uncertain. People and table annotations remain unknown.

Review sidecars bind the original video, session/device assignment, prepared
images, frame index and annotation revision to immutable hashes. Failed, running,
canceled, changed, contradictory or sealed-C inputs cannot add training examples
or verified duration. Overlapping intervals contribute their union once. Another
import of the same original assigned to a different session, device or role
blocks all affected data. Existing ball-positive or hidden/unknown review prevents
an overlapping absence confirmation. Video Project 6 must therefore be reviewed
by interval, never declared negative as a whole.

The optional decoder needs an installed training Python with OpenCV and NumPy.
The application checks that runtime on explicit request; it does not download one
or start decoding when the review page opens. Normal footage analysis remains
native. Actual-worker API tests use synthetic original videos, including spaces
in paths; they do not label or modify the user's recordings.

Evidence: `artifacts/full_absence_api_checks.json` (64 actual decoder/API checks),
`artifacts/absence_interval_training_tests_2026_10_11_v4.log`, and the packaged
interval check below. Session API, decoder ownership and readiness tests pass
43, eight and 24 checks respectively. Real complete-interval training absence
coverage is still zero until the user reviews appropriate original intervals.

## Native diagnosis and measurement

TensorRT export and native decoding use actual input/output shapes, including
640/960 and synthetic P2 contracts. Real checks cover the existing ball and scene
engines and an isolated 960 export; no new detector has been trained. A complete
272-frame native paired replay retains the previous candidate coordinates and
selection results, with actual tensor dimensions recorded in its journal.

Nineteen fixed development frames were checked through both CPU-linear and
CUDA-nearest preprocessing. Exact same-input raw PyTorch/TensorRT comparisons
pass on the evaluable 640 cases. One 960 CUDA case exceeds the one-source-pixel
p95 coordinate gate (approximately 1.05 px), so 960 parity is not accepted.
Both frameworks still miss real venue balls and produce absent-frame false
positives. Preprocessing outcomes are mixed; an interpolation switch is not
justified as an accuracy fix.

Diagnostic NMS now includes out-of-image proposals before source-center
validation, matching native candidate processing. Score parity is evaluated even
when coordinate support is absent. Reaudit of the saved native snapshots found
no changed failure classifications on these cases; all saved raw-score comparisons
pass, while the 960 coordinate failure remains. Nine diagnostic tests pass.

The release scorer counts wrong locations as both false positives and misses,
excludes predicted observations from detections, separates device domains and
uses strict 1080-equivalent localization thresholds. Sparse negatives cannot
establish video duration or false-track rates. Rounded duration metadata cannot
add time to observed source PTS, and missing track identities yield unavailable
episode rates. Nineteen scoring tests pass. Arithmetic gates alone cannot establish
independent test provenance or authorize promotion.

Evidence: `artifacts/yolo_shape_native_2026_10_10_v1/paired_shape_regression.json`,
`artifacts/yolo_upgrade_2026_10_10_v1/raw640_v2/report.json`, `raw960/report.json`,
`raw_postprocess_reaudit.json`, `raw_diagnostic_tests_reviewed_v2.log`, and
`release_and_raw_final_tests.log` (28 tests: 19 metrics and nine diagnostics).

## Local application checks

The isolated paired-training v2 application passes production compile/type
checks and six desktop service/security tests. Its packaged Chromium test passes
19 checks with zero renderer errors:
explicit A/device assignment, source-bound burst continuity, reload, manual-only
sealed-C review, same-origin saves, preserved labels, all 272 recording frames,
and shared footage/2D/3D source-frame coordinates. The new complete-interval
workflow decodes all 90 synthetic source frames, retains all 60 selected frames,
requires explicit human assertions, persists/reloads exactly 2.00 seconds of
timestamp-backed evidence and rejects a delayed old inspection after a capture
switch. Existing source/scene labels are unchanged. Closing frees both owned
service ports. Camera access remains unauthorized throughout these tests.

The recording dialog test exercises the native bridge using a controlled file
selection; actual manual Windows dialog clicking and actual live camera acceptance
remain unverified. There is no measured 60 FPS live or 30-minute camera soak result.
Offline model diagnostics and native replay are distinct from live capture-to-display
latency. Failed earlier harness reports are retained alongside the passing report.
The latest failed harness incorrectly expected an already assigned session role
to remain editable; the corrected check verifies its intentional immutability.

Start this build with `launch_desktop.bat`. The packaged runtime is
`desktop/dist/TT Trajectory Engine paired training v2/TT Trajectory Engine.exe`;
previous builds remain available as launcher fallbacks.

Evidence: `artifacts/yolo_upgrade_2026_10_10_v1/desktop_smoke_v4.json`,
`frontend_build_v2.log`, `desktop_unit_v2.log`, `desktop_package_v2.log`, and
the release manifest in the paired training v2 folder.

## Remaining release work

1. Reconcile added recordings to complete original sessions/devices, preserve
   this existing iPhone session for development/training, and collect the missing
   independent paired A/B/C camera data.
2. Review source-bound absence intervals, positive windows, precise centers,
   complete-rally review, second-pass annotation checks and leakage audits.
3. Run the controlled YOLO pilots and final multi-seed training after readiness
   passes. P2, YOLO11s and causal crop experiments remain conditional escalations.
4. Develop the learned YOLO–BlurBall selector only after standalone YOLO passes
   validation. BlurBall fine-tuning, grouped out-of-fold selector fitting and
   temporal-student fitting remain unfinished.
5. Verify native candidate export, sealed independent acceptance, actual camera
   timing and 30-minute operation before any default replacement. Maintain rollback.

The broader digital-twin/event/physics/tactics plan and final whole-chat closure
audit remain open in `REQUIREMENTS_LEDGER.md`. Successful builds and runnable
training controls do not satisfy model accuracy or those remaining features.
