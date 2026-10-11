# Hybrid precision implementation — updated October 10, 2026

The [implementation plan](HYBRID_PRECISION_PLAN.md) is being executed in stages. This delivery completes the easier absence-review workflow and a source-audited negative diagnostic. The October 10 review below supersedes the initially unreviewed inventory. It does not promote a filter, train new weights, or establish independent live accuracy.

## October 10: completed reviews and next training step

The user confirmed that the ball boxes in **Video Project 6.mp4** mark actual balls. It is a mixed recording, despite its earlier description as a no-ball scene. Preserve reviewed positives and explicit absences; never label the entire recording absent from its original description. A directly inspected saved image, uniform image 211 / source frame 6520, contains a real ball beside the net.

Both imported batches have saved reviews. The uniform batch has 65 positive and 235 absent images. The selected batch has 59 positive, 55 absent and six unknown images. Thirteen source frames overlap. Seven overlapping positive frames have slightly different boxes (IoU 0.623–0.806, center differences about 2–5 pixels), not presence/absence disagreements. Their conflicting copies remain excluded; no label is replaced, averaged or approved automatically. Six unknown frames remain excluded rather than becoming negatives.

Across all 14 captures, the fresh [readiness audit](artifacts/real_training_readiness_2026_10_10_v1.json) finds 750 saved images / 732 distinct source frames, **129 structurally eligible detector-positive frames and 284 reviewed absent frames**. There are 284 absent CNN targets and 172 structural three-frame windows, but **zero reviewed visible CNN/streak targets**. All recordings still belong to one session with no frozen independent split. Earlier zero-negative counts below are historical October 9 evidence, not the current inventory.

The [reviewed-negative subset comparison](artifacts/reviewed_negative_subset_2026_10_10_v2.json) audits the existing complete paired replay against these 284 unique reviewed absent frames only. YOLO selects a false ball on 42; causal BlurBall on 71; the experimental selector on zero. The selected batch was mined from detector hypotheses, and adjacent samples are correlated. These counts are not full-video false-positive totals or global precision/recall. Zero errors on this negative subset does not rescue the selector's previously measured 1.85% positive recall, so it remains unpromoted. Journal/source/model hashes pass; the older evaluator executable's recorded hash cannot be reverified after its rebuild, and every original frame was not independently redecoded for this subset audit. The new subset scorer and existing strict paired scorer pass [42 tests](artifacts/reviewed_negative_subset_tests_2026_10_10_v1.log), including unknown/positive exclusion, overlapping/conflicting duplicates, actual source PTS, stale-review protection and absence-history integrity.

The next small task is visible temporal supervision, not redoing the completed batches. In **IMG_7944.MOV**, saved images **16–20** (source frames 14960–14964, about 249.81 seconds) are already reviewed box positives. Inspect each in **Blur streak** mode: click the ball center for a sharp point, or drag the two visible ends for a blurred streak, then explicitly save. Preserve existing boxes. Five genuinely reviewed consecutive targets can supply three structural CNN windows; this is a handoff check, not enough data or proof of continuity/accuracy. Hidden, partial and uncertain balls must keep their real status. [Exact next-review inventory](artifacts/positive_gameplay_next_review_20261010.json), [duplicate details](artifacts/no_ball_review_conflicts_20261010.json).

Real candidate fitting still requires representative reviewed positive windows, continuity checks and frozen whole-session train/validation/test partitions. Capture validation and test in separate later sessions with normal rallies and ball-absent backgrounds. The current session can be used for training once explicitly assigned; nearby clips must not be scattered across splits. No weights, splits or user annotations changed during this audit. [Step-by-step handoff](NEXT_TRAINING_STEP.md).

## October 10: real reviewed-data handoff completed

The later October 10 handoff is now complete: the five IMG_7944 targets were saved
as manual center points. [Fresh readiness](artifacts/real_training_readiness_2026_10_10_v2.json)
finds 130 detector positives, 293 reviewed absences and eight visible CNN targets.
[Strict real-data staging](artifacts/reviewed_training_stage_2026_10_10_v1/manifest.json)
yields three positive and 175 absent windows with the session still unassigned.
Seventeen staging tests and eight smoke guard tests pass. The real network passes
[two reviewed-data quality-focal optimizer checks](artifacts/blurball_reviewed_training_smoke_qfl_2026_10_10_v1/report.json)
with finite gradients, changed in-memory parameters, and checkpoint/data hashes
reverified afterward. No candidate is saved or promoted. The shared helper retains
the [passing synthetic MSE path](artifacts/blurball_training_smoke_mse_2026_10_10_v3/report.json).
Three positive windows are insufficient for representative fine-tuning; separate
validation/test sessions and temporal-continuity audit remain required. The prior
inventory and subset comparison above are retained as historical evidence.

## Use the updated application

Launch [launch_desktop.bat](launch_desktop.bat). It selects the separate `desktop/dist/TT Trajectory Engine training review v2/` release. If the older application is open, close it normally first: the single-instance guard otherwise focuses the existing process. The earlier standard release and rollback folder are preserved.

1. Choose **Training review** on the native launcher. This opens review without starting inference or authorizing a camera.
2. Select **Video Project 6.mp4**. The uniformly sampled capture contains 300 saved images in 60 five-consecutive-frame bursts.
3. Expand **No-ball interval · review saved images together**. The initial range covers the saved source frames. Preview reports the affected images and exposes existing ball/occlusion conflicts.
4. Inspect the preview and navigate to individual images when needed. Only confirm an interval you know contains no ball; a hidden or uncertain ball requires individual review. Save pending drafts first.
5. Tick the explicit absence confirmation and save the interval. This labels only the saved images as ball-absent. Unsampled video and unchecked person/table labels do not become ground truth. Prior annotations and review history are retained.

The original clip is actually 1920×1080 H.264 at 30 FPS, with 9,164 sequentially decoded frames, approximately 5 minutes 5 seconds. The importer preserves original hashes and decoder timestamps. Its edit/source history remains unspecified; check cuts, repeated frames and capture interruptions before using temporal windows. It stays in `iphone_same_venue_session_1` with the earlier gameplay clips. This session cannot also serve as an independent validation/test session.

## What the negative diagnostic found

The existing BlurBall engine was replayed causally over every source frame: `[t-2,t-1,t]`, channel 2, two real warmup frames, zero lookahead. It selected 1,254 ball hypotheses, 13.68% of all frames, on this user-declared no-ball scene. There were 999 consecutive observation runs, with a longest run of 24 frames; these runs do not establish tracking identity.

These are suspected failures worth reviewing. Precision, recall and confirmed false-positive counts remain null because no source annotations were approved automatically. This is an offline replay; its adapter/GPU work times are not live capture-to-display latency. The report audits source, model, runtime, journal, index and completion hashes, full frame IDs and nominal journal timestamps.

[Diagnostic](artifacts/no_ball_blurball_diagnostic.json), [native log](artifacts/no_ball_blurball_causal_v1.log), [original import](artifacts/no_ball_recording_import.json).

The miner selected 24 nonoverlapping five-frame windows around accepted hypotheses, using transparent temporal/spatial diversity. These 120 images were imported separately as `phone_452627e98a3c2652_2`, with selection/source/model/journal hashes and reasons attached. They were unreviewed at import; October 10 saved-review counts are above. This is not learned embedding clustering or automatically reviewed data. [Selection and provenance](artifacts/no_ball_hard_negative_selection.json), [separate import](artifacts/no_ball_hard_negative_import.json), [historical import audit](artifacts/no_ball_review_import_audit.json). Thirteen source frames overlap the uniform capture, so the 420 saved images represent 407 distinct frames; training must account for duplication.

## Green-channel experiment

`lighting_preprocessing.py` now provides green-channel replication into all three input channels, preserving image size and original pixels. `compare_lighting_filters.py` includes it alongside raw, gamma, CLAHE and adaptive variants. This is grayscale information from the green channel, distinct from tinting the displayed footage green. Synthetic geometry/channel/clipping tests pass. A filter cannot restore information already lost to clipping or blur.

No filter is enabled automatically. Accuracy comparisons on fixed reviewed bright/blurred/absent frames, causal BlurBall preprocessing parity and independent validation are still required. Raising a model score is not a precision improvement.

## Verification delivered

- Source/image-bound absence preview and confirmation, stale revision/token rejection, positive/occluded conflicts, atomic history and scene-label preservation passed API checks: [report](artifacts/review_absence_api_checks.json).
- Page handlers passed interval preview, thumbnails, explicit confirmation and pending-draft tests: [report](artifacts/review_absence_ui_checks.json).
- Combined absence-provenance/dataset adapter/integrity run passed 38 tests; this aggregate includes imported shared test cases: [log](artifacts/review_absence_training_tests.log).
- Green-input and sequential-import regressions passed: [log](artifacts/green_filter_and_import_tests.log). Seven separately authored journal/mining tests passed: [log](artifacts/no_ball_hypothesis_tests.log).
- Selection import and journal regressions passed 12 synthetic tests, including stale evidence, count/FPS mismatch, ranges, source/session checks and exact-selection resume: [log](artifacts/review_selection_import_tests.log). The actual selected-frame/annotation audit passes without creating labels.
- Production frontend compile, type/lint and route generation passed: [build](artifacts/no_ball_frontend_build_final.log).
- The separate packaged app passed real Chromium preview/confirmation/save, source switching, 360/360 fixture-frame analysis, matching footage/2D/3D frame/coordinates, draft persistence and owned-service cleanup with developer runtime directories removed from PATH: [report](artifacts/no_ball_desktop_smoke_final.json), [review screenshot](artifacts/no_ball_desktop_smoke_final.absence.png).

Packaged tests used isolated fixtures, not the user's real labels. The camera stayed unauthorized. The native dialog was tested at its API boundary; manually clicking the Windows picker and a prolonged live soak remain separate checks. The test initially interrupted review navigation early; it now waits for the completed idle state. A filesystem sandbox denied one build directory operation; the normal-runtime rebuild passed without changing filesystem/security handling. The [delivery audit](artifacts/no_ball_release_delivery.json) verifies packaged desktop code, native binary and engine hashes; the frontend build is `OlNGJGTFv7aXap_T2iuGh`.

## Offline hybrid foundation

The new native paired recorder runs both unchanged engines on the same original decoded image, retains every postprocessed proposal and its source-pixel center, and records actual source PTS, image/model/source hashes, independent scores and selection reasons. BlurBall uses the causal `[t-2,t-1,t]` window with two warmup frames. The pure conservative selector passed [25 contract checks](artifacts/ball_candidate_selection_checks.json). It chooses an existing model coordinate; it never averages incomparable scores or blends distant coordinates. These are unfitted research rules, outside the production backend.

The full 272-frame official `22_003` development clip exposes a severe recall failure at the existing strict **less-than-4-source-pixel** localization criterion. Of its 270 visible-ball references, standalone causal BlurBall correctly localized 254, standalone YOLO 12, and the experimental selector only 5. Experimental precision is 100% on those five predictions, but recall is only 1.85%, versus BlurBall's 94.07%. This fails promotion; requiring unreliable complementary evidence must not veto nearly every valid BlurBall observation. [Paired metrics and failed gate](artifacts/paired_ball_official_22_003_v1/metrics_with_runtime_provenance.json). This single reused clip and checkpoint with unknown training lineage do not establish independent accuracy.

The complete [paired no-ball rerun](artifacts/paired_ball_no_ball_v2/metrics_with_runtime_provenance.json) accounted for every one of the 9,164 source frames with strictly increasing source PTS, zero lookahead and no identical consecutive decoded images. YOLO selected 2,557 hypotheses, causal BlurBall 1,254 and the experimental selector 15. Accuracy and promotion gates remain null without reviewed references; these counts cannot demonstrate improved precision. No identical consecutive images does not establish an unedited video. The journal audit recomputes candidate selection and index integrity; it does not independently decode and recompute every image digest. [Twenty-five scorer tests](artifacts/paired_ball_scorer_tests_runtime_provenance.log) cover frozen policy, score boundaries, corruption, invalid references and runtime-hash mismatch. Unicode/space source paths pass a [12-frame replay](artifacts/paired_ball_unicode_fixture_v1/metrics.json); [500-frame progress replay](artifacts/paired_ball_progress_prefix_v1/metrics.json) verifies actual shared index bytes and the archived executable hash.

The initial no-ball paired run was interrupted after a mistaken directory-size reading. Direct journal inspection subsequently verified **7,204** consecutive indexed frames, not zero. The earlier stall claim is withdrawn; [the correction](artifacts/paired_ball_no_ball_v1_correction.json) preserves the original log and termination snapshot. The traced 12-frame prefix and complete rerun returned successfully. Tracing and flushed progress are observability changes, not detector fixes. Each run records the evaluator's runtime hash; older overwritten binaries were not archived, so their current-file hashes cannot be reverified. Subsequent evaluator builds retain a binary snapshot. Native YOLO initialization still has inherited unchecked CUDA setup returns; this remains an operational acceptance limitation.

## Training readiness and remaining work

The historical October 9 [saved-label audit](artifacts/real_training_readiness_v3.json) found 750 saved images across 14 complete imported captures, representing 732 distinct source-frame pairs. At that time, nineteen frames passed the reviewed ball-positive validator, with zero verified absences or reviewed CNN targets. The October 10 audit above supersedes those label counts. Thirty scene annotations were structurally accepted by the earlier inventory, which does not prove that their labels are correct. There is still no frozen independent train/validation/test session split. Model suggestions remain excluded.

`audit_training_readiness.py` is a new read-only command. It verifies original source hashes and decoded JPEG pixels, deduplicates source/frame and image identity, reports conflicting reviewed duplicates and split leakage, and rejects metadata changes during the audit. Invalid CNN geometry does not erase an otherwise valid ball box. [Nine meaningful tests passed](artifacts/training_readiness_tests_final.log). The counts remain separate from temporal continuity, label correctness and model accuracy. Run instructions are in [the training guide](TRAINING_LIVE.md).

The real upstream network now also passes a synthetic forward/loss/backward/optimizer smoke for [MSE](artifacts/blurball_training_smoke_mse_v2/report.json) and [quality-focal loss](artifacts/blurball_training_smoke_qfl_v1/report.json). Each uses the audited RGB three-frame fixture adapter, one artificial visible-ball window and one blank window. All 212 output-connected parameter tensors have finite gradients; 206 parameter tensors change during each in-memory Adam step. The inspected network allocates 39 final auxiliary fusion parameter tensors whose outputs are not consumed by `out_scales=[0]`; exactly this source-derived unused set is recorded, while any unexpected missing gradient fails. The first overly broad all-parameter-gradient assertion and its full traceback are preserved in `artifacts/blurball_training_smoke_mse_v1/`.

Quality-focal uses the inspected loss directly with its dictionary API. The upstream `HeatmapLoss` wrapper passes tensors to that dictionary API and is not used. MSE follows the inspected `train_blur.yaml` loss behavior. This verifies the explicit smoke path, not the complete Hydra runner, real fine-tuning, additional dependency installation or improved detection. Model/config/adapter/loss hashes are recorded; the original checkpoint remains unchanged and no candidate weights are saved.

Negative confirmation and detector-box review have now progressed as recorded above. Reviewed visible CNN gameplay targets, audited temporal continuity, independent later sessions and real candidate fine-tuning remain required. A successful synthetic optimizer smoke does not replace those dependencies. Compact-model distillation follows validated teacher fusion. Physical calibration/height, uncertainty coverage, event accuracy, spin identifiability, live latency and the remaining advanced features retain their explicit statuses in [the complete ledger](REQUIREMENTS_LEDGER.md).

The final [delivery audit](artifacts/hybrid_training_delivery.json) rechecks the
workspace and packaged native backend and all three engine hashes. They remain
unchanged. It records the scored evidence hashes, 25 native selector checks,
25 scorer tests, nine readiness tests and both synthetic optimizer smokes. The
complete action plan and final whole-chat closure remain unfinished.

## October 10 — next implementation milestone

[The v2 implementation report](HYBRID_V2_IMPLEMENTATION_REPORT.md) supersedes the
first selector's results: full native replay matches BlurBall's development
precision/recall but fails venue-negative acceptance (75 false detections among
284 reviewed absences, versus BlurBall 71 and YOLO 42). No production promotion
occurs. Actual pretrained YOLO11n features now feed a causal three-frame heatmap/
visibility prototype; 15 tests and two real-data optimizer checks pass without a
saved candidate. A separate 15-test YOLO experiment runner verifies fresh private
fit data and frozen provenance, with full fitting still awaiting independent
session splits. Existing reviewed labels and deployed weights remain intact.
