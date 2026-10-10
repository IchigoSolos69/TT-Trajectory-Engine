# Precision implementation release — 2026-10-09

The local Windows app now includes the published BlurBall temporal CNN as an explicit experimental detector, easier blur-streak review, a repaired annotation-save origin check and a bounded C++ trajectory forecast baseline. Local fine-tuning and a learned neural forecaster have **not** been trained or promoted. The complete advanced project plan remains unfinished.

## Run the delivered app

Open `desktop/dist/TT Trajectory Engine/TT Trajectory Engine.exe`, or the existing desktop launcher. Choose **Ball model → BlurBall · experimental**, then select a recording. The same selector is available during a session. YOLO remains the default. Live inference uses only past/present frames and opens the camera only after authorization; recording analysis may use two frames of lookahead. The prior release is retained under `desktop/dist/TT Trajectory Engine before precision`.

## Implemented and checked

| Change | Evidence |
|---|---|
| Native BlurBall FP32 inference, centroid/streak telemetry and separate model selection | Fixed tensor contract, exact-input numerical parity, source-window/reset tests and final main/controller tests. Original YOLO/scene hashes are unchanged. [Implementation](BLURBALL_IMPLEMENTATION.md). |
| Complete recording analysis and synchronized views | Final app analyzed 360/360 YOLO fixture frames and 272/272 BlurBall frames. At paused frame120, footage/2D/3D share analysis identity and the same authoritative table coordinates. Source switching creates a new analysis. [Packaged runtime checks](artifacts/blurball_desktop_smoke_final.json). |
| “Same-origin review required” fix | Genuine Chromium renderer POST/save/reload returned HTTP200 at the app's dynamic loopback origin. Direct Host/Origin, alias/port, path and revision protection remain enforced. [Origin report](TRAINING_REVIEW_ORIGIN_FIX.md). |
| Optional one-drag blur annotation | Actual pointer drag and GUI Save persisted reviewed endpoints/center/half-length while preserving the existing ball box. [Rendered save](artifacts/blurball_desktop_smoke_final.streak.png). A streak-only visible label never becomes an empty-ball YOLO negative. |
| Audited CNN training handoff | The [dataset adapter](python_tools/blurball_reviewed_dataset.py) revalidates exported rows against original labels, images, indices and session manifests. It yields chronological RGB inputs `[9,288,512]` and heatmap targets `[3,288,512]`, with the inspected affine/normalization, correctly transformed streak extent/axis and a versioned continuous target that retains tiny positive streaks. Unknown/occluded/partial labels cannot become absent-ball targets. Structural, mathematical and independent integrity tests pass on synthetic fixtures; no fitting or real-data accuracy is claimed. [CNN/NN handoff plan](HIGH_PRECISION_IMPLEMENTATION_PLAN.md). |
| C++ short forecast | Recent observed positions only; motion/contact-candidate resets and 50ms expiration. Frontend renders stored C++ points, off by default. [Rendered frame30](artifacts/blurball_desktop_smoke_final.forecast.png). |
| Application lifecycle | Final sandboxed package ran with clean PATH, no renderer errors, invalid-file/cancel preservation, model switch cleanup and normal close releasing owned services/ports. Six desktop unit tests pass. Native picker handler was exercised with a selected fixture; actual manual OS dialog selection and live capture were not tested. |

The final native backend SHA-256 is `85b693c29b6ada72d1444b86c325f87bd3aff846fddeace903b6fec7919b3366`; frontend build is `mqJ8YZ4aT5bj980UF9QJg`. [Delivered release manifest](artifacts/blurball_release_delivery.json).

The final dataset run passes **31 tests**, including 16 independent integrity checks. [Test output](artifacts/blurball_dataset_adapter_tests.log), [synthetic target/lineage audit](artifacts/blurball_dataset_adapter_v1/adapter_audit.json). This adapter is a local Python training tool; the tested desktop package is unchanged by its addition.

## Accuracy evidence and limits

On one official labeled 272-frame clip at strict localization distance <4 source pixels, native BlurBall recording precision/recall are **98.48%/96.30%**, and causal latest-frame selection is **97.69%/94.07%**. There are only two absent-ball labels, and the checkpoint's training manifest is unavailable. This cannot establish blind venue/live generalization or justify default promotion. Independent Python/native OpenCV preprocessing has a retained version-dependent difference, despite tight identical-input inference parity. [Full evaluation](BLURBALL_EVALUATION.md).

The original dimmed iPhone video was replayed chronologically through source frame1694 without a brightness filter. At frame1685 the selected source center is `(1087.95, 578.53)` px, visually on the ball streak rather than the ceiling. All15 frames1680–1694 have selected observations, but those labels remain unreviewed: this is a qualitative diagnostic, **not precision/recall**. [Original-frame overlay](artifacts/blurball_research/phone_7943_causal/frame_1685.jpg), [scope and lineage](artifacts/blurball_research/phone_7943_causal/summary.json). Original annotations and footage were not edited.

The causal forecast test accounts for all272 frames. Across391 paired future targets, the quadratic baseline lowers mean error to4.792px versus constant velocity5.261px, but **worsens RMSE to10.863px versus10.245px**. Per-horizon p95 also worsens. It stays experimental/off by default. The recording-lookahead test is reported separately and cannot demonstrate unseen-future forecasting. Neither test supplies calibrated uncertainty, physical3D or contact-ground-truth accuracy. [Causal metrics](artifacts/blurball_research/imageplane_forecast_causal/metrics.json).

Offline multi-pass GPU inference timing is reported separately in the implementation report. **Single-session live capture-to-display accuracy and latency remain unmeasured**; no120FPS end-to-end claim is made.

## Next precision milestones

1. Obtain reviewed consecutive positive, absent-ball, bright-distractor, blur and occlusion examples. All ten existing iPhone clips belong to one session; independent validation/test sessions remain necessary. New live captures also need explicit session metadata before the training audit can accept them.
2. Connect the tested adapter to a reproducible CNN training runner with a complete compatible PyTorch/torchvision environment, explicit loss and saved settings. Freeze the independent evaluation labels and protocol before fitting. The paper-described quality-focal loss differs from the inspected upstream default MSE; the adapter requires an explicit choice.
3. Fine-tune and evaluate native causal inference on fixed held-out sets before promotion. Do not increase a displayed confidence score to conceal wrong detections.
4. Establish independent table/net landmarks and native MOV true-PTS/seek integrity before stronger physical strike/height/spin claims.
5. Compare a learned temporal/mixture forecaster against the strongest baseline only with valid future/contact references and coverage tests.

Both attached PDFs and all12 supplied URLs are accounted for in [the paper review](RESEARCH_APPLICATIONS.md). Three IEEE references remain unresolved and two full methods are inaccessible; their algorithms/constants were not guessed. [The requirement ledger](REQUIREMENTS_LEDGER.md) retains remaining research, data and integration gates. This release is progress toward the full plan, not final whole-project sign-off.
