# Detection failure follow-up — 9 October 2026

The ceiling-ball false positive remains a model error. This update fixes contained lower-leg duplicates, stale recording scene boxes and raw detections bypassing the experimental tracker. It does **not** claim improved ball precision/recall or completed venue training.

## Reproduction and changes

| Issue | Evidence and result |
|---|---|
| Ceiling feature labeled as ball | Native C++ TensorRT on original IMG_7943.MOV frame 1848 reproduces score **0.395996**, center **(1439.25, 290.25) px**. The unchanged deployed model still proposes it. Default overlay now explicitly says **BALL CANDIDATE · UNVALIDATED**. Relabeling is provenance, not an accuracy fix. |
| Duplicate leg box | Frame 1845 has a 0.512 person proposal contained in the lower half of the 0.792 full-person box. Conservative C++ scene postprocessing suppresses only lower-confidence, small, almost fully contained lower-half fragments. Suppressed proposals remain in telemetry with a reason. Similar-sized overlapping people, upper-body candidates and stronger small observations remain. This is a heuristic, not a learned identity association. |
| Old scene boxes on newer footage | Offline scene inference previously ran every four frames; frame 1848 displayed frame 1845's scene boxes. It now runs for every analyzed frame. Telemetry publishes `scene_frame_id`; live previews disclose older scene observations if present. |
| Rejected ball still drawn as measured | Experimental overlays previously used the raw top detector result despite tracker rejection. Overlay/spin/image velocity now use the authoritative accepted observation; raw rejected proposals cannot bypass it. Filtered image/table coordinates refer to the same trajectory sample. Experimental tracking remains opt-in and can still accept false detections. |
| Frontend velocity interpretation | The footage velocity arrow now renders C++ image velocity, rather than deriving another velocity from browser-delivered preview points. This does not validate legacy physical strike-speed estimates. |
| Grey table lacks calibration | The scene model detects a table box; the current contour refinement requires blue/green surface evidence. A high box score cannot establish valid corners or homography. Grey-table landmark calibration remains unfinished; no default rectangle or invented pose was added. |

## Verification

- Native scene unit checks cover duplicate legs, overlapping people, upper-body ambiguity, stronger small candidates, non-contained people, class-aware NMS and unavailable calibration.
- The same **779 test_2 regression frames** retain **2612 TP / 19 FP / 39 FN** for people and **779 TP / 0 FP / 0 FN** for table boxes at confidence 0.4 / IoU 0.5. Precision/recall are unchanged; this is existing dataset regression evidence, not a new blind test or live accuracy evaluation.
- Native replay in default and experimental modes accounts for source frames **1–1854** consecutively, with matching scene frame IDs and auditable suppression. **43 cases** exercise a rejected experimental observation that would otherwise have been drawn from the default raw detector output.
- The original recording and both model SHA-256 hashes remain unchanged. Production re-analysis completed **2114/2114** source frames. Native MOV true-PTS/seek acceptance is still a separate unresolved requirement; decoder frame IDs alone do not prove exact image identity.
- The existing complete-recording acceptance test passed pause, indexed seek, stepping, rate-independent analytics, source protection/isolation, invalid-file preservation and compatible-cache reuse.
- Candidate and production native builds and frontend production build passed. Current CSS/JS and favicon pass HTTP checks. Browser inspection confirms frame 1848's canvas/controls, two person boxes and explicit candidate status, with no captured browser errors. The production app was reopened on the same recording and paused at the original selected frame 1848.

Evidence: [summary](artifacts/detection_failure_20261009/verification_summary.json), [native authority](artifacts/detection_failure_20261009/native_authority.json), [scene regression](artifacts/detection_failure_20261009/test2_scene_after.json), [production telemetry](artifacts/detection_failure_20261009/production_runtime.json), [browser inspection](artifacts/detection_failure_20261009/browser_runtime.json), [browser image](artifacts/detection_failure_20261009/browser_overlay.jpg).

These are offline native diagnostics and browser smoke checks. They establish neither multi-pass GPU throughput nor single-session live capture-to-display latency. No camera was opened.

## What is needed to fix ball accuracy

Current annotation audit: **30 reviewed ball frames, 19 usable positives, 0 usable absent-ball negatives, 11 excluded ambiguous/conflicting frames**. The saved review labels have not trained or replaced the deployed model. All ten supplied phone recordings belong to one session, so they remain together in one dataset split.

A new unreviewed 15-frame batch is available in Training review as **IMG_7943.MOV · Ceiling false ball and duplicate leg: frames 1840–1854**. Its capture ID is `phone_c343f0cdb72df492_3`. The earlier 1680–1694 batch remains intact. No existing annotation was overwritten and no predicted box became ground truth.

Use ball-only review and 100%/200% zoom: place the box on the actual visible ball, save and advance; mark **Not in frame** only when you can verify absence; keep hidden/uncertain frames excluded. Correctly labeled positive frames also teach the detector that the visible ceiling feature is background. Add genuinely absent-ball footage to measure and train against false positives.

Candidate training/export still needs reviewed data and separate validation/test sessions, followed by native precision/recall and export-parity checks. Existing test_1/test_2 footage remains regression material. Earlier brightness-filter and fixed-grid experiments did not recover the reported ball and were not deployed. Neither arbitrary threshold changes nor a ceiling mask substitutes for measured model improvement.
