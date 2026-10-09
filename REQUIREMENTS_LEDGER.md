# Implementation ledger — approved complete action plan

Status is evidence-based: verified means the stated check passed, not universal accuracy. No requested feature is silently dropped. The final whole-chat audit is still pending.

| ID | Requirement | Current status | Next acceptance evidence |
|---|---|---|---|
| DESKTOP-1 | Replace browser workflow with a local Windows application | Packaged EXE runtime verified | `DESKTOP_VERIFICATION.md`; native window, independent ports, bundled runtimes, source controls, draft persistence and normal close tested; manual OS picker and soak pending |
| SOURCE-1 | Dataset-only simulation; authorized camera-only live | Verified earlier | Re-run source routing after recording-controller integration |
| OPS-1 | Reuse healthy project services; controlled restart; unrelated port protection | Verified isolated runtime | `artifacts/launcher_runtime.json`; inspection errors fail closed; 30-minute soak pending |
| REPLAY-1 | Every source frame analyzed independently of playback | Implemented; 360-frame runtime verified | `artifacts/complete_recording_runtime.json`; full-length/disk-fault acceptance pending |
| REPLAY-2 | Local Windows recording picker; invalid file preserves session | Implemented; source switch observed | Deterministic picker selection including Unicode paths |
| REPLAY-3 | Smooth footage, pause/restart/step/0.25–4× | Native replay verified | Cache reuse across process restart, step/seek and rate tests pass; rendered browser acceptance pending |
| VIEW-1 | Footage, 2D and 3D share one source frame/state | Verified presentation baseline | Re-run stale-decode/rapid-seek/reconnect browser checks |
| VIEW-2 | Real footage with automatic ball/person/table labels | Implemented; dataset evaluated | Live room/rally precision and negative tests |
| VIEW-3 | Clear realistic 2D/3D; camera-matched view | Partial | Pose-based view and reprojection checks; no fabricated height |
| STROKE-1 | Persistent velocity for each strike | Implemented; experimental event accuracy | Window-fit metrics and labeled event evaluation |
| SPIN-1 | Observed or identifiable motion-based spin; no speed-derived RPM | Observed camera-axis component implemented | Conditional motion fit and real reference data; unresolved otherwise |
| PHYSICS-1 | Realistic repeated/second bounce | Verified standalone sandbox | Shared C++ integrator analytical/convergence tests |
| INT-1 | Covariance-aware trajectory/states/certainty display | Experimental; coverage fails | 92–98% nominal-95% coverage with paired errors and region size |
| INT-2 | Persistent calibrated table digital twin | Experimental corner stabilization | Learned landmarks, intrinsics, independent reprojection metrics |
| INT-3 | Rally event graph, flight edges and synchronized timeline | Experimental rules | Per-class precision ≥95%, recall ≥80%, ±50 ms one-to-one matching |
| INT-4 | Active learning, diverse hard examples, reviewed lineage | Partial; bursts and extended review implemented | Burst/gap tests and geometry/events API/history tests pass; real capture/review UI and labeled live splits pending |
| INT-5 | Stroke intelligence and persistent player association | Partial | Incoming/outgoing fits, placement, identity uncertainty and labeled checks |
| INT-6 | Strike-seeded counterfactual replay/envelope | Remaining | Shared initial state, isolated parameter changes, explicit unknown assumptions |
| INT-7 | Explainable highlights, clips/JSON, explicit live replay buffer | Candidate intervals only | Source-aligned exports, padding/buffer truncation tests |
| INT-8 | Bounded temporal neural ball recovery | Candidate trained; promotion rejected | Native causal recovery precision ≥95%, no detector/negative regression |
| INT-9 | Blur/motion-aware ROI diagnostics | Experimental | Labeled blur bins and measured per-bin error/precision/recall |
| INT-10 | Contact-aware physics consistency gate | Motion gate prototype | Valid contacts retained; implausible jumps rejected with reasons |
| INT-11 | Player-relative tactical heatmaps | Remaining | Validated contacts only, correct perspective, raw/weighted counts |
| INT-12 | Adaptive inference scheduler and stage measurements | Remaining | Accuracy-preserving cadence and bounded live queues/latency |
| MODEL-1 | Better ball/person/table precision, recall and confidence quality | Dataset detector improved; live unverified | Fixed match/live/blur/occlusion/furniture comparison and calibration |
| MODEL-2 | Complete next temporal training experiment | Experiment verified; promotion rejected | v2 native parity passes; recovery precision 83.33%, below 95%; `artifacts/temporal_pipeline_v2_matched.json` |
| DATA-1 | Explicit training recording control | Implemented | Camera recording/release/disk failure tests and reviewed live data |
| DATA-2 | Exact live-training capture steps and warnings in GUI | Implemented; component runtime/build verified | Optional setup checklist, ordered coverage routine, mode limits and labeling/split warnings; `artifacts/training_capture_guide_checks.json`. Desktop visual check pending: app approval timed out |
| DATA-3 | Original phone recordings imported for review | Implemented; HTTP/data integrity verified | Ten MOVs, 300 unreviewed frames, source hashes/PTS, filename selection and same-session split guard; `artifacts/phone_recordings_import.json`, `artifacts/phone_review_runtime.json`. Reviewed labels and independent sessions remain required |
| MODEL-3 | Automatic background-dependent preprocessing | Experiment evaluated; not promoted | Nine sequential TensorRT frames compared across raw/gamma/CLAHE/adaptive inputs; all scores below 0.25. No accuracy labels, no default/filter/threshold change; `artifacts/phone_lighting_sequential/report.json` |
| DATA-4 | Review dimmed-footage ball misses and ceiling false positives | Diagnostic reproduced; targeted review implemented and browser verified | IMG_7943 frame 1685 approximately 0.376 ceiling score; filters/grid fail recovery. 15 unreviewed frames 1680–1694, Fit/100%/200% source-coordinate checks and browser zoom/scroll evidence; `artifacts/phone_dim_ball_diagnosis/report.json`, `artifacts/phone_failure_review_browser.json`. Reviewed labels, candidate training and accuracy gates remain unfinished |
| DATA-5 | Easier review while preserving training-label accuracy | Implemented; component/data/API/browser checks passed | Ball-only scope, one-click replacement, Save & next/shortcuts, bounded previous-box draft, persistent local drafts, wrapped review queue, optional scene/events. Preparation uses per-class reviewed flags and per-model datasets; ambiguous ball labels excluded. `artifacts/easy_ball_review_browser.json`, `artifacts/easy_review_data_audit.json`. No time-savings/label-accuracy/model-accuracy claim; independent session data and candidate training remain required |
| MODEL-4 | Reported ceiling-ball and duplicate-leg detection failures | Partial fix; ball accuracy unfinished | Conservative auditable person-fragment suppression, per-frame recording scene observations, experimental trajectory authority and explicit unchecked-candidate display. 779-frame scene regression unchanged; native/browser checks passed. Ceiling false positive and grey-table calibration remain unresolved; venue training awaits reviewed positives/negatives and separate sessions. `DETECTION_FAILURE_REPORT.md` |
| SYNC-MOV | Exact native MOV seek/frame/timestamp integrity | Remaining; issue identified | Python OpenCV random seeking disagrees with sequential frame/PTS mapping; metadata frame count differs from decode on IMG_7936. Native replay and true-PTS velocity acceptance required |
| CONTRACT-1 | C++ authoritative analytics; provenance and null unknowns | Partial | Versioned contract and removal of frontend analytical fallbacks |
| PERF-1 | Distinguish offline GPU throughput from live end-to-end | Documented | Per-stage timings, 30-minute soak, no unmeasured 120 FPS claim |
| AUDIT-1 | Re-read whole chat and all attachments after integration | Pending final closure | Requirement → implementation → test → result → limitation matrix |

## Fixed product decisions

Single camera first; conditional motion-based spin estimates; precision-first recovery; full recording analysis plus independent replay; both room tests and table rallies; C++ owns analytics. Unobserved height/spin remain unavailable. Candidate promotion requires evidence, not training mAP alone.

## Data dependency

Ten same-session iPhone gameplay recordings were imported as 300 initially unreviewed frames, plus two 15-frame targeted dimmed-footage failure batches. Review progress is stored separately and is never overwritten by import. Independent later sessions, representative webcam/room clips and reviewed labels are still required before claiming live accuracy. Existing test_1/test_2 sequences are development/regression material, not fresh blind tests. The grey tabletop also exposes the current blue/green-only tabletop refinement constraint; independent landmark calibration remains unfinished.

## Evidence retained

See `VERIFICATION.md`, `INTELLIGENCE_ROADMAP.md`, and their linked runtime/model reports. Temporal v1 causal validation adds 55 correct detections and 29 false positives; it remains out of the default pipeline. Phase completion and the whole-chat audit must not be declared while acceptance failures or missing data remain.
