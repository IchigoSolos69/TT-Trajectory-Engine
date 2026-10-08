# Implementation ledger — approved complete action plan

Status is evidence-based: verified means the stated check passed, not universal accuracy. No requested feature is silently dropped. The final whole-chat audit is still pending.

| ID | Requirement | Current status | Next acceptance evidence |
|---|---|---|---|
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
| CONTRACT-1 | C++ authoritative analytics; provenance and null unknowns | Partial | Versioned contract and removal of frontend analytical fallbacks |
| PERF-1 | Distinguish offline GPU throughput from live end-to-end | Documented | Per-stage timings, 30-minute soak, no unmeasured 120 FPS claim |
| AUDIT-1 | Re-read whole chat and all attachments after integration | Pending final closure | Requirement → implementation → test → result → limitation matrix |

## Fixed product decisions

Single camera first; conditional motion-based spin estimates; precision-first recovery; full recording analysis plus independent replay; both room tests and table rallies; C++ owns analytics. Unobserved height/spin remain unavailable. Candidate promotion requires evidence, not training mAP alone.

## Data dependency

No live capture directory was present at the last inspection. Representative room/rally clips and reviewed labels are required before claiming live accuracy. Existing test_1/test_2 sequences are development/regression material, not fresh blind tests.

## Evidence retained

See `VERIFICATION.md`, `INTELLIGENCE_ROADMAP.md`, and their linked runtime/model reports. Temporal v1 causal validation adds 55 correct detections and 29 false positives; it remains out of the default pipeline. Phase completion and the whole-chat audit must not be declared while acceptance failures or missing data remain.
