# Desktop application verification — 2026-10-09

The portable Windows release has been executed on the current machine with developer TensorRT/OpenCV/CUDA/Node directories removed from its PATH. It loaded its bundled detector engines, analyzed the Unicode/space-containing 360-frame AVI fixture, and rendered the dashboard in an actual sandboxed Electron BrowserWindow.

## Evidence

- `artifacts/desktop_native_build.log`: native candidate linked successfully with explicit data-root support.
- `artifacts/desktop_frontend_build.log`: production compile, type/lint checks and standalone route build passed.
- `artifacts/desktop_unit_tests.log`: 7 desktop/frontend test files/cases passed, including process ownership, source validation, IPC trust, atomic ordered draft storage and review-coordinate regressions.
- `artifacts/desktop_packaged_smoke.json` and `.log`: packaged runtime checks; `clean_runtime_path: true`, `packaged: true`, no renderer errors.
- `artifacts/desktop_packaged_smoke.png`: actual desktop renderer screenshot after selected-frame paint.
- `artifacts/desktop_recording_regression.log`: existing native complete-recording/indexed replay regression passed.
- `desktop/dist/TT Trajectory Engine/release-manifest.json`: unchanged ball/scene model hashes, binary fingerprint and bundled runtime inventory.

## Checks passed

The native launcher button traversed preload/IPC to start live mode, with `camera_active: false`. End session stopped that owned backend. The recording selection handler accepted the fixture through the native-dialog API boundary, analyzed exactly 360/360 source frames and supported indexed seeking. Frame 120 was checked in the painted footage canvas, 2D radar and 3D view; the two spatial views had identical table coordinates. This establishes synchronization, not physically correct 3D reconstruction.

Training-review navigation/API, dashboard assets and favicon returned HTTP 200. File-backed review drafts survived a fresh store instance and ordered concurrent saves. Invalid and canceled selections preserved the active recording. Closing the native window stopped both application-owned services, removed their PIDs and freed their selected ports. The existing processes on 3000 and 9002 remained healthy and unchanged.

The native-dialog selection/cancel results were supplied by the automated test at the Electron API boundary. Manual clicking through the actual Windows dialog remains a separate acceptance check; the test does not claim that interaction occurred. No actual webcam capture, prolonged soak, new live accuracy measurement or GPU throughput benchmark was performed for this conversion.

## Failures investigated and repaired

The first live startup failed its identity check because telemetry reported the workspace rather than the selected data root. Explicit data-root identity now reaches live telemetry and recording workers. An initial test incorrectly compared presentation generation before versus after seeking; it now checks the current source and analysis identity around invalid-file loading. Windows filesystem sandbox restrictions blocked the atomic draft rename in a unit run; the same test passed in the normal runtime environment without weakening atomic writes.

Initial packaging attempted to copy the old 5.57 GB public video and exhausted available space for that copy. Only generated partial-release/standalone copies were removed; original footage remained untouched. Packaging now excludes media assets and includes user-selected recordings by reference. A case-sensitive lookup of `cublasLt64_13.dll` was corrected to match the case-normalized dependency index.

## Remaining limits

The release is unsigned and portable, not an MSI/NSIS installer. It requires Windows and a GPU/driver compatible with the bundled TensorRT engines. Normal close/end-session cleanup is verified; forced desktop-process termination and a 30-minute soak are not verified. The earlier accuracy, calibration, uncertainty and advanced-analytics acceptance failures remain in `REQUIREMENTS_LEDGER.md`; desktop conversion does not resolve them.

## October 9: native BlurBall and final precision package

The final package is installed in the standard application folder, with the previous release retained as `desktop/dist/TT Trajectory Engine before precision`. Native backend SHA-256 is `85b693c29b6ada72d1444b86c325f87bd3aff846fddeace903b6fec7919b3366` and frontend build is `mqJ8YZ4aT5bj980UF9QJg`. The package guard verifies a completed frontend build before creating its output. [Delivery manifest](artifacts/blurball_release_delivery.json).

[Final packaged runtime checks](artifacts/blurball_desktop_smoke_final.json) passed with developer runtime directories excluded: 360/360 fixture frames, 272/272 BlurBall recording frames, actual model-selector restart/cache identity, invalid/canceled recording preservation, exact analysis/frame/table-coordinate agreement across footage/2D/3D and owned service cleanup. No renderer errors occurred. The browser supplied its genuine Origin header for annotation POST/save/reload, which returned HTTP200. An actual pointer drag and GUI save persisted a reviewed source-coordinate blur streak without changing the previous ball box. The off-default forecast rendered exactly the native frame-30 points.

Hidden-window capture initially returned a stale compositor image despite correct DOM assertions. The final test primes capture, waits for painting, rechecks current UI state and then captures. Root inspected the [saved streak](artifacts/blurball_desktop_smoke_final.streak.png), [forecast frame30](artifacts/blurball_desktop_smoke_final.forecast.png) and [BlurBall frame120](artifacts/blurball_desktop_smoke_final.blurball.png). The native picker remains tested at its API boundary, not by manually clicking the OS dialog. Neither this package check nor the causal file replay is a live camera accuracy/latency test.

[The precision release report](PRECISION_RELEASE_REPORT.md) records model results, the qualitative original-phone improvement, forecast regressions and outstanding training/calibration/physical/live gates. YOLO and scene weights remain unchanged; BlurBall is separately selectable and experimental. No original review labels were rewritten.

## October 9: easier negative review package

The separate `desktop/dist/TT Trajectory Engine training review v2/` package is selected by `launch_desktop.bat`, preserving the existing standard release. Its native/backend and all three detector engines retain the precision release's hashes. The production review build adds source/image-bound interval absence confirmation; the native launcher and Session menu can open Training review without starting a backend or camera.

[Final clean-PATH package checks](artifacts/no_ball_desktop_smoke_final.json) passed with no renderer errors: direct review navigation with `engine.child == null`, explicit fixture interval preview/checkbox/save, scene classes left unreviewed, existing streak pointer/save regression, 360/360-frame source analysis, exact paused frame/coordinate agreement, invalid/canceled source preservation, file-backed drafts and service cleanup. Root inspected [the interval review screenshot](artifacts/no_ball_desktop_smoke_final.absence.png). A navigation race in the test was repaired by waiting for review loading to finish before replacing its page. These tests do not mark any actual recording absent or establish detector accuracy.

The [new implementation report](HYBRID_PRECISION_IMPLEMENTATION.md) records the original no-ball import and full causal hypothesis diagnostic. No model weights changed. Normal close/reopen is necessary to leave an already-running older version because single-instance behavior otherwise focuses it. Camera hardware, manual OS-dialog clicking and prolonged soak remain unverified.
