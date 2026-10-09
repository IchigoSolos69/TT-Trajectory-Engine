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
