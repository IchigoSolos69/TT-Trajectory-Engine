# Local Windows application

Double-click `launch_engine.bat` or `desktop/dist/TT Trajectory Engine/TT Trajectory Engine.exe`. Keep the entire release folder together. This is a portable Windows application; no installer, browser, Node.js or Python installation is needed to run the built release.

Choose **Open a game** to select a MOV, MP4, AVI, MKV, WebM or M4V recording. The native file picker also opens with Ctrl+O. Choose **Start a live session** for your camera; capture remains off until **AUTHORIZE WEBCAM** is pressed. The desktop controls let you change source mode or end the session. Closing the application stops its own C++ engine and dashboard service.

The existing footage, radar, 3D, physics, strike history and training-review interfaces run inside the application. C++ remains authoritative for inference and analytics. The desktop shell does not change the detector weights or enable rejected experimental candidates.

## Data and ownership

When this release is launched from inside the project, it uses the existing project data directory. Saved captures, reviewed annotations and completed analyses remain available. When the release folder is moved elsewhere, it uses `%APPDATA%/TT Trajectory Engine`. **Local data** or **Help → Open local data and logs** opens the active directory.

Review drafts are saved separately under `artifacts/desktop/review-drafts.json`, never treated as reviewed training labels. Desktop drafts survive restarts and changes to the internal port. Unsaved drafts from the old browser profile remain in that browser profile; they are not imported automatically. Saved annotations are shared through the existing files.

The application chooses available loopback ports for its own services. It does not stop or attach to processes already listening on 3000/9002. HTTP and WebSocket transport remains internal to the local app; no hosted website or external browser is involved. The renderer has no Node.js access, runs sandboxed with context isolation, and exposes only specific native actions. Remote navigation, remote requests and browser camera permission are blocked. OpenCV handles the explicitly authorized camera.

## Build and tests

The Windows development prerequisites remain the existing C++/CUDA/TensorRT/OpenCV toolchain plus Node/npm. Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_desktop.ps1
```

The build uses `tt_intelligence_candidate.exe` as a separately linked binary to avoid replacing a running production engine. Despite its target name, the application does **not** pass `--experimental-intelligence`. Next.js builds into `.next-desktop`, leaving the browser build intact. Packaging includes Electron, the two deployed detector engines, required native DLLs and the standalone dashboard. The old 5.57 GB embedded video is excluded; recordings are selected from disk.

Packaging refuses to overwrite an existing release. Close the desktop app and move the old release folder aside before building another version. The release manifest records model/backend hashes, the frontend build ID and bundled runtime dependencies. These serialized TensorRT engines were tested on this machine's RTX 5050; a compatible NVIDIA GPU and driver remain required. Portability to a different GPU is not established.

```powershell
node --test desktop/tests/*.test.cjs
$env:TT_DESKTOP_SMOKE_VIDEO=(Resolve-Path -LiteralPath 'artifacts/replay fixture café.avi').Path
$env:TT_DESKTOP_TEST_EXE=(Resolve-Path -LiteralPath 'desktop/dist/TT Trajectory Engine/TT Trajectory Engine.exe').Path
$env:TT_DESKTOP_CLEAN_PATH='1'
node desktop/tests/smoke.cjs
```

See `DESKTOP_VERIFICATION.md` for the concrete verification and limits. Automated smoke tests never authorize a camera.
