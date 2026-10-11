# User screen recording inspection, 2026-10-10

Original: `C:\Users\Adi\Videos\2026-10-10 16-13-40.mp4`

- Decoded all 2,731 frames. Reported frame rate 60 FPS; image size 1920×1080. Last decoded PTS 45.5 seconds. Source hash and sampled frame PTS are in `probe.json`.
- This is a screen recording of the local application's dashboard. Its embedded recording is `Top-Points-of-SnapYT.App.mp4`, shown as 11,227 source frames / 449.1 seconds. The supplied screen recording is not an original training clip and was not imported into training.
- At screen-recording 15 seconds (`frame_000900.jpg`), the application's selector explicitly says **YOLO**. Its selected underlying source frame is 249, a close-up of a player. A yellow `BALL CANDIDATE score 27%` marker appears near that player's lower face/neck. Earlier selected frame 74 (`frame_000300.jpg`) has a broad cyan polyline crossing advertising and court regions. These are visible overlays, not a quantitative accuracy evaluation.
- At screen-recording 20 seconds (`frame_001200.jpg`), the overlay explicitly says **BLURBALL · EXPERIMENTAL · 2-FRAME LOOKAHEAD**. Its selected source frame is 105, a court shot. It shows a candidate peak 0.99 near the table and `blur extent 46.7 px · undirected axis`.
- At screen-recording 40 seconds (`frame_002400.jpg`), BlurBall similarly shows a candidate near the table (selected source frame 604, peak 0.98 and blur extent 47.7 px).
- At 25, 30, and 45.5 seconds, the BlurBall view shows close-ups between rallies and **SEARCHING FOR BALL** rather than a yellow ball marker. At 30 seconds the selected frame is 355; at 45.5 seconds it is 704. This is useful qualitative evidence about behavior, not proof of true-negative status without reference annotations.
- In inspected frames both model sections show **TABLE CALIBRATION UNAVAILABLE**. Strike history says it is waiting for a detected strike and calibrated table; stroke count is zero, speed unavailable, rotation unresolved. Thus improved ball localization by itself does not resolve current physical trajectory/speed analytics.
- Model sections are played at different underlying source frames and different analysis coverage. The observed BlurBall UI uses two future frames, so this recording cannot establish a like-for-like causal/live comparison or capture-to-display latency.
- No models, weights, labels, splits, app processes, camera state, or original media were changed. Only evidence frames, this note, a probe report, and the extraction script were created in this folder.

Ground-truth precision/recall/localization error was not measured from this screen recording. Model scores and heatmap peaks are not calibrated correctness probabilities and are not directly comparable between model families.
