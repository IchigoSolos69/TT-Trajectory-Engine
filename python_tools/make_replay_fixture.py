"""Make a short real-footage replay fixture; not a model accuracy benchmark."""
from pathlib import Path
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import cv2

root = Path(__file__).resolve().parents[1]
target = root/'artifacts'/'replay fixture café.avi'
if target.exists():
    raise FileExistsError(target)
capture = cv2.VideoCapture(str(resolve_public_dataset_video('game_1')))
if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, 9000):
    raise RuntimeError('Cannot open fixture source')
writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'MJPG'), 120, (960, 540))
if not writer.isOpened():
    raise RuntimeError('Cannot create fixture')
for _ in range(360):
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError('Fixture source decode failed')
    writer.write(cv2.resize(frame, (960, 540)))
writer.release(); capture.release()
print(target, flush=True)
