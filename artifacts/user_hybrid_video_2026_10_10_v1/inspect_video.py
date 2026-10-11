"""Read only user video; extract chronological evidence frames with decoded PTS."""
import hashlib
import json
from pathlib import Path
import cv2

source = Path(r'C:\Users\Adi\Videos\2026-10-10 16-13-40.mp4')
output = Path(__file__).resolve().parent
cap = cv2.VideoCapture(str(source))
if not cap.isOpened():
    raise RuntimeError('Cannot decode supplied source')
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
selected = {min(count - 1, max(0, round(t * fps))) for t in (0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60)}
selected.add(max(0, count - 1))
frames = []
decoded = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if decoded in selected:
        dest = output / f'frame_{decoded:06d}.jpg'
        if not cv2.imwrite(str(dest), frame):
            raise RuntimeError('Evidence frame save failed')
        frames.append({'zero_based_frame_id': decoded, 'decoded_pts_seconds': pts,
                       'width': frame.shape[1], 'height': frame.shape[0],
                       'evidence_file': str(dest)})
    decoded += 1
cap.release()
hash_value = hashlib.sha256()
with source.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        hash_value.update(block)
report = {'source': str(source), 'source_sha256': hash_value.hexdigest(),
          'reported_fps': fps, 'reported_frame_count': count, 'decoded_frame_count': decoded,
          'extraction_only': True, 'training_imported': False,
          'opencv_version': cv2.__version__, 'frames': frames}
(output / 'probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
