"""Confirm a received native JPEG depicts the exact declared source frame."""
import json
from pathlib import Path
import cv2
import numpy as np

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    meta=json.loads((root/'artifacts/preview_alignment.json').read_text())
    assert meta['mode']=='simulation'
    cap=cv2.VideoCapture(str(root/'python_tools/dataset/raw_videos'/meta['source_video']))
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,meta['frame_id']-1)
    ok,frame=cap.read();assert ok
    received=cv2.imread(str(root/'artifacts/preview_alignment.jpg'))
    expected=cv2.resize(frame,(received.shape[1],received.shape[0]),interpolation=cv2.INTER_AREA)
    ok,encoded=cv2.imencode('.jpg',expected,[cv2.IMWRITE_JPEG_QUALITY,80]);assert ok
    expected=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    mae=float(np.abs(received.astype(float)-expected.astype(float)).mean())
    report={'frame_id':meta['frame_id'],'source_video':meta['source_video'],'jpeg_mean_absolute_pixel_error':mae,'passed':mae<2}
    (root/'artifacts/preview_alignment_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));assert report['passed'], 'Preview does not match declared source frame'
