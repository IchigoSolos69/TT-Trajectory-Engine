"""Compare deployed C++ TensorRT top-1 output to PyTorch on identical decoded frames."""
import argparse
import csv
import json
import math
from pathlib import Path
from improve_detector import OUTPUT  # Adds isolated dependencies before ML imports.
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def verify(args):
    with args.predictions.open(newline='') as source:
        rows = [r for r in csv.DictReader(source) if float(r['confidence']) > .1]
    if len(rows) < 32:
        raise ValueError('Need at least 32 confident predictions for parity verification')
    rows = [rows[i] for i in np.linspace(0, len(rows)-1, 32, dtype=int)]
    model = YOLO(str(args.weights)).model.to('cuda').float().eval()
    model.fuse()
    cap = cv2.VideoCapture(str(args.video))
    checks = []
    with torch.inference_mode():
        for row in rows:
            fid = int(row['frame'])
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1):
                raise RuntimeError(f'Cannot seek frame {fid}')
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode frame {fid}')
            h, w = frame.shape[:2]
            scale = min(640/w, 640/h)
            uw, uh = round(w*scale), round(h*scale)
            left, top = (640-uw)//2, (640-uh)//2
            resized = cv2.resize(frame, (uw, uh))
            padded = cv2.copyMakeBorder(resized, top, 640-uh-top, left, 640-uw-left,
                                        cv2.BORDER_CONSTANT, value=(114,114,114))
            array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2,0,1), dtype=np.float32)/255
            pred = model(torch.from_numpy(array).unsqueeze(0).to('cuda'))
            if isinstance(pred, tuple):
                pred = pred[0]
            raw = pred[0].float().cpu().numpy()
            if raw.shape != (5, 8400):
                raise ValueError(f'Unexpected PyTorch prediction shape: {raw.shape}')
            idx = int(raw[4].argmax())
            x, y = (float(raw[0,idx])-left)/scale, (float(raw[1,idx])-top)/scale
            checks.append({'frame': fid,
                'center_delta_px': math.hypot(x-float(row['x']), y-float(row['y'])),
                'confidence_delta': abs(float(raw[4,idx])-float(row['confidence']))})
    cap.release()
    report = {'sample_count': len(checks), 'sampling': '32 evenly spaced C++ predictions above 0.1',
              'max_center_delta_px': max(r['center_delta_px'] for r in checks),
              'max_confidence_delta': max(r['confidence_delta'] for r in checks), 'frames': checks}
    report['passed'] = report['max_center_delta_px'] <= 3 and report['max_confidence_delta'] <= .02
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'frames'}, indent=2))
    if not report['passed']:
        raise RuntimeError('TensorRT/PyTorch parity exceeded 3 pixels or 0.02 confidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('weights', 'predictions', 'video', 'output'):
        parser.add_argument(name, type=Path)
    verify(parser.parse_args())
