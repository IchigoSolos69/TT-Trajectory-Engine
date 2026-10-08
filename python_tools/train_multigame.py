"""Add a second training match while keeping every test sequence held out."""
from pathlib import Path
import sys
import json
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/python_deps'))
from improve_detector import annotation_status, BASELINE
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    output = ROOT / 'artifacts/multigame'
    data = output / 'dataset'
    lists = {'train': [], 'val': []}
    for split in lists:
        for part in ('images', 'labels'):
            (data / part / split).mkdir(parents=True, exist_ok=True)
        for image in (ROOT / f'artifacts/detector_improvement/clean_dataset/images/{split}').glob('*.jpg'):
            lists[split].append(image.resolve().as_posix())
    labels = json.loads((ROOT / 'python_tools/dataset/raw_markup/game_2/ball_markup.json').read_text())
    video = ROOT / 'python_tools/dataset/raw_videos/game_2.mp4'
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Video not readable: {video}')
    cutoff = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * .8)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    selected = {int(k): v for k, v in sorted(labels.items(), key=lambda item:int(item[0]))[::3]
                if annotation_status(v) != 'unknown' and (int(k) <= cutoff or int(k) >= cutoff+120)}
    counts = {'train': 0, 'val': 0, 'negative': 0}
    for fid in range(1, max(selected)+1):
        if not cap.grab():
            raise RuntimeError(f'Decode ended early: {fid}')
        if fid not in selected:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            raise RuntimeError(f'Cannot retrieve {fid}')
        split = 'train' if fid <= cutoff else 'val'
        image = data / 'images' / split / f'game_2_{fid}.jpg'
        if not image.exists() and not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f'Cannot write {image}')
        value = selected[fid]
        text = ''
        if annotation_status(value) == 'positive':
            x, y = value['x'], value['y']
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f'Out of bounds label: {fid}: {value}')
            x1,x2,y1,y2=max(0,x-15),min(w,x+15),max(0,y-15),min(h,y+15)
            text=f'0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n'
        else:
            counts['negative'] += 1
        (data / 'labels' / split / f'game_2_{fid}.txt').write_text(text)
        lists[split].append(image.resolve().as_posix())
        counts[split] += 1
    cap.release()
    for split, paths in lists.items():
        (output / f'{split}.txt').write_text('\n'.join(paths)+'\n')
    yaml=output/'data.yaml'
    yaml.write_text(f'train: {(output/"train.txt").as_posix()}\nval: {(output/"val.txt").as_posix()}\nnames:\n  0: ball\n')
    (output/'data_audit.json').write_text(json.dumps({'game_2_added':counts,'total':{k:len(v) for k,v in lists.items()},'test_sequences_used':False,'game_2_cutoff':cutoff},indent=2))
    print(counts,flush=True)
    YOLO(str(BASELINE)).train(data=str(yaml), epochs=16, imgsz=640, batch=16,
        device=0, workers=1, optimizer='AdamW', lr0=.00025, lrf=.1,
        cos_lr=True, mosaic=.15, close_mosaic=4, scale=.2, translate=.08,
        hsv_s=.5, hsv_v=.5, seed=42, deterministic=True, patience=6,
        project=str(output),name='candidate')
