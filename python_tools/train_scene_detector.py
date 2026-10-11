"""Train a separate person/table detector from real OpenTTGames masks.

Masks are BGR: green = people (including officials), red = table. People are
not assigned player identities. Chronological split with one-second exclusion.
No test_* footage is used. This model is not a camera calibration oracle.
"""
from pathlib import Path
if __package__:
    from .public_dataset_source import resolve_public_dataset_video
else:
    from public_dataset_source import resolve_public_dataset_video
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/python_deps'))
import cv2
from ultralytics import YOLO


def prepare():
    output = ROOT / 'artifacts/scene_detector'
    dataset = output / 'dataset'
    counts = {'train': 0, 'val': 0}
    for split in counts:
        for part in ('images', 'labels'):
            (dataset / part / split).mkdir(parents=True, exist_ok=True)
    for game in ('game_1', 'game_2'):
        masks = ROOT / f'python_tools/dataset/raw_markup/{game}/segmentation_masks'
        paths = sorted(masks.glob('*.png'), key=lambda p: int(p.stem))[::3]
        if not paths:
            continue
        cap = cv2.VideoCapture(str(resolve_public_dataset_video(game)))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot decode {game}')
        cutoff = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * .8)
        wanted = {int(p.stem): p for p in paths if int(p.stem) <= cutoff or int(p.stem) >= cutoff + 120}
        for fid in range(1, max(wanted) + 1):
            if not cap.grab():
                raise RuntimeError(f'Unexpected decode end: {game}:{fid}')
            if fid not in wanted:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f'Cannot retrieve {game}:{fid}')
            mask = cv2.imread(str(wanted[fid]))
            h, w = mask.shape[:2]
            labels = []
            for class_id, channel in ((0, 1), (1, 2)):
                binary = (mask[:, :, channel] > 75).astype('uint8')
                count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
                for x, y, bw, bh, area in stats[1:]:
                    if area < (30 if class_id == 0 else 200):
                        continue
                    labels.append(f'{class_id} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}')
            split = 'train' if fid <= cutoff else 'val'
            name = f'{game}_{fid}'
            image = dataset / 'images' / split / f'{name}.jpg'
            if not image.exists() and not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f'Cannot write {image}')
            (dataset / 'labels' / split / f'{name}.txt').write_text('\n'.join(labels) + '\n')
            counts[split] += 1
        cap.release()
        print(game, counts, flush=True)
    yaml = output / 'data.yaml'
    yaml.write_text(f'path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: table\n')
    (output / 'data_audit.json').write_text(json.dumps({'counts': counts, 'mask_channels_bgr': {'person': 1, 'table': 2}, 'test_sequences_used': False}, indent=2))
    return yaml


if __name__ == '__main__':
    data = prepare()
    YOLO(str(ROOT / 'yolo11n.pt')).train(data=str(data), epochs=15, imgsz=640,
        batch=16, device=0, workers=1, optimizer='AdamW', lr0=.001,
        mosaic=.3, close_mosaic=3, scale=.35, translate=.1, hsv_v=.4,
        seed=42, deterministic=True, patience=6,
        project=str(ROOT / 'artifacts/scene_detector'), name='candidate')
