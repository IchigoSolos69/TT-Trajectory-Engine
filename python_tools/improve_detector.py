"""Auditable candidate training; never overwrites the deployed detector.

Missing annotations are UNKNOWN, never inferred negatives. test_* sequences are
excluded from training. test_1 has already been used for development in this repo.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'python_tools/dataset'
BASELINE = ROOT / 'runs/detect/python_tools/runs/detect/yolo11_ball_game1/weights/best.pt'
OUTPUT = ROOT / 'artifacts/detector_improvement'
# Optional complete CUDA torchvision wheel, isolated from system packages.
LOCAL_DEPS = ROOT / 'artifacts/python_deps'
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))


def annotation_status(value):
    if not isinstance(value, dict):
        return 'unknown'
    x, y = value.get('x'), value.get('y')
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return 'unknown'
    import math
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('Non-finite annotation')
    if x >= 0 and y >= 0:
        return 'positive'
    # Only the explicit paired absent sentinel is a verified negative.
    if x == -1 and y == -1:
        return 'negative'
    # Partially missing coordinates do not establish visible-ball absence.
    return 'unknown'


def prepare():
    import cv2
    OUTPUT.mkdir(parents=True, exist_ok=True)
    markup = json.loads((DATA / 'raw_markup/game_1/ball_markup.json').read_text())
    cap = cv2.VideoCapture(str(DATA / 'raw_videos/game_1.mp4'))
    if not cap.isOpened():
        raise RuntimeError('Cannot open game_1 dataset video')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    cutoff, val_start = int(total * .8), int(total * .8) + int(fps)
    report = {'train_cutoff': cutoff, 'val_start': val_start, 'splits': {}}
    dataset = DATA / 'yolo_ball_game1'
    clean = OUTPUT / 'clean_dataset'
    # Reuse source images without rewriting or deleting the existing dataset.
    for split in ('train', 'val'):
        paths, rows = [], []
        counts = {'positive': 0, 'negative': 0, 'unknown_excluded': 0, 'labels_corrected': 0}
        (clean / 'images' / split).mkdir(parents=True, exist_ok=True)
        (clean / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for path in sorted((dataset / 'images' / split).glob('*.jpg')):
            fid = int(path.stem.split('_f')[1])
            expected = fid <= cutoff if split == 'train' else fid >= val_start
            if not expected:
                raise ValueError(f'Chronological split violation: {path}')
            value = markup.get(str(fid))
            status = annotation_status(value)
            if status == 'unknown':
                counts['unknown_excluded'] += 1
                continue
            label = dataset / 'labels' / split / f'{path.stem}.txt'
            expected_label = ''
            if status == 'positive':
                x, y = value['x'], value['y']
                if x >= width or y >= height:
                    counts['unknown_excluded'] += 1
                    continue
                x1, x2 = max(0, x - 15), min(width, x + 15)
                y1, y2 = max(0, y - 15), min(height, y + 15)
                expected_label = f'0 {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}\n'
            counts['labels_corrected'] += label.read_text().strip() != expected_label.strip()
            clean_image = clean / 'images' / split / path.name
            if not clean_image.exists():
                import os
                os.link(path, clean_image)
            (clean / 'labels' / split / label.name).write_text(expected_label)
            paths.append(clean_image.resolve().as_posix())
            counts[status] += 1
            rows.append({'frame': fid, 'image': path.resolve().as_posix(), 'status': status,
                         'x': value['x'], 'y': value['y']})
        if not paths:
            raise ValueError(f'Empty {split} split')
        (OUTPUT / f'{split}.txt').write_text('\n'.join(paths) + '\n')
        (OUTPUT / f'{split}_frames.json').write_text(json.dumps(rows, indent=2))
        report['splits'][split] = counts
    yaml = f"path: {clean.as_posix()}\ntrain: {(OUTPUT / 'train.txt').as_posix()}\nval: {(OUTPUT / 'val.txt').as_posix()}\nnames:\n  0: ball\n"
    (OUTPUT / 'data.yaml').write_text(yaml)
    (OUTPUT / 'data_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def train(args):
    import torch
    import torchvision
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA training unavailable; refusing silent CPU fallback')
    kept = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device='cuda'),
                               torch.tensor([.9, .8], device='cuda'), .5)
    if kept.tolist() != [0]:
        raise RuntimeError('CUDA torchvision NMS verification failed')
    print(f'CUDA NMS verified: torchvision {torchvision.__version__}', flush=True)
    prepare()
    model = YOLO(str(BASELINE))
    if model.names != {0: 'ball'}:
        raise ValueError(f'Expected a single ball class, got {model.names}')
    print(f'CUDA device: {torch.cuda.get_device_name(0)}', flush=True)
    model.train(data=str(OUTPUT / 'data.yaml'), epochs=args.epochs, imgsz=640,
                batch=16, device=0, workers=4, val=True, patience=8,
                optimizer='AdamW', lr0=0.0003, lrf=0.1, cos_lr=True,
                mosaic=0.0, mixup=0.0, scale=0.15, translate=0.05,
                fliplr=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
                seed=42, deterministic=True, amp=True, plots=True,
                project=str(OUTPUT), name='candidate', exist_ok=False)


def export(args):
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    if model.names != {0: 'ball'}:
        raise ValueError(f'Expected ball model: {model.names}')
    result = model.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                          simplify=False, opset=17)
    print(f'Exported {result}', flush=True)


def audit():
    import torch
    import cv2
    import ultralytics
    report = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
              'opencv': cv2.__version__, 'cuda': torch.cuda.is_available(), 'videos': {}}
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name(0)
    for path in sorted((DATA / 'raw_videos').glob('*.mp4')):
        cap = cv2.VideoCapture(str(path))
        markup = json.loads((DATA / 'raw_markup' / path.stem / 'ball_markup.json').read_text())
        statuses = [annotation_status(v) for v in markup.values()]
        report['videos'][path.stem] = {'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS), 'positive': statuses.count('positive'),
            'explicit_negative': statuses.count('negative'),
            'model_sha256': hashlib.sha256(BASELINE.read_bytes()).hexdigest()}
        cap.release()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'environment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'prepare', 'train', 'export'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if args.action == 'train':
        train(args)
    elif args.action == 'export':
        if not args.weights or not args.weights.is_file():
            parser.error('--weights must name an existing candidate checkpoint')
        export(args)
    elif args.action == 'prepare':
        prepare()
    else:
        audit()
