"""Add causal detector-conditioned TRAINING crops; never import validation cases.

Request coordinates are exported by the C++ policy. Labels are only used here,
after dispatch, to supervise heatmap/presence. Unannotated frames are excluded.
"""
import argparse
import csv
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from temporal_recovery import crop, status, DATA, SIDE, ROOT


def prepare(base, output, requests):
    if output.exists():
        raise FileExistsError('Use a fresh candidate directory')
    output.mkdir(parents=True)
    additions, targets, origins = [], [], []
    audit = json.loads((base/'dataset.json').read_text())
    audit['causal_hard_cases'] = []
    audit['detector_engine_sha256'] = hashlib.sha256((ROOT/'models/yolov11_ttgames.engine').read_bytes()).hexdigest()
    for game, source in requests:
        if game not in ('game_1', 'game_2'):
            raise ValueError('Only the declared training games are permitted')
        labels_path = DATA/'raw_markup'/game/'ball_markup.json'
        labels = json.loads(labels_path.read_text())
        rows = list(csv.DictReader(source.open(newline='')))
        if not rows or min(int(r['frame']) for r in rows) < 1 or max(int(r['frame']) for r in rows) > 20000:
            raise ValueError('Hard-case import must stay inside frames 1..20000, the original training range')
        ids = [int(r['frame']) for r in rows]
        if ids != list(range(ids[0], ids[-1]+1)) or any(not 0 <= int(r['missed_frames']) <= 4 for r in rows):
            raise ValueError('Native request export is not a consecutive bounded replay')
        selected = {int(r['frame']): r for r in rows if int(r['missed_frames']) > 0 and
                    status(labels.get(r['frame'])) != 'unknown'}
        if not selected:
            raise ValueError('No labeled detector-conditioned cases')
        needed = {fid-d for fid in selected for d in range(5)}
        cap = cv2.VideoCapture(str(DATA/'raw_videos'/f'{game}.mp4'))
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, min(needed)-1):
            raise RuntimeError('Cannot decode training video')
        frames = deque(maxlen=5)
        positive = negative = 0
        for fid in range(min(needed), max(selected)+1):
            if not cap.grab():
                raise RuntimeError('Training video ended early')
            if fid not in needed:
                frames.clear()
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError('Cannot retrieve training frame')
            frames.append((fid, frame))
            if fid not in selected:
                continue
            if len(frames) != 5 or frames[0][0] != fid-4:
                raise RuntimeError('Nonconsecutive temporal crop')
            row, label = selected[fid], labels[str(fid)]
            left, top = int(row['roi_x']), int(row['roi_y'])
            x, y = (label['x']-left)/SIDE, (label['y']-top)/SIDE
            present = status(label) == 'positive' and 0 <= x < 1 and 0 <= y < 1
            additions.append(np.stack([crop(frame, left, top) for _, frame in frames]))
            targets.append([x if present else 0, y if present else 0, float(present)])
            origins.append(dict(game=game, frame=fid, left=left, top=top,
                                label='positive' if present else 'crop_negative',
                                roi_source='CAUSAL_NATIVE_DETECTOR_REQUEST'))
            positive += int(present); negative += int(not present)
        cap.release()
        audit['causal_hard_cases'].append(dict(game=game, positive=positive, negative=negative,
            request_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            markup_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest()))
    original = np.load(base/'train.npz')
    images = np.concatenate((original['images'], np.stack(additions)))
    labels = np.concatenate((original['targets'], np.asarray(targets, np.float32)))
    np.savez_compressed(output/'train.npz', images=images, targets=labels)
    provenance = json.loads((base/'train_provenance.json').read_text()) + origins
    (output/'train_provenance.json').write_text(json.dumps(provenance))
    for name in ('val.npz', 'val_provenance.json'):
        shutil.copyfile(base/name, output/name)
    audit['splits']['train'] = {'samples': len(images), 'positive': int(labels[:,2].sum())}
    audit['base_dataset_sha256'] = hashlib.sha256((base/'dataset.json').read_bytes()).hexdigest()
    audit['validation_unchanged'] = True
    (output/'dataset.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--game1-requests', type=Path, required=True)
    parser.add_argument('--game2-requests', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.output, [('game_1', args.game1_requests), ('game_2', args.game2_requests)])
