"""Prepare reviewed live examples, split by recording, and retain match examples.

init creates an unreviewed annotation template. prepare accepts explicit clip
splits and only human-reviewed labels. Missing labels never become negatives.
Boxes are [left, top, width, height], normalized to the clean source image.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('ball', 'person', 'table')


def validate_annotation(annotation):
    if annotation.get('reviewed') is not True:
        return None
    visibility = annotation.get('geometry', {}).get('visibility')
    # A hidden ball is not a verified absent-ball negative, even if someone
    # saved the scene boxes as reviewed. Keep such examples out of this dataset.
    if visibility == 'occluded' or (visibility == 'partial' and not annotation.get('ball')):
        return None
    result = {}
    for name in CLASSES:
        boxes = annotation.get(name)
        if not isinstance(boxes, list):
            raise ValueError(f'Reviewed annotations must explicitly label {name}; [] means confirmed absent')
        result[name] = []
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f'Invalid {name} box')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                raise ValueError(f'Non-finite {name} box')
            x, y, w, h = box
            if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x+w <= 1.000001 and y+h <= 1.000001):
                raise ValueError(f'Out-of-bounds {name} box: {box}')
            result[name].append(box)
    return result


def initialize(clip):
    target = clip / 'annotations.json'
    if target.exists():
        raise FileExistsError(f'Annotations already exist: {target}')
    frames = [json.loads(line) for line in (clip/'frames.jsonl').read_text().splitlines()]
    records = {row['file']: {'reviewed': False, **{name: [] for name in CLASSES}} for row in frames}
    target.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Created {len(records)} UNREVIEWED entries: {target}')


def check_capture_session(manifest, split, sessions):
    if manifest.get('source') not in ('live', 'recorded_camera'):
        raise ValueError('Only real camera captures/recordings belong in this dataset')
    session = manifest.get('session_id')
    if manifest.get('source') == 'recorded_camera' and (not isinstance(session, str) or not session.strip()):
        raise ValueError('Imported camera recordings require a session ID')
    if session is not None:
        if not isinstance(session, str) or not session.strip():
            raise ValueError('Invalid capture session ID')
        if session in sessions and sessions[session] != split:
            raise ValueError('Capture session overlaps dataset splits')
        sessions[session] = split


def load_reviewed(captures, splits):
    if set(splits) != {'train', 'val', 'test'}:
        raise ValueError('Provide train, val and test recording lists')
    seen_clips, hashes, rows, sessions = set(), {}, {key: [] for key in splits}, {}
    skipped = 0
    for split, clips in splits.items():
        if not isinstance(clips, list) or not clips:
            raise ValueError(f'At least one separate recording is required for {split}')
        for clip in clips:
            if not isinstance(clip, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', clip) or clip in seen_clips:
                raise ValueError(f'Invalid or overlapping recording: {clip}')
            seen_clips.add(clip)
            directory = (captures/clip).resolve()
            if not directory.is_relative_to(captures.resolve()):
                raise ValueError('Recording path escapes capture directory')
            check_capture_session(json.loads((directory/'capture.json').read_text()), split, sessions)
            annotations = json.loads((directory/'annotations.json').read_text())
            frames = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
            for row in frames:
                filename = row['file']
                if not re.fullmatch(r'frame_[0-9]+\.jpg', filename):
                    raise ValueError('Invalid capture filename')
                labels = validate_annotation(annotations.get(filename, {}))
                if labels is None:
                    skipped += 1
                    continue
                image = (directory/filename).resolve()
                if not image.is_relative_to(directory):
                    raise ValueError('Image path escapes recording directory')
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in hashes:
                    if hashes[digest] != split:
                        raise ValueError('Identical image found across dataset splits')
                    continue
                hashes[digest] = split
                rows[split].append((image, labels, digest, clip))
        if not rows[split]:
            raise ValueError(f'No reviewed examples in {split}; unknown frames are not negatives')
    return rows, skipped


def prepare(captures, split_file, output):
    splits = json.loads(split_file.read_text())
    rows, skipped = load_reviewed(captures, splits)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    audit = {'splits': splits, 'unreviewed_excluded': skipped, 'contains_test_in_training': False,
             'reviewed_counts': {k:len(v) for k,v in rows.items()}, 'frames': []}
    for kind, classes in [('ball', ('ball',)), ('scene', ('person', 'table'))]:
        lists = {}
        for split in rows:
            images, labels_dir = output/kind/'images'/split, output/kind/'labels'/split
            images.mkdir(parents=True); labels_dir.mkdir(parents=True)
            paths = []
            for image, labels, digest, clip in rows[split]:
                destination = images/f'{clip}_{image.name}'
                shutil.copy2(image, destination)
                lines = []
                for class_id, name in enumerate(classes):
                    for x,y,w,h in labels[name]:
                        lines.append(f'{class_id} {x+w/2:.8f} {y+h/2:.8f} {w:.8f} {h:.8f}')
                (labels_dir/f'{destination.stem}.txt').write_text('\n'.join(lines), encoding='utf-8')
                paths.append(destination.resolve().as_posix())
                if kind == 'ball':
                    audit['frames'].append({'split':split, 'clip':clip, 'file':image.name, 'sha256':digest,
                                            'objects':{k:len(v) for k,v in labels.items()}})
            lists[split] = paths
            (output/kind/f'live_{split}.txt').write_text('\n'.join(paths)+'\n', encoding='utf-8')
        # Preserve original chronological match splits; never import test_* footage.
        if kind == 'ball':
            original = {s:(ROOT/f'artifacts/multigame/{s}.txt').read_text().splitlines() for s in ('train','val')}
        else:
            original = {s:[p.resolve().as_posix() for p in sorted((ROOT/f'artifacts/scene_detector/dataset/images/{s}').glob('*.jpg'))] for s in ('train','val')}
        for split in ('train','val'):
            if not original[split] or any(not Path(p).is_file() or 'test_' in Path(p).name for p in original[split]):
                raise ValueError('Original match data missing or test sequence leakage detected')
            selected = original[split]
            if split == 'train':
                # A deterministic evenly spaced match subset keeps a 3:1 mix at most.
                count = min(len(selected), 3*len(lists[split]))
                selected = [selected[i*len(selected)//count] for i in range(count)]
            merged = selected + lists[split]
            (output/kind/f'{split}.txt').write_text('\n'.join(merged)+'\n', encoding='utf-8')
        names='\n'.join(f'  {i}: {name}' for i,name in enumerate(classes))
        config = f'train: {(output/kind/"train.txt").resolve().as_posix()}\nval: {(output/kind/"val.txt").resolve().as_posix()}\nnames:\n{names}\n'
        (output/kind/'data.yaml').write_text(config, encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='frames'},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    init=commands.add_parser('init');init.add_argument('clip',type=Path)
    prep=commands.add_parser('prepare');prep.add_argument('splits',type=Path);prep.add_argument('output',type=Path)
    prep.add_argument('--captures',type=Path,default=ROOT/'training_data/live')
    args=parser.parse_args()
    if args.command=='init':initialize(args.clip)
    else:prepare(args.captures,args.splits,args.output)
