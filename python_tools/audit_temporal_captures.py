"""Audit reviewed five-frame live bursts without converting suggestions to labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from prepare_live_training import validate_annotation


def audit(root, split_file):
    splits = json.loads(split_file.read_text())
    if set(splits) != {'train', 'val', 'test'}:
        raise ValueError('Separate train, val and test recording lists are required')
    seen_clips, image_splits, accepted, excluded = set(), {}, [], []
    root = root.resolve()
    for split, clips in splits.items():
        if not isinstance(clips, list) or not clips:
            raise ValueError('Every split needs at least one complete recording')
        for clip in clips:
            directory = (root/clip).resolve()
            if directory.parent != root or clip in seen_clips:
                raise ValueError('A recording must remain inside the capture root and in exactly one split')
            seen_clips.add(clip)
            annotations = json.loads((directory/'annotations.json').read_text())
            groups = defaultdict(list)
            manifest = (directory/'frames.jsonl').read_text()
            for line in manifest.split('\n')[:-1]:
                if not line:
                    continue
                row = json.loads(line)
                if row.get('burst_length') == 5:
                    groups[row['burst_id']].append(row)
            for bid, frames in groups.items():
                frames.sort(key=lambda f:f['burst_index'])
                reason = None
                if len(frames) != 5 or [f['burst_index'] for f in frames] != list(range(5)):
                    reason = 'INCOMPLETE_BURST'
                elif [f['source_frame'] for f in frames] != list(range(frames[0]['source_frame'], frames[0]['source_frame']+5)):
                    reason = 'SOURCE_FRAME_GAP'
                elif any(b['elapsed_ms'] <= a['elapsed_ms'] for a,b in zip(frames, frames[1:])):
                    reason = 'INVALID_CAPTURE_TIMES'
                hashes, labels = [], []
                for frame in frames:
                    image = (directory/frame['file']).resolve()
                    if image.parent != directory:
                        raise ValueError('Frame path escapes its recording')
                    digest = hashlib.sha256(image.read_bytes()).hexdigest()
                    if digest in image_splits and image_splits[digest] != split:
                        raise ValueError('Duplicate image across recording splits')
                    image_splits[digest] = split
                    hashes.append(digest)
                    annotation = validate_annotation(annotations.get(frame['file'], {}))
                    if annotation is None:
                        reason = reason or 'UNREVIEWED_FRAME'
                    labels.append(annotation)
                record = dict(split=split, clip=clip, burst_id=bid, files=[f['file'] for f in frames],
                              frames=[f['source_frame'] for f in frames], sha256=hashes)
                if reason:
                    excluded.append(dict(**record, reason=reason))
                else:
                    accepted.append(dict(**record, labels=labels, elapsed_ms=[f['elapsed_ms'] for f in frames]))
    counts = {split:sum(row['split']==split for row in accepted) for split in splits}
    return dict(schema_version=1, split_manifest_sha256=hashlib.sha256(split_file.read_bytes()).hexdigest(),
                accepted_counts=counts, ready_for_training=all(counts.values()), accepted=accepted, excluded=excluded,
                scope='Audited reviewed live bursts; this is not an accuracy or sample-size acceptance result.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures',type=Path);parser.add_argument('splits',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=audit(args.captures,args.splits)
    with args.output.open('x') as output:
        json.dump(result,output,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('accepted','excluded')},indent=2))
