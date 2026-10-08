"""Import original camera recordings as unreviewed consecutive-frame bursts.

Never transcodes or edits originals; never assigns ground truth or dataset splits.
Timestamp values come from the decoder, not an assumed constant 60 FPS.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def import_recordings(folder, output, session, bursts=6, resume=False, *,
                      start_frame=None, capture_sequence=1, note=''):
    import cv2
    import numpy as np
    if not session.strip() or not 1 <= bursts <= 100:
        raise ValueError('Provide a session ID and 1–100 bursts per recording')
    if not isinstance(capture_sequence, int) or capture_sequence < 1:
        raise ValueError('Capture sequence must be a positive integer')
    if start_frame is not None and (not isinstance(start_frame, int) or start_frame < 1 or not folder.is_file()):
        raise ValueError('Targeted sampling requires one recording and a positive 1-based start frame')
    files = [folder] if folder.is_file() and folder.suffix.lower() in ('.mov', '.mp4', '.m4v') else (
        sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.mov', '.mp4', '.m4v')) if folder.is_dir() else [])
    if not files:
        raise ValueError('No camera recordings found')
    output.mkdir(parents=True, exist_ok=True)
    report = {'session_id': session, 'split': None, 'contains_ground_truth': False,
              'scope': 'Full sequential decoder count/timestamps plus sampled image import; not accuracy evaluation or independent codec validation',
              'recordings': []}
    for video in files:
        before = video.stat()
        sha = digest(video)
        clip = output / ('phone_' + sha[:16] + '_' + str(capture_sequence))
        if clip.exists():
            if not resume:
                raise FileExistsError(f'Capture already exists; no files overwritten: {clip}')
            entry = json.loads((clip/'capture.json').read_text())
            rows = [json.loads(line) for line in (clip/'frames.jsonl').read_text().splitlines()]
            if entry.get('original_sha256') != sha or entry.get('session_id') != session or len(rows) != entry.get('sampled_frames'):
                raise ValueError(f'Existing capture provenance differs: {clip}')
            if start_frame is not None and entry.get('selection') != {'method': 'targeted_failure', 'start_source_frame': start_frame, 'requested_bursts': bursts}:
                raise ValueError(f'Existing targeted sampling differs: {clip}')
            if start_frame is None and entry.get('selection', {}).get('method') == 'targeted_failure':
                raise ValueError(f'Existing capture is targeted, not uniform: {clip}')
            if any(not (clip/row['file']).is_file() for row in rows):
                raise ValueError(f'Existing capture images missing: {clip}')
            report['recordings'].append({'clip': clip.name, **entry})
            print(f'{video.name}: verified existing import; annotations preserved', flush=True)
            continue
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                raise RuntimeError(f'Cannot decode {video}')
            reported_count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
            if reported_count < 5 or not np.isfinite(fps) or fps <= 0:
                raise RuntimeError(f'Invalid frame count/rate in {video}')
            codec = ''.join(chr((int(capture.get(cv2.CAP_PROP_FOURCC)) >> (8 * i)) & 255) for i in range(4))
            # MOV frame counts can be estimated from duration/rate. Count actual
            # decoded frames and audit every source timestamp before sampling.
            source_times = []
            while capture.grab():
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if not np.isfinite(timestamp) or timestamp < 0 or (source_times and timestamp <= source_times[-1]):
                    raise RuntimeError(f'Unavailable/nonmonotonic source timestamps in {video}')
                source_times.append(timestamp)
                if len(source_times) % 10000 == 0:
                    print(f'{video.name}: inspected {len(source_times)} source frames', flush=True)
            count = len(source_times)
            if count < 5:
                raise RuntimeError(f'Insufficient decodable frames in {video}')
            source_intervals = np.diff(source_times)
            if start_frame is not None:
                if start_frame + 5 * bursts - 1 > count:
                    raise ValueError('Requested consecutive bursts exceed the decoded source frames')
                starts = list(range(start_frame - 1, start_frame - 1 + 5 * bursts, 5))
            else:
                starts = []
                for value in sorted(set(int(value) for value in np.linspace(0, count - 5, bursts))):
                    if not starts or value >= starts[-1] + 5:
                        starts.append(value)
            rows, intervals, annotations = [], [], {}
            pending = output / ('.pending_' + clip.name)
            pending.mkdir()
            size = None
            # Random frame seeking is not reliable for variable-timing MOVs.
            # Decode the second pass in source order and retrieve only samples.
            capture.release()
            capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
            cursor = 0
            for burst_id, start in enumerate(starts, 1):
                while cursor < start:
                    if not capture.grab():
                        raise RuntimeError(f'Sequential sample decode ended early in {video}')
                    cursor += 1
                previous = None
                for burst_index in range(5):
                    if not capture.grab():
                        raise RuntimeError(f'Incomplete burst in {video} at frame {start + burst_index + 1}')
                    cursor += 1
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise RuntimeError(f'Incomplete burst in {video} at frame {start + burst_index + 1}')
                    frame_id = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                    if frame_id != start + burst_index + 1:
                        raise RuntimeError(f'Unexpected decoder frame position in {video}')
                    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    if not np.isfinite(timestamp) or timestamp < 0 or (previous is not None and timestamp <= previous):
                        raise RuntimeError(f'Unavailable/nonmonotonic frame timestamps in {video}')
                    if previous is not None:
                        intervals.append(timestamp - previous)
                    previous = timestamp
                    size = (frame.shape[1], frame.shape[0])
                    name = f'frame_{len(rows) + 1}.jpg'
                    if abs(timestamp - source_times[frame_id - 1]) > .5:
                        raise RuntimeError(f'Seek timestamp differs from sequential decode in {video}')
                    if not cv2.imwrite(str(pending / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f'Cannot write {pending / name}')
                    rows.append({'file': name, 'source_frame': frame_id, 'elapsed_ms': timestamp,
                                 'burst_id': burst_id, 'burst_index': burst_index, 'burst_length': 5})
                    if start_frame is not None:
                        rows[-1]['review_metadata'] = {'reasons': ['USER_REPORTED_DETECTION_FAILURE'], 'capture_note': note}
                    annotations[name] = {'reviewed': False, 'ball': [], 'person': [], 'table': []}
            after = video.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise RuntimeError(f'Source changed during import: {video}')
            entry = {'source': 'recorded_camera', 'capture_origin': 'iphone', 'session_id': session,
                     'original_path': str(video.resolve()), 'original_sha256': sha,
                     'original_bytes': before.st_size, 'source_frames': count, 'reported_source_frames': reported_count,
                     'frame_count_discrepancy': reported_count - count, 'source_fps_average': fps,
                     'full_source_decode_counted': True,
                     'source_interval_ms_p50': float(np.median(source_intervals)),
                     'source_interval_ms_min': float(source_intervals.min()),
                     'source_interval_ms_max': float(source_intervals.max()),
                     'codec_fourcc': codec, 'width': size[0], 'height': size[1], 'burst_frames': 5,
                     'timestamp_basis': 'opencv_ffmpeg_source_pts_ms', 'opencv_version': cv2.__version__,
                     'sampled_interval_ms_min': min(intervals), 'sampled_interval_ms_max': max(intervals),
                     'sampled_frames': len(rows), 'contains_ground_truth': False, 'split': None,
                     'sampling': (f'{len(starts)} adjacent five-consecutive-frame bursts from source frame {start_frame}'
                                  if start_frame is not None else f'{len(starts)} evenly spaced five-consecutive-frame bursts'),
                     'selection': {'method': 'targeted_failure' if start_frame is not None else 'uniform',
                                   'start_source_frame': start_frame, 'requested_bursts': bursts},
                     'capture_note': note,
                     'rotation_handling': 'decoder default; inspect imported orientation',
                     'hdr_metadata': None, 'hdr_metadata_reason': 'Not exposed by this OpenCV decoder'}
            (pending / 'capture.json').write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
            (pending / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            (pending / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n', encoding='utf-8')
            pending.rename(clip)
            report['recordings'].append({'clip': clip.name, **entry})
            print(f'{video.name}: {size[0]}x{size[1]}, {fps:.3f} average FPS, {count} frames; imported {len(rows)} unreviewed frames', flush=True)
        finally:
            capture.release()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--session', required=True)
    parser.add_argument('--bursts', type=int, default=6)
    parser.add_argument('--resume', action='store_true', help='Verify and reuse completed imports; never overwrite annotations')
    parser.add_argument('--start-frame', type=int, help='1-based start of adjacent five-frame bursts; input must be one recording')
    parser.add_argument('--capture-sequence', type=int, default=1, help='Distinct capture ID suffix; existing captures are never overwritten')
    parser.add_argument('--note', default='', help='Human-readable review context; never a ground-truth label')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = import_recordings(args.folder, args.output, args.session, args.bursts, args.resume,
                               start_frame=args.start_frame, capture_sequence=args.capture_sequence, note=args.note)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
