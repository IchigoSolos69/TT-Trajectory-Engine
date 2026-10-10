"""Import original camera recordings as unreviewed consecutive-frame bursts.

Never transcodes or edits originals; never assigns ground truth or dataset splits.
Timestamp values come from the decoder, not an assumed constant 60 FPS.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Review selection has duplicate JSON keys')
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=pairs)
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Review selection contains nonfinite values')
        if isinstance(item, dict):
            for nested in item.values(): finite(nested)
        elif isinstance(item, list):
            for nested in item: finite(nested)
    finite(value)
    if not isinstance(value, dict):
        raise ValueError('Review selection must be a JSON object')
    return value


def _review_selection(path, video, session, source_sha, decoded_count=None, decoded_fps=None):
    """Read-only audit of the exact selector and completed journal evidence."""
    path = Path(path)
    data = path.read_bytes()
    selection = _strict_json(data)
    identity = selection.get('identity')
    bursts = selection.get('bursts')
    if type(selection.get('schema_version')) is not int or selection['schema_version'] != 1:
        raise ValueError('Unsupported review selection schema')
    if selection.get('selection_method') != 'deterministic_greedy_temporal_spatial_grid_v1':
        raise ValueError('Unsupported review selection method')
    if selection.get('session_id') != session:
        raise ValueError('Review selection session differs; keep one session in one split')
    if (selection.get('burst_frames') != 5 or type(selection.get('burst_frames')) is not int
            or not isinstance(bursts, list) or not 1 <= len(bursts) <= 100):
        raise ValueError('Review selection requires 1–100 exact five-frame bursts')
    for key, expected in (('selected_bursts',len(bursts)),('selected_source_frames',len(bursts)*5)):
        if type(selection.get(key)) is not int or selection[key] != expected:
            raise ValueError('Review selection burst/frame count mismatch')
    if selection.get('nonoverlapping') is not True or selection.get('automatic_labels_created') is not False or selection.get('images_imported') is not False:
        raise ValueError('Review selection must remain unlabelled and nonoverlapping')
    truth = selection.get('truth')
    if (not isinstance(truth,dict) or truth.get('review_status') != 'UNREVIEWED'
            or truth.get('ground_truth_available') is not False
            or truth.get('declaration') != 'USER_DECLARED_NO_BALL'):
        raise ValueError('Review selection truth must remain unreviewed')
    if (not isinstance(identity,dict) or identity.get('source_sha256') != source_sha
            or identity.get('source_hash_verified') is not True or identity.get('model_hash_verified') is not True):
        raise ValueError('Review selection original source hash differs')
    hashes = ('source_sha256','ball_model_sha256','runtime_sha256','journal_sha256',
              'index_sha256','manifest_sha256','completion_sha256')
    for key in hashes:
        value = identity.get(key)
        if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
            raise ValueError('Review selection missing valid '+key)
    count = identity.get('source_frame_count')
    fps = identity.get('source_fps')
    if type(count) is not int or count < 5 or isinstance(fps,bool) or not isinstance(fps,(int,float)) or abs(fps) > 10000 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Review selection source count/rate invalid')
    if decoded_count is not None and count != decoded_count:
        raise ValueError('Review selection decoded source frame count differs')
    if decoded_fps is not None and abs(fps-decoded_fps) > .00001:
        raise ValueError('Review selection decoded source FPS differs')
    recorded_source = identity.get('source')
    if not isinstance(recorded_source,str) or Path(recorded_source).resolve() != video.resolve():
        raise ValueError('Review selection original source path differs')
    journal_name = identity.get('journal_directory')
    if not isinstance(journal_name,str) or not Path(journal_name).is_absolute():
        raise ValueError('Review selection completed journal path is invalid')
    journal = Path(journal_name)
    journal_files = {'manifest.json':'manifest_sha256','completion.json':'completion_sha256',
                     'frames.jsonl':'journal_sha256','frames.idx':'index_sha256'}
    contents = {}
    for name,key in journal_files.items():
        content = (journal/name).read_bytes()
        if hashlib.sha256(content).hexdigest() != identity[key]:
            raise ValueError('Review selection referenced journal evidence is stale: '+name)
        contents[name] = content
    manifest = _strict_json(contents['manifest.json'])
    completion = _strict_json(contents['completion.json'])
    model = manifest.get('ball_model')
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1
            or not isinstance(model,dict) or model.get('model_sha256') != identity['ball_model_sha256']
            or type(model.get('lookahead_frames')) is not int or model['lookahead_frames'] != 0
            or model.get('temporal_policy') != 'causal_channel_2'
            or manifest.get('source_sha256') != source_sha or manifest.get('source') != recorded_source
            or manifest.get('total_frames') != count or manifest.get('source_declared_frames') != count
            or manifest.get('source_fps') != fps or manifest.get('source_interval') != 'COMPLETE_RECORDING'
            or manifest.get('ball_model_sha256') != identity['ball_model_sha256']
            or manifest.get('runtime_sha256') != identity['runtime_sha256']
            or manifest.get('ball_model') != identity.get('model_provenance')
            or manifest.get('evaluation_policy') != 'CAUSAL_SOURCE_REPLAY_NOT_LIVE_LATENCY'):
        raise ValueError('Review selection completed source/model provenance mismatch')
    if (completion != {'status':'COMPLETE','analyzed_frames':count,'total_frames':count}
            or type(completion.get('analyzed_frames')) is not int or type(completion.get('total_frames')) is not int
            or len(contents['frames.idx']) != count*8):
        raise ValueError('Review selection referenced journal is incomplete')
    journal_rows = []
    index = [item[0] for item in struct.iter_unpack('<Q',contents['frames.idx'])]
    offset = 0
    for line in contents['frames.jsonl'].splitlines(keepends=True):
        frame = len(journal_rows)+1
        if frame > count or not line.endswith(b'\n') or index[frame-1] != offset:
            raise ValueError('Review selection journal index/coverage differs')
        row = _strict_json(line)
        timestamp = row.get('source_time')
        if (type(row.get('frame_id')) is not int or row['frame_id'] != frame
                or isinstance(timestamp,bool) or not isinstance(timestamp,(int,float))
                or abs(timestamp) > 1e12 or abs(timestamp-(frame-1)/fps) > .000002
                or row.get('ball_model') != model):
            raise ValueError('Review selection journal source/model/frame identity differs')
        journal_rows.append(row); offset += len(line)
    if len(journal_rows) != count:
        raise ValueError('Review selection journal is missing completed source frames')
    previous_end = 0
    for burst in bursts:
        if not isinstance(burst,dict): raise ValueError('Invalid review selection burst')
        start, end, anchor = (burst.get(key) for key in ('start_source_frame','end_source_frame','anchor_source_frame'))
        if (any(type(value) is not int for value in (start,end,anchor)) or start < 1
                or end != start+4 or end > count or start <= previous_end or not start <= anchor <= end):
            raise ValueError('Review selection bursts overlap, repeat, are unsorted, or exceed source bounds')
        if burst.get('reason') != 'ACCEPTED_BALL_HYPOTHESIS_IN_USER_DECLARED_NO_BALL_RECORDING':
            raise ValueError('Review selection burst reason is missing or unsupported')
        observation = journal_rows[anchor-1].get('ball')
        if (not isinstance(observation,dict) or burst.get('anchor_timestamp_s') != journal_rows[anchor-1]['source_time']
                or burst.get('anchor_center_normalized') != [observation.get('x'),observation.get('y')]
                or burst.get('anchor_uncalibrated_model_score') != observation.get('confidence')):
            raise ValueError('Review selection anchor differs from accepted journal hypothesis')
        accepted = sum(row.get('ball') is not None for row in journal_rows[start-1:end])
        if type(burst.get('accepted_hypotheses_in_burst')) is not int or burst['accepted_hypotheses_in_burst'] != accepted:
            raise ValueError('Review selection burst hypothesis count differs')
        previous_end = end
    if digest(path) != hashlib.sha256(data).hexdigest():
        raise ValueError('Review selection changed while auditing')
    return {'method':'review_selection','selection_path':str(path.resolve()),
            'selection_sha256':hashlib.sha256(data).hexdigest(),
            'requested_bursts':len(bursts),'burst_frames':5,
            'selection_method':selection['selection_method'],
            'identity':identity,'truth':truth,'bursts':bursts,
            'temporal_continuity_reviewed':False,
            'temporal_continuity_reason':'Source edit/cut/duplicate history remains unverified; review selection is not temporal training clearance.'}


def import_recordings(folder, output, session, bursts=6, resume=False, *,
                      start_frame=None, capture_sequence=1, note='', capture_origin='iphone',
                      review_selection=None):
    import cv2
    import numpy as np
    if not session.strip() or type(bursts) is not int or not 1 <= bursts <= 100:
        raise ValueError('Provide a session ID and 1–100 bursts per recording')
    if capture_origin not in ('iphone', 'camera', 'edited_export', 'unspecified'):
        raise ValueError('Choose the actual capture origin, or unspecified when its edit/source history is unknown')
    if type(capture_sequence) is not int or capture_sequence < 1:
        raise ValueError('Capture sequence must be a positive integer')
    if start_frame is not None and (not isinstance(start_frame, int) or start_frame < 1 or not folder.is_file()):
        raise ValueError('Targeted sampling requires one recording and a positive 1-based start frame')
    files = [folder] if folder.is_file() and folder.suffix.lower() in ('.mov', '.mp4', '.m4v') else (
        sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.mov', '.mp4', '.m4v')) if folder.is_dir() else [])
    if not files:
        raise ValueError('No camera recordings found')
    if review_selection is not None:
        if not folder.is_file() or len(files) != 1 or start_frame is not None or bursts != 6:
            raise ValueError('Review selection requires one recording without start-frame or a separate bursts override')
        if capture_sequence < 2:
            raise ValueError('Review selection requires a distinct capture sequence of 2 or higher')
        if capture_origin != 'unspecified':
            raise ValueError('Review selection imports require unspecified source origin until cut/edit history is audited')
        note = (note.rstrip()+' Source cut/edit/duplicate history remains unverified; selected hypotheses are not ground-truth labels.').strip()
    output.mkdir(parents=True, exist_ok=True)
    report = {'session_id': session, 'split': None, 'contains_ground_truth': False,
              'scope': 'Full sequential decoder count/timestamps plus sampled image import; not accuracy evaluation or independent codec validation',
              'recordings': []}
    for video in files:
        before = video.stat()
        sha = digest(video)
        selected = _review_selection(review_selection,video,session,sha) if review_selection is not None else None
        clip = output / ('phone_' + sha[:16] + '_' + str(capture_sequence))
        if clip.exists():
            if not resume:
                raise FileExistsError(f'Capture already exists; no files overwritten: {clip}')
            entry = json.loads((clip/'capture.json').read_text())
            rows = [json.loads(line) for line in (clip/'frames.jsonl').read_text().splitlines()]
            if entry.get('original_sha256') != sha or entry.get('session_id') != session or len(rows) != entry.get('sampled_frames'):
                raise ValueError(f'Existing capture provenance differs: {clip}')
            if entry.get('capture_origin', 'iphone') != capture_origin:
                raise ValueError(f'Existing capture origin differs: {clip}')
            if start_frame is not None and entry.get('selection') != {'method': 'targeted_failure', 'start_source_frame': start_frame, 'requested_bursts': bursts}:
                raise ValueError(f'Existing targeted sampling differs: {clip}')
            if selected is not None:
                if (entry.get('selection') != selected or entry.get('source_frames') != selected['identity']['source_frame_count']
                        or abs(entry.get('source_fps_average',0)-selected['identity']['source_fps']) > .00001):
                    raise ValueError(f'Existing review selection differs: {clip}')
                expected_frames = [frame for burst in selected['bursts'] for frame in range(burst['start_source_frame'],burst['end_source_frame']+1)]
                if [row.get('source_frame') for row in rows] != expected_frames:
                    raise ValueError(f'Existing review selection frame index differs: {clip}')
            elif entry.get('selection',{}).get('method') == 'review_selection':
                raise ValueError(f'Existing capture requires its exact review selection: {clip}')
            if start_frame is None and entry.get('selection', {}).get('method') == 'targeted_failure':
                raise ValueError(f'Existing capture is targeted, not uniform: {clip}')
            if any(not (clip/row['file']).is_file() for row in rows):
                raise ValueError(f'Existing capture images missing: {clip}')
            if selected is not None and _review_selection(review_selection,video,session,sha) != selected:
                raise ValueError(f'Review selection changed during resume: {clip}')
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
            if selected is not None:
                if _review_selection(review_selection,video,session,sha,count,fps) != selected:
                    raise ValueError('Review selection changed during sequential source count')
                starts = [burst['start_source_frame']-1 for burst in selected['bursts']]
            elif start_frame is not None:
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
                    elif selected is not None:
                        burst = selected['bursts'][burst_id-1]
                        rows[-1]['review_metadata'] = {'reasons':[burst['reason']], 'capture_note':note,
                                                      'selection_sha256':selected['selection_sha256'],
                                                      'selection_burst':burst}
                    annotations[name] = {'reviewed': False, 'ball': [], 'person': [], 'table': []}
            after = video.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise RuntimeError(f'Source changed during import: {video}')
            if selected is not None:
                if digest(video) != sha or _review_selection(review_selection,video,session,sha,count,fps) != selected:
                    raise ValueError('Review selection/source/journal changed before import publication')
            entry = {'source': 'recorded_camera', 'capture_origin': capture_origin, 'session_id': session,
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
                     'sampling': (f'{len(starts)} review-selected five-consecutive-frame bursts; hypotheses remain unreviewed'
                                  if selected is not None else f'{len(starts)} adjacent five-consecutive-frame bursts from source frame {start_frame}'
                                  if start_frame is not None else f'{len(starts)} evenly spaced five-consecutive-frame bursts'),
                     'selection': selected if selected is not None else {'method': 'targeted_failure' if start_frame is not None else 'uniform',
                                   'start_source_frame': start_frame, 'requested_bursts': bursts},
                     'capture_note': note,
                     'rotation_handling': 'decoder default; inspect imported orientation',
                     'hdr_metadata': None, 'hdr_metadata_reason': 'Not exposed by this OpenCV decoder'}
            (pending / 'capture.json').write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
            (pending / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            (pending / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n', encoding='utf-8')
            if selected is not None and _review_selection(review_selection,video,session,sha,count,fps) != selected:
                raise ValueError('Review selection journal changed during import publication')
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
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument('--bursts', type=int, default=6)
    selection_group.add_argument('--review-selection', type=Path, help='Exact audited review-burst selection; requires one recording, origin unspecified, and distinct capture sequence >=2')
    parser.add_argument('--resume', action='store_true', help='Verify and reuse completed imports; never overwrite annotations')
    parser.add_argument('--start-frame', type=int, help='1-based start of adjacent five-frame bursts; input must be one recording')
    parser.add_argument('--capture-sequence', type=int, default=1, help='Distinct capture ID suffix; existing captures are never overwritten')
    parser.add_argument('--note', default='', help='Human-readable review context; never a ground-truth label')
    parser.add_argument('--capture-origin', choices=['iphone','camera','edited_export','unspecified'], default='iphone', help='Preserve actual source history; unspecified does not establish temporal continuity')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = import_recordings(args.folder, args.output, args.session, args.bursts, args.resume,
                               start_frame=args.start_frame, capture_sequence=args.capture_sequence, note=args.note, capture_origin=args.capture_origin,
                               review_selection=args.review_selection)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
