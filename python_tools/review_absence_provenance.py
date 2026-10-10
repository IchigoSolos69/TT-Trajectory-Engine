"""Validate explicit interval absence against immutable source/review evidence."""
import copy
import hashlib
import re


def checked_interval_absence(annotation, capture, row, index_sha256, image_sha256, history_directory):
    value = annotation.get('ball_absence_review')
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('method') != 'manual_interval':
        raise ValueError('Invalid interval absence review provenance')
    flags = annotation.get('reviewed_classes')
    if not isinstance(flags, dict) or flags.get('ball') is not True or annotation.get('ball') != [] or (annotation.get('geometry') or {}).get('visibility') != 'absent' or (annotation.get('geometry') or {}).get('blur_streak') is not None:
        raise ValueError('Interval absence review conflicts with ball annotation')
    for key in ('original_sha256', 'index_sha256', 'image_sha256', 'previous_annotation_revision'):
        if not isinstance(value.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', value[key]):
            raise ValueError('Interval absence requires source/review hashes')
    if value['original_sha256'] != capture.get('original_sha256') or value['index_sha256'] != index_sha256 or value['image_sha256'] != image_sha256 or not value.get('session_id') or value['session_id'] != capture.get('session_id'):
        raise ValueError('Interval absence source, image, index or session changed')
    start, end, count = (value.get(key) for key in ('start_source_frame', 'end_source_frame', 'sampled_frames'))
    if any(type(item) is not int for item in (start, end, count)) or not 1 <= start <= row['source_frame'] <= end or not 1 <= count <= 5000:
        raise ValueError('Interval absence does not cover this source frame')
    if not isinstance(value.get('review_id'), str) or not value['review_id'] or not isinstance(value.get('reviewed_at'), str) or not value['reviewed_at'] or annotation.get('reviewed_at') != value['reviewed_at']:
        raise ValueError('Interval absence reviewer/time evidence is missing')
    previous = history_directory / (value['previous_annotation_revision'] + '.json')
    if not previous.is_file() or hashlib.sha256(previous.read_bytes()).hexdigest() != value['previous_annotation_revision']:
        raise ValueError('Interval absence immutable prior review is unavailable or changed')
    return copy.deepcopy(value)
