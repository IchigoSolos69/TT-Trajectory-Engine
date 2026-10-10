"""Synthetic explicit-review subset tests; never modify real review annotations."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

# The scorer's CLI imports local tools directly; retain that import contract when
# this test is collected through the repository-root python_tools package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_reviewed_negative_subset as subset_module
from score_reviewed_negative_subset import score_subset

try:
    from .test_score_paired_ball_candidates import JournalFixture
except ImportError:
    from test_score_paired_ball_candidates import JournalFixture


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ReviewedSubsetFixture:
    """A complete strict journal plus real synthetic capture/index/JPEG files."""
    def __init__(self, root):
        self.root = root
        self.journal_dir = root / 'journal'
        self.journal_dir.mkdir()
        self.journal = JournalFixture(self.journal_dir)
        self.captures_root = root / 'captures'
        self.captures_root.mkdir()
        self.capture_ids = []

    def add_capture(self, source_frames, labels=None):
        labels = labels or {}
        source_hash = self.journal.manifest['source_sha256']
        clip = 'phone_' + source_hash[:16] + '_' + str(len(self.capture_ids) + 1)
        folder = self.captures_root / clip
        folder.mkdir()
        self.capture_ids.append(clip)
        rows, annotations = [], {}
        for position, fid in enumerate(source_frames, 1):
            filename = f'frame_{position}.jpg'
            elapsed = self.journal.rows[fid - 1]['source_pts_s'] * 1000
            row = {'file': filename, 'source_frame': fid, 'elapsed_ms': elapsed,
                   'burst_id': position, 'burst_index': 0, 'burst_length': 1}
            rows.append(row)
            # Different source frames have different decoded JPEG pixels. Copies
            # of one source frame retain identical JPEG bytes across captures.
            Image.new('RGB', (1024, 1024), (fid * 13, fid * 7, fid * 3)).save(
                folder / filename, format='JPEG', quality=95)
            kind = labels.get(fid, 'absent')
            annotation = {'reviewed': kind != 'unreviewed',
                          'reviewed_classes': {'ball': kind != 'unreviewed', 'person': False, 'table': False},
                          'ball': [[.1, .1, .05, .05]] if kind == 'positive' else [],
                          'person': [], 'table': [], 'reviewed_at': '2026-10-10T10:00:00.000Z',
                          'source_frame': fid, 'elapsed_ms': elapsed,
                          'geometry': {'visibility': 'visible' if kind == 'positive' else 'unknown' if kind in ('unknown', 'unreviewed') else kind,
                                       'blur_streak': None}}
            annotations[filename] = annotation
        capture = {'source': 'recorded_camera', 'capture_origin': 'unspecified',
                   'session_id': 'synthetic_review_session',
                   'original_path': self.journal.manifest['source_path'],
                   'original_sha256': source_hash, 'source_frames': 13,
                   'reported_source_frames': 13, 'frame_count_discrepancy': 0,
                   'full_source_decode_counted': True, 'source_fps_average': 50,
                   'sampled_frames': len(rows), 'width': 1024, 'height': 1024,
                   'contains_ground_truth': False, 'split': None,
                   'selection': {'method': 'uniform'},
                   'timestamp_basis': 'opencv_ffmpeg_source_pts_ms'}
        self.write_json(folder / 'capture.json', capture)
        self.write_index(folder, rows)
        self.write_json(folder / 'annotations.json', annotations)
        return folder

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n', encoding='utf-8')

    @staticmethod
    def write_index(folder, rows):
        (folder / 'frames.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')

    def interval_evidence(self, folder, filename='frame_1.jpg'):
        annotations_path = folder / 'annotations.json'
        annotations = json.loads(annotations_path.read_text())
        rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        capture = json.loads((folder / 'capture.json').read_text())
        previous = b'{"synthetic_prior_review":true}\n'
        previous_hash = sha(previous)
        history = folder / 'annotation_history'
        history.mkdir(exist_ok=True)
        (history / (previous_hash + '.json')).write_bytes(previous)
        annotation = annotations[filename]
        annotation['ball_absence_review'] = {
            'method': 'manual_interval', 'review_id': 'synthetic-reviewed-absence-interval',
            'reviewed_at': annotation['reviewed_at'], 'original_sha256': capture['original_sha256'],
            'session_id': capture['session_id'], 'index_sha256': sha((folder / 'frames.jsonl').read_bytes()),
            'image_sha256': sha((folder / filename).read_bytes()),
            'previous_annotation_revision': previous_hash,
            'start_source_frame': min(row['source_frame'] for row in rows),
            'end_source_frame': max(row['source_frame'] for row in rows), 'sampled_frames': len(rows)}
        self.write_json(annotations_path, annotations)


class ReviewedNegativeSubsetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture = ReviewedSubsetFixture(self.root)

    def run_score(self):
        return score_subset(self.fixture.journal_dir, self.fixture.captures_root,
                            self.fixture.capture_ids)

    def test_explicit_absent_subset_counts_real_audited_hypotheses_only(self):
        folder = self.fixture.add_capture([1, 3, 4, 11])
        annotation_bytes = (folder / 'annotations.json').read_bytes()
        result = self.run_score()
        self.assertEqual(result['journal_audit']['status'], 'PASS')
        self.assertEqual(result['source_frames_in_journal'], 13)
        self.assertEqual(result['reviewed_absent_unique_frames_scored'], 4)
        self.assertEqual(result['unsampled_source_frames_unknown'], 9)
        self.assertEqual(result['capture_reports'][0]['reviewed_absent_scored'], 4)
        expected = {'standalone_yolo': (3, 1), 'standalone_blurball_causal': (2, 2),
                    'experimental_selector': (1, 3)}
        for key, (false_positives, true_negatives) in expected.items():
            metrics = result['deduplicated_union_metrics'][key]
            self.assertEqual(metrics['reviewed_absent_frames'], 4)
            self.assertEqual(metrics['confirmed_absent_frame_false_positives'], false_positives)
            self.assertEqual(metrics['true_negatives'], true_negatives)
            self.assertIsNone(metrics['precision'])
            self.assertIsNone(metrics['recall'])
            self.assertIsNone(metrics['ball_miss_rate'])
        self.assertEqual((folder / 'annotations.json').read_bytes(), annotation_bytes)
        self.assertIsNone(result['full_video_accuracy'])
        self.assertIsNone(result['full_video_false_positive_count'])
        self.assertFalse(result['review_jpeg_matches_original_decoded_pixels_independently_verified'])
        self.assertFalse(result['promotion_approved'])
        self.assertTrue(result['snapshot_reverified_before_report'])

    def test_positive_unknown_occluded_and_unreviewed_are_excluded_without_relabeling(self):
        folder = self.fixture.add_capture([1, 2, 3, 4, 5, 6],
                                          {2: 'positive', 4: 'unknown', 5: 'occluded', 6: 'unreviewed'})
        original = (folder / 'annotations.json').read_bytes()
        result = self.run_score()
        self.assertEqual(result['reviewed_absent_unique_frames_scored'], 2)
        self.assertEqual(result['sampled_source_frames_excluded'], 4)
        records = {row['source_frame_id']: row for row in result['review_records']}
        self.assertEqual(records[2]['status'], 'positive')
        self.assertEqual(records[2]['reason'], 'REVIEWED_POSITIVE_EXCLUDED')
        for fid in (4, 5):
            self.assertEqual(records[fid]['reason'], 'UNKNOWN_OCCLUDED_OR_UNSUPPORTED_REVIEW_EXCLUDED')
        self.assertEqual(records[6]['reason'], 'NOT_EXPLICITLY_SAVED_BALL_REVIEW')
        self.assertEqual((folder / 'annotations.json').read_bytes(), original)

    def test_overlap_deduplicates_same_source_frame_in_union(self):
        self.fixture.add_capture([1, 3, 4])
        self.fixture.add_capture([3, 4, 11])
        result = self.run_score()
        self.assertEqual(result['sampled_review_records'], 6)
        self.assertEqual(result['unique_sampled_source_frames'], 4)
        self.assertEqual(result['duplicate_sample_records'], 2)
        self.assertEqual(result['reviewed_absent_unique_frames_scored'], 4)
        self.assertEqual(result['deduplicated_union_metrics']['experimental_selector']['confirmed_absent_frame_false_positives'], 1)
        self.assertEqual([capture['reviewed_absent_scored'] for capture in result['capture_reports']], [3, 3])

    def test_conflicting_absent_positive_duplicate_excludes_entire_source_group(self):
        self.fixture.add_capture([1, 3, 4])
        self.fixture.add_capture([3, 4, 11], {3: 'positive'})
        result = self.run_score()
        self.assertEqual(result['reviewed_absent_unique_frames_scored'], 3)
        self.assertEqual(result['sampled_source_frames_excluded'], 1)
        self.assertEqual(result['conflicting_source_frames'][0]['source_frame_id'], 3)
        conflicting = [row for row in result['review_records'] if row['source_frame_id'] == 3]
        self.assertEqual(len(conflicting), 2)
        self.assertTrue(all(row['status'] == 'conflicting' and row['reason'] ==
                            'CONFLICTING_DUPLICATE_REVIEWED_BALL_LABELS' for row in conflicting))
        self.assertEqual([capture['reviewed_absent_scored'] for capture in result['capture_reports']], [2, 2])
        self.assertEqual(result['deduplicated_union_metrics']['experimental_selector']['confirmed_absent_frame_false_positives'], 0)

    def test_differently_boxed_positive_duplicates_remain_positive_evidence_and_are_excluded(self):
        first = self.fixture.add_capture([1, 3], {3: 'positive'})
        second = self.fixture.add_capture([3, 11], {3: 'positive'})
        path = second / 'annotations.json'
        annotations = json.loads(path.read_text())
        annotations['frame_1.jpg']['ball'] = [[.2, .2, .05, .05]]
        self.fixture.write_json(path, annotations)
        originals = [(folder / 'annotations.json').read_bytes() for folder in (first, second)]
        result = self.run_score()
        self.assertEqual(result['reviewed_absent_unique_frames_scored'], 2)
        self.assertEqual(result['conflicting_source_frames'][0]['source_frame_id'], 3)
        self.assertEqual(result['conflicting_source_frames'][0]['reviewed_label_kinds'], ['positive'])
        positive_records = [row for row in result['review_records'] if row['source_frame_id'] == 3]
        self.assertTrue(all(row['status'] == 'conflicting' and row['label']['ball'] for row in positive_records))
        self.assertEqual([(folder / 'annotations.json').read_bytes() for folder in (first, second)], originals)

    def test_same_original_recording_cannot_claim_different_capture_sessions(self):
        self.fixture.add_capture([1, 3])
        second = self.fixture.add_capture([4, 11])
        path = second / 'capture.json'
        metadata = json.loads(path.read_text())
        metadata['session_id'] = 'different-synthetic-session'
        self.fixture.write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, '[Ss]ession'):
            self.run_score()

    def test_duplicate_source_frame_with_different_jpeg_pixels_rejected(self):
        self.fixture.add_capture([3])
        second = self.fixture.add_capture([3])
        Image.new('RGB', (1024, 1024), (200, 200, 200)).save(second / 'frame_1.jpg', format='JPEG', quality=95)
        with self.assertRaisesRegex(ValueError, 'different review image pixels'):
            self.run_score()

    def test_saved_source_frame_and_annotation_pts_must_match_capture_index(self):
        folder = self.fixture.add_capture([3])
        path = folder / 'annotations.json'
        original = json.loads(path.read_text())
        for field, value in [('source_frame', 4), ('elapsed_ms', 41)]:
            with self.subTest(field=field):
                annotations = copy.deepcopy(original)
                annotations['frame_1.jpg'][field] = value
                self.fixture.write_json(path, annotations)
                with self.assertRaisesRegex(ValueError, 'Saved review'):
                    self.run_score()

    def test_capture_pts_must_match_actual_journal_pts(self):
        folder = self.fixture.add_capture([3])
        rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        rows[0]['elapsed_ms'] += 2
        self.fixture.write_index(folder, rows)
        with self.assertRaisesRegex(ValueError, 'actual journal PTS'):
            self.run_score()

    def test_original_source_identity_and_bytes_are_required(self):
        folder = self.fixture.add_capture([3])
        path = folder / 'capture.json'
        capture = json.loads(path.read_text())
        capture['original_sha256'] = 'a' * 64
        self.fixture.write_json(path, capture)
        with self.assertRaisesRegex(ValueError, 'original source identity'):
            self.run_score()
        capture['original_sha256'] = self.fixture.journal.manifest['source_sha256']
        replacement = self.root / 'different_source.bin'
        replacement.write_bytes(b'changed recording')
        capture['original_path'] = str(replacement)
        self.fixture.write_json(path, capture)
        with self.assertRaisesRegex(ValueError, 'Original recording bytes changed'):
            self.run_score()

    def test_review_image_dimensions_match_the_audited_source_frame(self):
        folder = self.fixture.add_capture([3])
        Image.new('RGB', (32, 32), (10, 10, 10)).save(folder / 'frame_1.jpg', format='JPEG')
        with self.assertRaisesRegex(ValueError, 'source dimensions differ'):
            self.run_score()

    def test_interval_absence_evidence_pins_jpeg_index_source_and_prior_revision(self):
        folder = self.fixture.add_capture([3, 4])
        self.fixture.interval_evidence(folder)
        self.assertIsNotNone(self.run_score()['review_records'][0]['interval_absence_provenance'])
        path = folder / 'annotations.json'
        original = json.loads(path.read_text())
        for field in ('original_sha256', 'index_sha256', 'image_sha256',
                      'previous_annotation_revision', 'session_id'):
            with self.subTest(field=field):
                annotations = copy.deepcopy(original)
                annotations['frame_1.jpg']['ball_absence_review'][field] = 'b' * 64
                self.fixture.write_json(path, annotations)
                with self.assertRaises(ValueError):
                    self.run_score()
        self.fixture.write_json(path, original)
        Image.new('RGB', (1024, 1024), (240, 240, 240)).save(folder / 'frame_1.jpg', format='JPEG', quality=95)
        with self.assertRaisesRegex(ValueError, 'source, image, index or session changed'):
            self.run_score()

    def test_interval_absence_prior_review_bytes_cannot_change(self):
        folder = self.fixture.add_capture([3])
        self.fixture.interval_evidence(folder)
        annotations = json.loads((folder / 'annotations.json').read_text())
        previous = annotations['frame_1.jpg']['ball_absence_review']['previous_annotation_revision']
        (folder / 'annotation_history' / (previous + '.json')).write_bytes(b'overwritten prior review')
        with self.assertRaisesRegex(ValueError, 'prior review is unavailable or changed'):
            self.run_score()

    def test_annotation_revision_race_fails_closed_using_actual_snapshot_verifier(self):
        folder = self.fixture.add_capture([3])
        path = folder / 'annotations.json'
        real_verify = subset_module.verify_snapshot

        def mutate_then_verify(watched):
            path.write_bytes(path.read_bytes() + b'\n')
            return real_verify(watched)

        with mock.patch.object(subset_module, 'verify_snapshot', side_effect=mutate_then_verify):
            with self.assertRaisesRegex(ValueError, 'review changed during scoring'):
                self.run_score()

    def test_only_complete_strict_journals_are_accepted(self):
        self.fixture.add_capture([3])
        completion = self.fixture.journal_dir / 'completion.json'
        value = json.loads(completion.read_text())
        value['status'] = 'PARTIAL'
        self.fixture.write_json(completion, value)
        with self.assertRaisesRegex(ValueError, 'status'):
            self.run_score()

    def test_saved_review_time_and_class_flags_are_explicit(self):
        folder = self.fixture.add_capture([3])
        path = folder / 'annotations.json'
        original = json.loads(path.read_text())
        for reviewed_at in (None, 'not-a-time', '2026-10-10T10:00:00'):
            with self.subTest(reviewed_at=reviewed_at):
                annotations = copy.deepcopy(original)
                annotations['frame_1.jpg']['reviewed_at'] = reviewed_at
                self.fixture.write_json(path, annotations)
                with self.assertRaises(ValueError):
                    self.run_score()
        annotations = copy.deepcopy(original)
        annotations['frame_1.jpg'].pop('reviewed_classes')
        self.fixture.write_json(path, annotations)
        result = self.run_score()
        self.assertEqual(result['reviewed_absent_unique_frames_scored'], 0)
        self.assertEqual(result['review_records'][0]['reason'], 'NOT_EXPLICITLY_SAVED_BALL_REVIEW')

    def test_cli_creates_new_report_and_does_not_overwrite(self):
        self.fixture.add_capture([1, 3, 4, 11])
        output = self.root / 'negative_metrics.json'
        command = [sys.executable, str(Path(__file__).with_name('score_reviewed_negative_subset.py')),
                   str(self.fixture.journal_dir), str(self.fixture.captures_root), str(output),
                   '--capture', self.fixture.capture_ids[0]]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = output.read_bytes()
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
