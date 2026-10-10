"""Independent source/lineage checks for the reviewed BlurBall handoff.

These fixtures are isolated, explicitly reviewed synthetic labels. They never
touch live annotations or count as model-accuracy evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from PIL import Image

from export_reviewed_streaks import export
from blurball_reviewed_dataset import audit_export, ReviewedBlurBallDataset


class IndependentDatasetIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.captures = self.root / 'captures'
        self.captures.mkdir()
        self.splits = self.root / 'splits.json'
        self.splits.write_text(json.dumps({split: [split] for split in ('train', 'val', 'test')}), encoding='utf-8')
        self.output_number = 0
        for split_number, split in enumerate(('train', 'val', 'test')):
            directory = self.captures / split
            directory.mkdir()
            original = self.root / f'{split}.synthetic_source'
            original.write_bytes(f'Isolated structural fixture, not gameplay: {split}'.encode())
            capture = {'source': 'recorded_camera', 'session_id': f'independent_session_{split}',
                       'original_path': str(original), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                       'width': 192, 'height': 108, 'burst_frames': 5, 'sampled_frames': 5}
            (directory / 'capture.json').write_text(json.dumps(capture), encoding='utf-8')
            frames, annotations = [], {}
            for number in range(1, 6):
                name = f'frame_{number}.jpg'
                Image.new('RGB', (192, 108), (split_number * 70, number * 30, 90)).save(directory / name)
                row = {'file': name, 'source_frame': number, 'elapsed_ms': (number - 1) * 1000 / 60,
                       'burst_id': 1, 'burst_index': number - 1, 'burst_length': 5}
                frames.append(row)
                annotations[name] = {'reviewed': True, 'reviewed_classes': {'ball': True, 'person': False, 'table': False},
                                     'ball': [], 'person': [], 'table': [], 'source_frame': number,
                                     'elapsed_ms': row['elapsed_ms'], 'reviewed_at': '2026-10-09T00:00:00Z',
                                     'geometry': {'visibility': 'visible', 'blur_streak': {
                                         'method': 'manual_streak_endpoints', 'reviewed': True,
                                         'endpoints': [[.2, .3], [.3, .4]], 'source_width': 192, 'source_height': 108}}}
            self.write_rows(split, frames)
            self.write_annotations(split, annotations)

    def tearDown(self):
        self.temp.cleanup()

    def read_rows(self, split='train'):
        return [json.loads(line) for line in (self.captures / split / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]

    def write_rows(self, split, rows):
        (self.captures / split / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def read_annotations(self, split='train'):
        return json.loads((self.captures / split / 'annotations.json').read_text(encoding='utf-8'))

    def write_annotations(self, split, value):
        (self.captures / split / 'annotations.json').write_text(json.dumps(value), encoding='utf-8')

    def prepare_and_audit(self):
        self.output_number += 1
        output = self.root / f'export_{self.output_number}'
        export(self.captures, self.splits, output)
        return output, audit_export(output, self.captures, self.splits, loss_choice='quality_focal')

    def test_fresh_handoff_and_inputs_are_immutable(self):
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.captures.rglob('*') if path.is_file()}
        output, result = self.prepare_and_audit()
        manifest, windows = result
        self.assertIsInstance(manifest, dict)
        self.assertEqual(set(windows), {'train', 'val', 'test'})
        self.assertTrue((output / 'audit.json').is_file())
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items()))

    def test_unterminated_index_is_rejected_before_training(self):
        path = self.captures / 'train' / 'frames.jsonl'
        path.write_text(path.read_text(encoding='utf-8').rstrip('\n'), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_blank_index_row_is_rejected_before_training(self):
        path = self.captures / 'train' / 'frames.jsonl'
        lines = path.read_text(encoding='utf-8').splitlines()
        path.write_text(lines[0] + '\n\n' + '\n'.join(lines[1:]) + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_duplicate_filename_and_id_are_rejected_globally(self):
        original = self.read_rows()
        rows = copy.deepcopy(original)
        rows[-1]['file'] = rows[0]['file']
        self.write_rows('train', rows)
        with self.assertRaises(ValueError):
            self.prepare_and_audit()
        rows = copy.deepcopy(original)
        rows[-1]['source_frame'] = rows[0]['source_frame']
        self.write_rows('train', rows)
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_source_frame_ids_are_positive(self):
        rows = self.read_rows()
        for row in rows:
            row['source_frame'] -= 1
        self.write_rows('train', rows)
        annotations = self.read_annotations()
        for value in annotations.values():
            value['source_frame'] -= 1
        self.write_annotations('train', annotations)
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_stale_annotation_source_identity_is_rejected(self):
        annotations = self.read_annotations()
        annotations['frame_3.jpg']['source_frame'] = 999
        self.write_annotations('train', annotations)
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_stale_annotation_timestamp_is_rejected(self):
        annotations = self.read_annotations()
        annotations['frame_3.jpg']['elapsed_ms'] += 10
        self.write_annotations('train', annotations)
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_declared_sample_count_requires_complete_index(self):
        rows = self.read_rows()
        self.write_rows('train', rows[:-1])
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_global_source_order_cannot_hide_invalid_windows(self):
        rows = self.read_rows()
        rows[-2], rows[-1] = rows[-1], rows[-2]
        self.write_rows('train', rows)
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_absence_requires_explicit_empty_observations(self):
        annotations = self.read_annotations()
        annotations['frame_3.jpg']['geometry'] = {'visibility': 'absent', 'blur_streak': None}
        del annotations['frame_3.jpg']['ball']
        self.write_annotations('train', annotations)
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_exact_decoded_image_leakage_rejects_different_jpeg_bytes(self):
        source = self.captures / 'train' / 'frame_1.jpg'
        destination = self.captures / 'test' / 'frame_1.jpg'
        comment = b'independent duplicate-encoding fixture'
        raw = source.read_bytes()
        destination.write_bytes(raw[:2] + b'\xff\xfe' + struct.pack('>H', len(comment) + 2) + comment + raw[2:])
        self.assertNotEqual(hashlib.sha256(source.read_bytes()).digest(), hashlib.sha256(destination.read_bytes()).digest())
        with Image.open(source) as first, Image.open(destination) as second:
            self.assertEqual(first.convert('RGB').tobytes(), second.convert('RGB').tobytes())
        with self.assertRaises(ValueError):
            self.prepare_and_audit()

    def test_one_gameplay_session_cannot_cross_splits(self):
        path = self.captures / 'test' / 'capture.json'
        capture = json.loads(path.read_text(encoding='utf-8'))
        capture['session_id'] = 'independent_session_train'
        path.write_text(json.dumps(capture), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'session overlaps'):
            self.prepare_and_audit()

    def test_original_recording_cannot_cross_splits_with_different_session_names(self):
        source = json.loads((self.captures / 'train' / 'capture.json').read_text(encoding='utf-8'))
        path = self.captures / 'test' / 'capture.json'
        capture = json.loads(path.read_text(encoding='utf-8'))
        capture['original_sha256'], capture['original_path'] = source['original_sha256'], source['original_path']
        path.write_text(json.dumps(capture), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Original recording overlaps'):
            self.prepare_and_audit()

    def test_source_annotation_mutation_invalidates_saved_handoff(self):
        output, _ = self.prepare_and_audit()
        annotations = self.read_annotations()
        annotations['frame_1.jpg']['reviewed_at'] = '2026-10-10T00:00:00Z'
        self.write_annotations('train', annotations)
        with self.assertRaises(ValueError):
            audit_export(output, self.captures, self.splits, loss_choice='quality_focal')

    def test_split_row_target_mutation_is_rejected(self):
        output, _ = self.prepare_and_audit()
        path = output / 'train.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['frames'][0]['target']['position_px'][0] += 1
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        with self.assertRaises(ValueError):
            audit_export(output, self.captures, self.splits, loss_choice='quality_focal')

    def test_unknown_targets_do_not_become_negative_or_future_context(self):
        annotations = self.read_annotations()
        annotations['frame_5.jpg']['geometry'] = {'visibility': 'unknown', 'blur_streak': None}
        annotations['frame_5.jpg']['ball'] = []
        self.write_annotations('train', annotations)
        output, (_, rows) = self.prepare_and_audit()
        self.assertEqual(len(rows['train']), 2)
        self.assertTrue(all(frame['source_frame'] != 5 for row in rows['train'] for frame in row['frames']))
        dataset = ReviewedBlurBallDataset(output, self.captures, self.splits, 'train', loss_choice='quality_focal')
        inputs, targets, provenance = dataset.sample(0)
        self.assertEqual(inputs.shape, (9, 288, 512))
        self.assertEqual(targets[0].shape, (3, 288, 512))
        self.assertEqual([frame['source_frame'] for frame in provenance['frames']], [1, 2, 3])
        self.assertEqual(provenance['latest_source_frame'], 3)
        self.assertEqual(provenance['frames'][-1]['elapsed_ms'], 2000 / 60)
        self.assertTrue(all(frame['source_frame'] <= provenance['latest_source_frame'] for frame in provenance['frames']))
        self.assertTrue(all(target.any() for target in targets[0]), 'Reviewed visible labels must never become blank absence targets')


if __name__ == '__main__':
    unittest.main()
