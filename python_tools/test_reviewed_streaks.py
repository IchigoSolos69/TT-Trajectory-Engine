import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from export_reviewed_streaks import target, audit, export
from prepare_live_training import validate_annotation


def label():
    return {'reviewed': True, 'reviewed_classes': {'ball': True, 'person': False, 'table': False},
            'ball': [], 'person': [], 'table': [], 'geometry': {'visibility': 'visible',
            'blur_streak': {'method': 'manual_streak_endpoints', 'reviewed': True,
            'endpoints': [[.2, .3], [.3, .4]], 'source_width': 192, 'source_height': 108,
            'center': [1, 1], 'half_length_px': 999, 'axis_rad': 999}}}


class StreakTests(unittest.TestCase):
    def test_recompute_source_geometry_and_preserve_input(self):
        annotation = label(); before = copy.deepcopy(annotation)
        result = target(annotation, (192, 108))
        self.assertEqual(result['position_px'], [48, 37.8])
        self.assertAlmostEqual(result['half_length_px'], (19.2**2+10.8**2)**.5/2)
        self.assertEqual(annotation, before)
        annotation['geometry']['blur_streak']['endpoints'] = [[.4, .5], [.4, .5]]
        self.assertIsNone(target(annotation, (192, 108))['axis_rad'])
        self.assertEqual(target(annotation, (192, 108))['half_length_px'], 0)

    def test_unknown_and_occluded_never_negative(self):
        for visibility in ('unknown', 'occluded'):
            annotation = label(); annotation['geometry']['visibility'] = visibility
            self.assertIsNone(target(annotation, (192, 108)))
        annotation = label(); annotation['reviewed_classes']['ball'] = False
        self.assertIsNone(target(annotation, (192, 108)))
        annotation = label(); annotation['geometry']['blur_streak']['reviewed'] = False
        self.assertIsNone(target(annotation, (192, 108)))
        self.assertIsNone(validate_annotation(label(), ('ball',)), 'YOLO export cannot silently use a streak-only frame as an empty negative')
        absent = label(); absent['geometry'] = {'visibility': 'absent', 'blur_streak': None}
        self.assertEqual(target(absent, (192, 108))['visibility'], 'absent')
        self.assertEqual(validate_annotation(absent, ('ball',)), {'ball': []})
        absent['geometry']['blur_streak'] = label()['geometry']['blur_streak']
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            target(absent, (192, 108))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            target(label(), (384, 216))

    def fixture(self, root):
        captures = root/'captures'; captures.mkdir(); splits = {split: [split] for split in ('train', 'val', 'test')}
        split_file = root/'splits.json'; split_file.write_text(json.dumps(splits))
        for split_index, split in enumerate(splits):
            folder = captures/split; folder.mkdir()
            (folder/'capture.json').write_text(json.dumps({'source': 'recorded_camera', 'session_id': split}))
            records, annotations = [], {}
            for i in range(3):
                file = f'frame_{i+1}.jpg'; Image.new('RGB', (192, 108), (split_index*70, i*70, 60)).save(folder/file)
                records.append({'file': file, 'source_frame': i+1, 'elapsed_ms': i*1000/60, 'burst_id': 1})
                annotations[file] = label()
            (folder/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
            (folder/'annotations.json').write_text(json.dumps(annotations))
        return captures, split_file

    def test_windows_export_lineage_and_session_gate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as temp:
            root = Path(temp); captures, split_file = self.fixture(root)
            original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in captures.rglob('*') if path.is_file()}
            result = audit(captures, split_file)
            self.assertTrue(result['ready_for_training']); self.assertEqual(result['accepted_counts'], {'train': 1, 'val': 1, 'test': 1})
            output = root/'export'; export(captures, split_file, output)
            window = json.loads((output/'train.jsonl').read_text())
            self.assertEqual(window['latest_source_frame'], 3); self.assertEqual(window['input_order'], 'oldest_to_latest')
            self.assertEqual(window['frames'][0]['source_width'], 192)
            self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in original.items()))
            with self.assertRaises(FileExistsError): export(captures, split_file, output)
            manifest = captures/'test/capture.json'; manifest.write_text(json.dumps({'source': 'recorded_camera', 'session_id': 'train'}))
            with self.assertRaisesRegex(ValueError, 'session overlaps'): audit(captures, split_file)

    def test_window_gaps_and_duplicate_frames(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as temp:
            captures, split_file = self.fixture(Path(temp))
            index = captures/'train/frames.jsonl'; rows = [json.loads(line) for line in index.read_text().splitlines()]; rows[2]['source_frame'] = 5
            index.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result = audit(captures, split_file); self.assertFalse(result['ready_for_training']); self.assertEqual(result['excluded']['source_frame_gap'], 1)
            destination = captures/'test/frame_1.jpg'; destination.write_bytes((captures/'train/frame_1.jpg').read_bytes())
            with self.assertRaisesRegex(ValueError, 'across dataset splits'): audit(captures, split_file)


if __name__ == '__main__': unittest.main()
