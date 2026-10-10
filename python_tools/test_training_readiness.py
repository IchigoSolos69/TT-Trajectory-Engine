import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from audit_training_readiness import audit


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, name, seed, session, count=5, reviewed=True):
        folder = self.root / name; folder.mkdir()
        source = self.root / (name + '.mov'); source.write_bytes(name.encode())
        cap = {'source': 'recorded_camera', 'original_path': str(source),
               'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'session_id': session, 'sampled_frames': count, 'width': 32, 'height': 24, 'split': None}
        (folder / 'capture.json').write_text(json.dumps(cap))
        rows, anns = [], {}
        for i in range(count):
            file = f'frame_{i+1}.jpg'
            Image.new('RGB', (32, 24), (seed, i * 5, 10)).save(folder / file)
            rows.append({'file': file, 'source_frame': i+1, 'elapsed_ms': i * 16.7, 'burst_id': i // 5 + 1})
            absent = i % 6 == 0
            geometry = {'visibility': 'absent' if absent else 'visible'}
            if not absent:
                geometry['blur_streak'] = {'reviewed': True, 'method': 'manual_streak_endpoints',
                    'source_width': 32, 'source_height': 24, 'endpoints': [[.3, .4], [.35, .4]]}
            anns[file] = {'reviewed': reviewed, 'reviewed_classes': {'ball': reviewed, 'person': False, 'table': False},
                           'ball': [] if absent else [[.3, .3, .1, .1]], 'person': [], 'table': [], 'geometry': geometry}
        (folder / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (folder / 'annotations.json').write_text(json.dumps(anns))
        return folder

    def splits(self, value):
        file = self.root / 'splits.json'; file.write_text(json.dumps(value)); return file

    def test_unknowns_never_become_absence(self):
        self.capture('clip', 20, 'one', reviewed=False)
        r = audit(self.root)
        self.assertEqual(r['counts_by_split']['unassigned']['ball_unknown_or_excluded'], 5)
        self.assertEqual(r['counts_by_split']['unassigned'].get('ball_verified_absent', 0), 0)
        self.assertFalse(r['ball']['structurally_ready'])

    def test_same_source_frame_duplicate_counts_once(self):
        import shutil
        one = self.capture('one', 20, 'same')
        shutil.copytree(one, self.root / 'copy')
        r = audit(self.root)
        self.assertEqual(r['saved_images'], 10)
        self.assertEqual(r['deduplicated_review_images'], 5)
        self.assertEqual(r['counts_by_split']['unassigned']['ball_positive'], 4)
        self.assertEqual(r['blurball']['structural_windows'], 3)

    def test_duplicate_review_conflict_excluded(self):
        import shutil
        one = self.capture('one', 20, 'same')
        copy = self.root / 'copy'; shutil.copytree(one, copy)
        path = copy / 'annotations.json'; anns = json.loads(path.read_text())
        anns['frame_2.jpg']['ball'] = [[.6, .3, .1, .1]]
        path.write_text(json.dumps(anns))
        r = audit(self.root)
        self.assertIn('CONFLICTING_DUPLICATE_LABELS', r['ball']['reason_codes'])
        self.assertEqual(r['counts_by_split']['unassigned']['ball_positive'], 3)

    def test_session_leak_and_decoded_image_leak_reported(self):
        import shutil
        a = self.capture('a', 20, 'same'); b = self.capture('b', 40, 'same'); self.capture('c', 60, 'third')
        shutil.copyfile(a / 'frame_1.jpg', b / 'frame_1.jpg')
        r = audit(self.root, self.splits({'train': ['a'], 'val': ['b'], 'test': ['c']}))
        self.assertIn('SESSION_SPLIT_LEAKAGE', r['ball']['reason_codes'])
        self.assertIn('SOURCE_OR_DECODED_IMAGE_SPLIT_LEAKAGE', r['ball']['reason_codes'])

    def test_structural_ball_gate_does_not_claim_cnn_continuity(self):
        for i, s in enumerate(('train', 'val', 'test')):
            self.capture(s, 20 + i * 40, s, count=30)
        r = audit(self.root, self.splits({s: [s] for s in ('train', 'val', 'test')}))
        self.assertTrue(r['ball']['structurally_ready'])
        self.assertFalse(r['blurball']['ready_for_fitting'])
        self.assertIn('TEMPORAL_CONTINUITY_NOT_AUDITED', r['blurball']['reason_codes'])

    def test_occlusion_and_frame_gap_do_not_train_as_negative_or_window(self):
        folder = self.capture('one', 20, 'one')
        ap = folder / 'annotations.json'; anns = json.loads(ap.read_text())
        anns['frame_1.jpg']['geometry']['visibility'] = 'occluded'; ap.write_text(json.dumps(anns))
        ip = folder / 'frames.jsonl'; rows = [json.loads(l) for l in ip.read_text().splitlines()]
        rows[-1]['source_frame'] = 20; rows[-1]['elapsed_ms'] = 200
        ip.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        r = audit(self.root)
        self.assertEqual(r['counts_by_split']['unassigned'].get('ball_verified_absent', 0), 0)
        self.assertEqual(r['blurball']['structural_windows'], 1)

    def test_changed_original_source_fails_closed(self):
        self.capture('one', 20, 'one')
        (self.root / 'one.mov').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            audit(self.root)

    def test_invalid_cnn_geometry_preserves_valid_ball_target(self):
        folder = self.capture('one', 20, 'one')
        path = folder / 'annotations.json'; anns = json.loads(path.read_text())
        anns['frame_2.jpg']['geometry']['blur_streak']['source_width'] = 999
        path.write_text(json.dumps(anns))
        r = audit(self.root)
        self.assertEqual(r['counts_by_split']['unassigned']['ball_positive'], 4)
        self.assertIn('INVALID_CNN_TARGET', r['blurball']['reason_codes'])
        self.assertNotIn('INVALID_CNN_TARGET', r['ball']['reason_codes'])

    def test_annotation_revision_change_during_audit_fails_closed(self):
        from audit_training_readiness import target as original_target
        folder = self.capture('one', 20, 'one')
        path = folder / 'annotations.json'
        def changed(annotation, dimensions):
            value = original_target(annotation, dimensions)
            saved = json.loads(path.read_text())
            saved['revision_marker'] = {'reviewed': False}
            path.write_text(json.dumps(saved))
            return value
        with patch('audit_training_readiness.target', changed):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                audit(self.root)


if __name__ == '__main__':
    unittest.main()
