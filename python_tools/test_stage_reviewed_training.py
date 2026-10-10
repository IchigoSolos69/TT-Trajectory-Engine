import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from stage_reviewed_training import (StagedReviewedDataset, load_stage, snapshot,
                                     write_stage)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.captures = self.base / 'captures'
        self.captures.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, name='one', seed=20, session='session', count=5, kind='point'):
        folder = self.captures / name
        folder.mkdir()
        original = self.base / (name + '.mov')
        original.write_bytes((name + ' original').encode())
        capture = {'source': 'recorded_camera', 'session_id': session, 'split': None,
                   'original_path': str(original), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                   'sampled_frames': count, 'width': 32, 'height': 24}
        (folder / 'capture.json').write_text(json.dumps(capture))
        rows, annotations = [], {}
        for i in range(count):
            filename = f'frame_{i+1}.jpg'
            Image.new('RGB', (32, 24), (seed, i * 20, 10)).save(folder / filename)
            rows.append({'file': filename, 'source_frame': i+1, 'elapsed_ms': i * 16.7, 'burst_id': 1})
            geometry = {'visibility': 'absent' if kind == 'absent' else 'visible'}
            if kind != 'absent':
                geometry['blur_streak'] = {'reviewed': True, 'method': 'manual_streak_endpoints',
                                          'source_width': 32, 'source_height': 24,
                                          'endpoints': [[.3, .4], [.3 if kind == 'point' else .4, .4]]}
            annotations[filename] = {'reviewed': True,
                                    'reviewed_classes': {'ball': True, 'person': False, 'table': False},
                                    'ball': [] if kind == 'absent' else [[.25, .3, .2, .2]],
                                    'person': [], 'table': [], 'geometry': geometry}
        (folder / 'annotations.json').write_text(json.dumps(annotations))
        self.write_rows(folder, rows)
        return folder

    @staticmethod
    def write_rows(folder, rows):
        (folder / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    @staticmethod
    def edit_annotations(folder, edit):
        path = folder / 'annotations.json'
        annotations = json.loads(path.read_text())
        edit(annotations)
        path.write_text(json.dumps(annotations))

    def test_new_manual_sharp_points_form_real_causal_windows(self):
        self.capture()
        report = write_stage(self.captures, 'session', self.base / 'stage')
        self.assertEqual(report['counts']['visible_point_frames'], 5)
        self.assertEqual(report['counts']['all_visible_windows'], 3)
        self.assertFalse(report['ready_for_fitting'])
        dataset = StagedReviewedDataset(self.base / 'stage')
        x, y, provenance = dataset.sample(0)
        self.assertEqual(x.shape, (9, 288, 512))
        self.assertEqual(y[0].shape, (3, 288, 512))
        self.assertTrue(np.all(y[0].sum(axis=(1, 2)) > 0))
        self.assertEqual([frame['source_frame'] for frame in provenance['frames']], [1, 2, 3])
        self.assertEqual(provenance['partition'], 'unassigned')
        self.assertEqual(provenance['latest_source_frame'], 3)
        self.assertEqual(provenance['target_method'], 'continuous_binary_capsule_v1')
        self.assertTrue(np.isfinite(x).all())

    def test_absent_frames_produce_blank_targets_not_unknown_negatives(self):
        folder = self.capture(kind='absent')
        self.edit_annotations(folder, lambda annotations: annotations['frame_5.jpg']['reviewed_classes'].update(ball=False))
        report = write_stage(self.captures, 'session', self.base / 'stage')
        self.assertEqual(report['counts']['absent_frames'], 4)
        self.assertEqual(report['counts']['unknown_or_excluded_frames'], 1)
        self.assertEqual(report['counts']['all_absent_windows'], 2)
        self.assertFalse(StagedReviewedDataset(self.base / 'stage').sample(0)[1][0].any())

    def test_unknown_partial_occluded_or_missing_streak_cannot_pad_windows(self):
        for ordinal, kind in enumerate(('unknown', 'partial', 'occluded', 'missing_streak')):
            with self.subTest(kind=kind):
                folder = self.capture(name=kind, seed=30 + ordinal * 20)
                def edit(annotations):
                    annotation = annotations['frame_3.jpg']
                    if kind == 'missing_streak':
                        annotation['geometry']['blur_streak'] = None
                    else:
                        annotation['geometry']['visibility'] = kind
                self.edit_annotations(folder, edit)
        report, windows = snapshot(self.captures, 'session')
        self.assertEqual(len(windows), 0)
        self.assertEqual(report['counts']['unknown_or_excluded_frames'], 4)

    def test_consistent_cross_capture_source_and_image_duplicates_count_once(self):
        one = self.capture()
        shutil.copytree(one, self.captures / 'copy')
        report, windows = snapshot(self.captures, 'session')
        self.assertEqual(report['saved_frames'], 10)
        self.assertEqual(report['unique_review_images'], 5)
        self.assertEqual(len(windows), 3)
        self.assertEqual(report['exclusions']['duplicate_window'], 3)

    def test_conflicting_duplicate_cnn_targets_exclude_all_overlapping_windows(self):
        one = self.capture()
        other = self.captures / 'copy'
        shutil.copytree(one, other)
        self.edit_annotations(other, lambda annotations: annotations['frame_3.jpg']['geometry']['blur_streak'].update(endpoints=[[.6, .4], [.6, .4]]))
        report, windows = snapshot(self.captures, 'session')
        self.assertEqual(len(windows), 0)
        self.assertEqual(report['counts']['visible_point_frames'], 4)
        self.assertEqual(report['counts']['unknown_or_excluded_frames'], 1)
        self.assertEqual([r['kind'] for r in report['conflicts']], ['cnn'])

    def test_ball_rectangle_conflict_or_invalid_box_does_not_erase_reviewed_cnn(self):
        one = self.capture()
        other = self.captures / 'copy'
        shutil.copytree(one, other)
        self.edit_annotations(other, lambda annotations: annotations['frame_3.jpg'].update(ball=[[.6, .3, .1, .1]]))
        self.edit_annotations(one, lambda annotations: annotations['frame_4.jpg'].update(ball=[[2., .3, .1, .1]]))
        report, windows = snapshot(self.captures, 'session')
        self.assertEqual(len(windows), 3)
        self.assertEqual(report['counts']['visible_point_frames'], 5)
        self.assertEqual(report['conflicts'][0]['kind'], 'ball')
        self.assertEqual(report['errors'][0]['kind'], 'ball')

    def test_source_frame_disagrees_on_pixels_or_pts_fails_closed(self):
        one = self.capture()
        other = self.captures / 'copy'
        shutil.copytree(one, other)
        Image.new('RGB', (32, 24), 'red').save(other / 'frame_3.jpg')
        with self.assertRaisesRegex(ValueError, 'conflicting decoded pixels'):
            snapshot(self.captures, 'session')

    def test_repeated_decoded_image_inside_window_is_not_temporal_padding(self):
        one = self.capture(count=3)
        shutil.copyfile(one / 'frame_1.jpg', one / 'frame_2.jpg')
        report, windows = snapshot(self.captures, 'session')
        self.assertEqual(windows, [])
        self.assertEqual(report['exclusions']['repeated_source_or_decoded_image_window'], 1)

    def test_missing_burst_and_frame_gap_cannot_form_windows(self):
        one = self.capture(count=3)
        rows = [json.loads(line) for line in (one / 'frames.jsonl').read_text().splitlines()]
        rows[-1]['source_frame'] = 10
        rows[-1]['elapsed_ms'] = 300
        self.write_rows(one, rows)
        self.assertEqual(snapshot(self.captures, 'session')[1], [])
        rows[-1]['source_frame'] = 3
        rows[0]['burst_id'] = None
        self.write_rows(one, rows)
        self.assertEqual(snapshot(self.captures, 'session')[1], [])

    def test_complete_source_stale_annotation_image_or_index_rejected_on_load(self):
        for ordinal, kind in enumerate(('annotation', 'image', 'index', 'original')):
            with self.subTest(kind=kind):
                folder = self.capture(name=kind, seed=20 + ordinal * 20, session=kind)
                stage = self.base / ('stage_' + kind)
                write_stage(self.captures, kind, stage)
                if kind == 'annotation':
                    self.edit_annotations(folder, lambda annotations: annotations['frame_3.jpg']['geometry']['blur_streak'].update(endpoints=[[.6, .4], [.6, .4]]))
                elif kind == 'image':
                    Image.new('RGB', (32, 24), 'red').save(folder / 'frame_3.jpg')
                elif kind == 'index':
                    rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
                    rows[-1]['elapsed_ms'] += 1
                    self.write_rows(folder, rows)
                else:
                    (self.base / (kind + '.mov')).write_bytes(b'changed original')
                with self.assertRaises(ValueError):
                    load_stage(stage)
                # Restore this test source so later session catalogs stay valid.
                if kind == 'original':
                    (self.base / (kind + '.mov')).write_bytes((kind + ' original').encode())

    def test_loaded_dataset_detects_later_edits_and_protects_public_rows(self):
        folder = self.capture()
        write_stage(self.captures, 'session', self.base / 'stage')
        dataset = StagedReviewedDataset(self.base / 'stage')
        rows = dataset.rows
        rows[0]['frames'][0]['target']['visibility'] = 'absent'
        self.assertTrue(dataset.sample(0)[1][0].any())
        self.edit_annotations(folder, lambda annotations: annotations['frame_3.jpg'].update(reviewed_at='new edit'))
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            dataset.sample(0)

    def test_sessions_never_get_invented_train_validation_test_assignments(self):
        self.capture(name='one', seed=20, session='first')
        self.capture(name='two', seed=40, session='later')
        report, windows = snapshot(self.captures, 'first')
        self.assertEqual(report['capture_count'], 1)
        self.assertTrue(all(row['session_id'] == 'first' and row['partition'] == 'unassigned' for row in windows))
        self.assertFalse(report['ready_for_fitting'])
        self.assertIn('INDEPENDENT_VALIDATION_TEST_SESSIONS_MISSING', report['reason_codes'])

    def test_same_recording_or_decoded_image_cannot_be_renamed_to_new_session(self):
        one = self.capture()
        other = self.captures / 'copy'
        shutil.copytree(one, other)
        capfile = other / 'capture.json'
        cap = json.loads(capfile.read_text()); cap['session_id'] = 'different'; capfile.write_text(json.dumps(cap))
        with self.assertRaisesRegex(ValueError, 'different sessions'):
            snapshot(self.captures, 'session')

    def test_different_recording_duplicate_pixels_across_sessions_rejected(self):
        one = self.capture(name='one', seed=20)
        other = self.capture(name='other', seed=40, session='different')
        shutil.copyfile(one / 'frame_1.jpg', other / 'frame_1.jpg')
        with self.assertRaisesRegex(ValueError, 'decoded image occurs in different sessions'):
            snapshot(self.captures, 'session')

    def test_declared_split_and_tampered_stage_rejected(self):
        folder = self.capture()
        stage = self.base / 'stage'
        write_stage(self.captures, 'session', stage)
        with (stage / 'windows.jsonl').open('ab') as stream:
            stream.write(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'windows changed'):
            load_stage(stage)
        cp = folder / 'capture.json'; cap = json.loads(cp.read_text()); cap['split'] = 'train'; cp.write_text(json.dumps(cap))
        with self.assertRaisesRegex(ValueError, 'unassigned captures'):
            snapshot(self.captures, 'session')

    def test_existing_evidence_is_never_overwritten(self):
        self.capture()
        stage = self.base / 'stage'
        write_stage(self.captures, 'session', stage)
        previous = (stage / 'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            write_stage(self.captures, 'session', stage)
        self.assertEqual(previous, (stage / 'manifest.json').read_bytes())

    def test_edit_during_snapshot_rejected(self):
        folder = self.capture()
        from stage_reviewed_training import target as real_target
        def changed(annotation, dimensions):
            result = real_target(annotation, dimensions)
            self.edit_annotations(folder, lambda annotations: annotations['frame_1.jpg'].update(reviewed_at='changed'))
            return result
        with patch('stage_reviewed_training.target', changed):
            with self.assertRaisesRegex(ValueError, 'changed during staging'):
                snapshot(self.captures, 'session')


if __name__ == '__main__':
    unittest.main()
