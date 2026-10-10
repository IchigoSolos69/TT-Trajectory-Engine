"""Synthetic interval review tests; no source videos or real labels changed."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from review_absence_provenance import checked_interval_absence
from export_reviewed_streaks import audit, export
from test_reviewed_streaks import StreakTests
from prepare_live_training import load_reviewed

ROOT = Path(__file__).resolve().parents[1]


class IntervalAbsenceTests(unittest.TestCase):
    def fixture(self, root):
        captures, splits = StreakTests().fixture(root)
        for split in ('train', 'val', 'test'):
            folder = captures / split
            capture = json.loads((folder/'capture.json').read_text())
            capture['original_sha256'] = hashlib.sha256(split.encode()).hexdigest()
            (folder/'capture.json').write_text(json.dumps(capture))
            old = (folder/'annotations.json').read_bytes()
            previous = hashlib.sha256(old).hexdigest()
            history = folder/'annotation_history'; history.mkdir()
            (history/(previous+'.json')).write_bytes(old)
            annotations = json.loads(old)
            index_hash = hashlib.sha256((folder/'frames.jsonl').read_bytes()).hexdigest()
            for filename, annotation in annotations.items():
                annotation['geometry'] = {'visibility': 'absent', 'blur_streak': None}
                annotation['reviewed_at'] = '2026-10-09T12:00:00.000Z'
                annotation['ball_absence_review'] = {
                    'method': 'manual_interval', 'review_id': 'synthetic_'+split,
                    'reviewed_at': annotation['reviewed_at'], 'session_id': split,
                    'original_sha256': capture['original_sha256'], 'index_sha256': index_hash,
                    'image_sha256': hashlib.sha256((folder/filename).read_bytes()).hexdigest(),
                    'previous_annotation_revision': previous,
                    'start_source_frame': 1, 'end_source_frame': 3, 'sampled_frames': 3,
                }
            (folder/'annotations.json').write_text(json.dumps(annotations))
        return captures, splits

    def test_provenance_survives_temporal_handoff_and_ball_only_yolo(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
            root=Path(temporary); captures,splits=self.fixture(root)
            result=audit(captures,splits)
            self.assertTrue(result['ready_for_training'])
            for window in result['windows']:
                for frame in window['frames']:
                    self.assertEqual(frame['target']['visibility'],'absent')
                    self.assertEqual(frame['target']['interval_review']['image_sha256'],frame['sha256'])
            export(captures,splits,root/'export')
            rows,skipped=load_reviewed(captures,json.loads(splits.read_text()),('ball',))
            self.assertEqual(skipped,0)
            self.assertTrue(all(labels=={'ball':[]} for group in rows.values() for _,labels,_,_ in group))
            with self.assertRaisesRegex(ValueError,'No reviewed examples'):
                load_reviewed(captures,json.loads(splits.read_text()),('person','table'))

    def test_changed_evidence_is_not_accepted_as_training_absence(self):
        for field in ('session_id','original_sha256','image_sha256','index_sha256','previous_annotation_revision','start_source_frame','reviewed_at'):
            with self.subTest(field=field),tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
                captures,splits=self.fixture(Path(temporary))
                file=captures/'train/annotations.json'; labels=json.loads(file.read_text())
                provenance=labels['frame_1.jpg']['ball_absence_review']
                provenance[field]=2 if field=='start_source_frame' else ('b'*64 if 'sha256' in field or field=='previous_annotation_revision' else 'changed')
                file.write_text(json.dumps(labels))
                with self.assertRaisesRegex(ValueError,'Interval absence'):
                    audit(captures,splits)
                with self.assertRaisesRegex(ValueError,'Interval absence'):
                    load_reviewed(captures,json.loads(splits.read_text()),('ball',))

    def test_old_individual_absence_labels_remain_compatible(self):
        self.assertIsNone(checked_interval_absence({'ball':[]}, {}, {}, '', '', Path('unused')))


if __name__=='__main__': unittest.main()
