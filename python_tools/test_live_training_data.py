import json
import tempfile
import unittest
from pathlib import Path
import prepare_live_training as preparation
from prepare_live_training import load_reviewed, validate_annotation, check_capture_session


class LiveDataTests(unittest.TestCase):
    def test_phone_session_leakage(self):
        sessions = {}
        manifest = {'source': 'recorded_camera', 'session_id': 'iphone_session_1'}
        check_capture_session(manifest, 'train', sessions)
        check_capture_session(manifest, 'train', sessions)
        with self.assertRaisesRegex(ValueError, 'session overlaps'):
            check_capture_session(manifest, 'test', sessions)
        with self.assertRaisesRegex(ValueError, 'session ID'):
            check_capture_session({'source': 'recorded_camera'}, 'train', {})
        with self.assertRaisesRegex(ValueError, 'real camera'):
            check_capture_session({'source': 'dataset'}, 'train', {})

    def test_unknown_is_not_negative(self):
        self.assertIsNone(validate_annotation({}))
        self.assertIsNone(validate_annotation({'reviewed':False,'ball':[]}))
        self.assertEqual(validate_annotation({'reviewed':True,'ball':[],'person':[],'table':[]})['ball'],[])
        for visibility in ('occluded','partial'):
            self.assertIsNone(validate_annotation({'reviewed':True,'ball':[],'person':[],'table':[], 'geometry':{'visibility':visibility}}))
        with self.assertRaises(ValueError):validate_annotation({'reviewed':True,'ball':[]})

    def test_box_validation(self):
        base={'reviewed':True,'ball':[[.1,.2,.03,.04]],'person':[],'table':[]}
        self.assertEqual(len(validate_annotation(base)['ball']),1)
        for box in ([.99,.1,.1,.1],[0,0,0,.1],[float('nan'),0,.1,.1]):
            with self.assertRaises(ValueError):validate_annotation({**base,'ball':[box]})

    def test_class_scoped_review_and_conflicting_visibility(self):
        partial={'reviewed':True,'reviewed_classes':{'ball':True,'person':False,'table':False},
                 'ball':[[.1,.2,.03,.04]],'person':[],'table':[], 'geometry':{'visibility':'visible'}}
        self.assertEqual(validate_annotation(partial,('ball',))['ball'],partial['ball'])
        self.assertIsNone(validate_annotation(partial,('person','table')))
        self.assertIsNone(validate_annotation(partial))
        self.assertIsNone(validate_annotation({**partial,'ball':[]},('ball',)))
        self.assertIsNone(validate_annotation({**partial,'ball':[],'geometry':{'visibility':'unknown'}},('ball',)))
        negative={**partial,'ball':[],'geometry':{'visibility':'absent'}}
        self.assertEqual(validate_annotation(negative,('ball',)),{'ball':[]})
        self.assertIsNone(validate_annotation({**negative,'geometry':{'visibility':'occluded'}},('ball',)))
        # The existing visible-without-box reviews must not silently become negatives.
        self.assertIsNone(validate_annotation({'reviewed':True,'ball':[],'person':[],'table':[],'geometry':{'visibility':'visible'}},('ball',)))
        self.assertIsNone(validate_annotation({'reviewed':True,'ball':[],'person':[],'table':[],'geometry':{'visibility':'unknown'}},('ball',)))
        scene={**negative,'reviewed_classes':{'ball':False,'person':True,'table':True},'geometry':{'visibility':'occluded'}}
        self.assertEqual(validate_annotation(scene,('person','table')),{'person':[],'table':[]})

    def test_ball_only_preparation_leaves_scene_unprepared(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);captures=root/'captures';captures.mkdir();splits={key:[key] for key in ('train','val','test')}
            split_file=root/'splits.json';split_file.write_text(json.dumps(splits))
            for split in splits:
                folder=captures/split;folder.mkdir()
                (folder/'capture.json').write_text(json.dumps({'source':'recorded_camera','session_id':split}))
                (folder/'frame_1.jpg').write_bytes(split.encode())
                (folder/'frames.jsonl').write_text(json.dumps({'file':'frame_1.jpg'})+'\n')
                labels={'reviewed':True,'reviewed_classes':{'ball':True,'person':False,'table':False},'ball':[[.1,.1,.01,.01]],'person':[],'table':[],'geometry':{'visibility':'visible'}}
                (folder/'annotations.json').write_text(json.dumps({'frame_1.jpg':labels}))
            matches=root/'artifacts/multigame';matches.mkdir(parents=True)
            for split in ('train','val'):
                image=root/f'game_{split}.jpg';image.write_bytes(split.encode())
                (matches/f'{split}.txt').write_text(str(image)+'\n')
            original_root=preparation.ROOT
            try:
                preparation.ROOT=root;output=root/'prepared'
                preparation.prepare(captures,split_file,output,kind='ball')
                audit=json.loads((output/'audit.json').read_text())
                self.assertEqual(audit['prepared_kinds'],['ball'])
                self.assertEqual(audit['reviewed_counts_by_kind']['ball'],{'train':1,'val':1,'test':1})
                self.assertFalse((output/'scene').exists())
                self.assertTrue((output/'ball/labels/train/train_frame_1.txt').read_text().startswith('0 '))
                self.assertTrue(all(row['kind']=='ball' and set(row['objects'])=={'ball'} for row in audit['frames']))
            finally:preparation.ROOT=original_root

    def test_split_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for clip in ('a','b','c'):
                folder=root/clip;folder.mkdir()
                (folder/'capture.json').write_text(json.dumps({'source':'live'}))
                (folder/'frames.jsonl').write_text(json.dumps({'file':'frame_1.jpg'})+'\n')
                (folder/'frame_1.jpg').write_bytes(clip.encode())
                (folder/'annotations.json').write_text(json.dumps({'frame_1.jpg':{'reviewed':True,'ball':[],'person':[],'table':[]}}))
            splits={'train':['a'],'val':['b'],'test':['c']}
            rows,skipped=load_reviewed(root,splits)
            self.assertEqual([len(v) for v in rows.values()],[1,1,1]);self.assertEqual(skipped,0)
            with self.assertRaises(ValueError):load_reviewed(root,{**splits,'test':['a']})
            (root/'c/frame_1.jpg').write_bytes(b'a')
            with self.assertRaises(ValueError):load_reviewed(root,splits)


if __name__=='__main__':unittest.main()
