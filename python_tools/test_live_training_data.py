import json
import tempfile
import unittest
from pathlib import Path
from prepare_live_training import load_reviewed, validate_annotation


class LiveDataTests(unittest.TestCase):
    def test_unknown_is_not_negative(self):
        self.assertIsNone(validate_annotation({}))
        self.assertIsNone(validate_annotation({'reviewed':False,'ball':[]}))
        self.assertEqual(validate_annotation({'reviewed':True,'ball':[],'person':[],'table':[]})['ball'],[])
        with self.assertRaises(ValueError):validate_annotation({'reviewed':True,'ball':[]})

    def test_box_validation(self):
        base={'reviewed':True,'ball':[[.1,.2,.03,.04]],'person':[],'table':[]}
        self.assertEqual(len(validate_annotation(base)['ball']),1)
        for box in ([.99,.1,.1,.1],[0,0,0,.1],[float('nan'),0,.1,.1]):
            with self.assertRaises(ValueError):validate_annotation({**base,'ball':[box]})

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
