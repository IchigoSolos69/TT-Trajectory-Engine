import json
import tempfile
import unittest
from pathlib import Path
from audit_temporal_captures import audit


class BurstAuditTests(unittest.TestCase):
    def test_complete_unknown_gap_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);splits={key:[key] for key in ('train','val','test')}
            splitfile=root/'splits.json';splitfile.write_text(json.dumps(splits))
            for key in splits:
                directory=root/key;directory.mkdir();rows=[];labels={}
                for i in range(5):
                    file=f'frame_{i+1}.jpg';(directory/file).write_bytes(f'{key}-{i}'.encode())
                    rows.append(dict(file=file,source_frame=i+1,burst_id=1,burst_index=i,burst_length=5,elapsed_ms=i*10))
                    labels[file]=dict(reviewed=True,ball=[],person=[],table=[])
                (directory/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                (directory/'annotations.json').write_text(json.dumps(labels))
            result=audit(root,splitfile)
            self.assertTrue(result['ready_for_training']);self.assertEqual(len(result['accepted']),3)
            (root/'val'/'annotations.json').write_text('{}')
            result=audit(root,splitfile);self.assertFalse(result['ready_for_training'])
            self.assertEqual(result['excluded'][0]['reason'],'UNREVIEWED_FRAME')
            rows[2]['source_frame']=99
            (root/'test'/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result=audit(root,splitfile)
            self.assertIn('SOURCE_FRAME_GAP',[r['reason'] for r in result['excluded']])
            (root/'test'/'frame_1.jpg').write_bytes((root/'train'/'frame_1.jpg').read_bytes())
            with self.assertRaisesRegex(ValueError,'Duplicate image'):
                audit(root,splitfile)


if __name__=='__main__':unittest.main()
