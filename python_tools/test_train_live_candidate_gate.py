"""Exercise candidate CLI audit gates without importing models or starting training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CandidateAuditTests(unittest.TestCase):
    def test_ball_only_v2_counts_and_safety_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prepared=root/'prepared';prepared.mkdir();output=root/'candidate'
            audit={'schema_version':2,'contains_test_in_training':False,
                   'reviewed_counts_by_kind':{'ball':{'train':30,'val':30,'test':30}},
                   'frames':[{'kind':'ball','split':split,'objects':{'ball':int(i<15)}} for split in ('train','val','test') for i in range(30)]}
            def run(kind='ball'):
                (prepared/'audit.json').write_text(json.dumps(audit))
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('train_live_candidates.py')),str(prepared),str(output),'--kind',kind,'--epochs','3'],capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(output.exists(),'Gate checks must not start training')
                return result.stderr
            self.assertIn('at least four epochs',run(),'Valid ball-only audits must reach the epoch gate')
            self.assertIn('30 reviewed frames',run('scene'),'Unprepared scene data must be rejected')
            audit['contains_test_in_training']=True
            self.assertIn('30 reviewed frames',run())
            audit['contains_test_in_training']=False
            for frame in audit['frames']:frame['objects']['ball']=0
            self.assertIn('ball-positive',run())


if __name__=='__main__':unittest.main()
