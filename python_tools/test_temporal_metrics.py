import json
from pathlib import Path
import tempfile
import unittest
from score_temporal_pipeline import evaluate, metrics


class RecoveryScoringTests(unittest.TestCase):
    def test_wrong_positive_counts_both_errors(self):
        result = metrics([(True, True, 16), (True, True, 15), (False, True, float('inf')),
                          (False, False, float('inf'))])
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['tn']), (1, 2, 1, 1))

    def test_unknown_excluded_and_gap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'labels.json').write_text(json.dumps({'1': {'x': 20, 'y': 30},
                '2': {'x': -1, 'y': -1}, '3': {'x': -1, 'y': 30}}))
            (root/'engine').write_bytes(b'fixture')
            header = 'frame,baseline_x,baseline_y,baseline_score,recovery_x,recovery_y,recovery_score,recovery_accepted,missed_frames,roi_x,roi_y,gpu_ms\n'
            rows = '1,-1,-1,.1,20,30,.96,1,1,0,0,.2\n2,-1,-1,.1,-1,-1,0,0,0,0,0,0\n3,50,50,.8,-1,-1,0,0,0,0,0,0\n'
            (root/'replay.csv').write_text(header+rows)
            result = evaluate(root/'replay.csv', root/'labels.json', root/'engine')
            self.assertEqual(result['unknown_excluded'], 1)
            self.assertEqual(result['hybrid']['tp'], 1)
            self.assertEqual(result['hybrid']['tn'], 1)
            self.assertEqual(result['baseline']['fn'], 1)
            (root/'replay.csv').write_text(header+rows.replace('2,-1,-1,.1,-1,-1,0,0,0,0,0,0\n', ''))
            with self.assertRaises(ValueError):
                evaluate(root/'replay.csv', root/'labels.json', root/'engine')


if __name__ == '__main__':
    unittest.main()
