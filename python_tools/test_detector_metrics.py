import unittest
from improve_detector import annotation_status
from score_detector_predictions import score


class DetectorMetricsTest(unittest.TestCase):
    def test_unknown_is_not_negative(self):
        self.assertEqual(annotation_status(None), 'unknown')
        self.assertEqual(annotation_status({'x': -1, 'y': 508}), 'unknown')
        self.assertEqual(annotation_status({'x': -1, 'y': -1}), 'negative')
        self.assertEqual(annotation_status({'x': 0, 'y': 0}), 'positive')

    def test_localization_miss_counts_fp_and_fn(self):
        rows = [dict(x=20, y=0, gt_x=0, gt_y=0, confidence=.9, status='positive'),
                dict(x=0, y=0, gt_x=0, gt_y=0, confidence=.9, status='positive'),
                dict(x=0, y=0, gt_x=-1, gt_y=-1, confidence=.9, status='negative'),
                dict(x=0, y=0, gt_x=-1, gt_y=-1, confidence=.1, status='negative')]
        result = score(rows, .25)
        self.assertEqual([result[k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 2, 1, 1])
        self.assertAlmostEqual(result['precision'], 1/3)
        self.assertAlmostEqual(result['recall'], .5)


if __name__ == '__main__':
    unittest.main()
