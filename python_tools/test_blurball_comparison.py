"""Geometry and evaluation safeguards for the isolated BlurBall challenger."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate_blurball_candidate import components, select
from score_blurball_comparison import score


class BlurBallComparisonTests(unittest.TestCase):
    def test_blur_endpoints_and_weighted_centroid_in_source_pixels(self):
        heat = np.zeros((10, 10), np.float32)
        heat[4, 3:7] = [.8, .9, .9, .8]
        # A heatmap pixel represents 4 source pixels; source also has an offset.
        inverse = np.array([[4., 0., 10.], [0., 4., 20.]])
        blob, = components(heat, inverse, .7, 100, 100)
        np.testing.assert_allclose(blob['center_px'], [28., 36.], atol=1e-5)
        self.assertAlmostEqual(blob['blur_extent_source_px'], 12.)
        self.assertGreater(blob['component_mass'], 1.)
        self.assertIsNone(blob['confidence'])
        self.assertAlmostEqual(blob['blur_axis_deg'] % 180, 0.)

    def test_tracking_gate_does_not_fabricate_or_renew_missing_observations(self):
        previous = {'center_px': [10., 10.], 'component_mass': 3.}
        far = {'center_px': [310., 10.], 'component_mass': 10.}
        self.assertIsNone(select([far], previous, 300.))  # Strict upstream bound.
        self.assertIsNone(select([], previous, 300.))
        self.assertEqual(select([far], None, 300.), far)  # Reacquire without stale anchor.

    def test_wrong_location_counts_as_missed_reference_and_false_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / 'labels.csv'
            labels.write_text('Frame,Visibility,X,Y\n0,1,10,10\n1,1,20,20\n2,0,0,0\n')
            rows = [{'frame_id': fid, **{key: candidate for key in
                    ['yolo', 'blurball_offline', 'blurball_causal']}}
                    for fid, candidate in [(1, {'center_px': [10, 10]}),
                                           (2, {'center_px': [24, 20]}), (3, None)]]
            predictions = root / 'predictions.jsonl'
            predictions.write_text('\n'.join(json.dumps(row) for row in rows))
            result = score(predictions, labels)['full_recording']['blurball_offline']
            self.assertEqual(result['tp'], 1)
            self.assertEqual(result['wrong_location_predictions'], 1)
            self.assertEqual(result['recall'], .5)
            self.assertEqual(result['upstream_recall_convention'], 1.)
            labels.write_text('Frame,Visibility,X,Y\n0,2,10,10\n')
            with self.assertRaisesRegex(ValueError, 'Unknown/occluded'):
                score(predictions, labels)


if __name__ == '__main__':
    unittest.main()
