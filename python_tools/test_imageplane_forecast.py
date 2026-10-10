"""Focused safeguards for paired exact-frame image-plane forecast evaluation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_imageplane_forecast import evaluate


def pixel(t):
    return [10 + 20 * t + 30 * t * t, 40 - 10 * t + 4 * t * t]


def fixture(count=12, fps=100.):
    model = {'backend': 'blurball', 'lookahead_frames': 2, 'temporal_policy': 'all_windows_two_frame_lookahead', 'experimental': True}
    manifest = {'total_frames': count, 'source_fps': fps, 'ball_model': model,
                'source_sha256': 'fixture_source', 'ball_model_sha256': 'fixture_model', 'runtime_sha256': 'fixture_runtime'}
    rows = []
    for frame in range(1, count + 1):
        time = (frame - 1) / fps
        support = list(range(max(1, frame - 8), frame + 1))
        ready = len(support) >= 5
        scale = (support[-1] - support[0]) / fps
        coefficients = [[pixel(time)[0], (20 + 60 * time) * scale, 30 * scale * scale],
                        [pixel(time)[1], (-10 + 8 * time) * scale, 4 * scale * scale]]
        flight = {'status': 'EXPERIMENTAL_FORECAST' if ready else 'UNAVAILABLE',
                  'reason': 'OBSERVED_FLIGHT_FIT' if ready else 'INSUFFICIENT_CONTINUOUS_OBSERVATIONS',
                  'method': 'recent_observations_quadratic_v1', 'coordinate_basis': 'source_pixels',
                  'reference_frame': frame, 'reference_timestamp': time, 'segment_id': 0, 'support_frames': support,
                  'maximum_horizon_s': .05, 'fit_origin': time if ready else None,
                  'fit_time_scale': scale if ready else None, 'fit_coefficients': coefficients if ready else None,
                  'fit_rmse_px': 0. if ready else None, 'uncertainty': None,
                  'points': [{'timestamp': time, 'pixel': pixel(time)}] if ready else []}
        rows.append({'frame_id': frame, 'source_time': time, 'ball_model': copy.deepcopy(model),
                     'ball': {'x': 0., 'y': 0.}, 'ball_blur': {'center_px': pixel(time)}, 'ball_flight': flight})
    labels = [{'frame': frame - 1, 'visible': 1, 'pixel': pixel((frame - 1) / fps)} for frame in range(1, count + 1)]
    return manifest, rows, labels


class ImagePlaneForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest, self.rows, self.labels = fixture()

    def tearDown(self):
        self.temp.cleanup()

    def run_evaluation(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        journal = self.root / 'frames.jsonl'
        journal.write_text(''.join(json.dumps(row) + '\n' for row in self.rows), encoding='utf-8')
        labels = self.root / 'labels.csv'
        labels.write_text('Frame,Visibility,X,Y\n' + ''.join(f"{row['frame']},{row['visible']},{row['pixel'][0]},{row['pixel'][1]}\n" for row in self.labels), encoding='utf-8')
        return evaluate(journal, labels)

    def test_exact_polynomial_and_paired_same_subset(self):
        result = self.run_evaluation()
        stats = result['paired_all_horizons']
        self.assertLess(stats['quadratic']['rmse_euclidean_px'], 1e-10)
        self.assertGreater(stats['last_position']['rmse_euclidean_px'], .1)
        self.assertGreater(stats['constant_velocity']['rmse_euclidean_px'], .0001)
        self.assertEqual({value['samples'] for value in stats.values()}, {13})
        for offset in (1, 2):
            account = result['horizons_source_frames'][str(offset)]['accounting']
            self.assertEqual(account['visible_target_opportunities'] + account['target_outside_recording'], len(self.rows))
            self.assertEqual(account['paired_scored'] + account['fit_unavailable'], account['visible_target_opportunities'])
        self.assertFalse(result['causal_future_prediction_benchmark'])
        self.assertIn('NOT a causal', result['future_support_warning'])
        self.assertEqual(result['detector_lookahead_frames'], 2)
        self.assertFalse(result['promotion_approved'])

    def test_future_fit_support_rejected_even_for_abstention(self):
        self.rows[0]['ball_flight']['support_frames'] = [2]
        with self.assertRaisesRegex(ValueError, 'Future observation'):
            self.run_evaluation()

    def test_causal_policy_retains_exact_paired_scoring(self):
        causal_model = {'backend': 'blurball', 'lookahead_frames': 0,
                        'temporal_policy': 'causal_channel_2', 'experimental': True}
        self.manifest['ball_model'] = copy.deepcopy(causal_model)
        for row in self.rows:
            row['ball_model'] = copy.deepcopy(causal_model)
        result = self.run_evaluation()
        self.assertTrue(result['causal_future_prediction_benchmark'])
        self.assertEqual(result['detector_lookahead_frames'], 0)
        self.assertIsNone(result['future_support_warning'])
        self.assertEqual(result['paired_all_horizons']['quadratic']['samples'], 13)
        self.assertLess(result['paired_all_horizons']['quadratic']['rmse_euclidean_px'], 1e-10)
        self.assertFalse(result['promotion_approved'])

    def test_absent_reference_excluded_without_imagined_position(self):
        self.labels[7] = {'frame': 7, 'visible': 0, 'pixel': [-999, -999]}
        result = self.run_evaluation()
        self.assertFalse(any(sample['target_frame'] == 8 for sample in result['samples']))
        self.assertEqual(result['paired_all_horizons']['quadratic']['samples'], 11)
        for horizon in result['horizons_source_frames'].values():
            self.assertEqual(horizon['accounting']['target_absent_no_location_scored'], 1)
        self.assertIsNone(result['contacts']['contact_subset_accuracy'])

    def test_missing_rows_and_missing_labels_rejected(self):
        del self.rows[4]
        with self.assertRaisesRegex(ValueError, 'every consecutive source frame'):
            self.run_evaluation()
        self.manifest, self.rows, self.labels = fixture()
        del self.labels[4]
        with self.assertRaisesRegex(ValueError, 'every exact zero-based source frame'):
            self.run_evaluation()

    def test_wrong_origin_and_nonobserved_support_rejected(self):
        self.rows[5]['ball_flight']['fit_origin'] += .001
        with self.assertRaisesRegex(ValueError, 'last past observation'):
            self.run_evaluation()
        self.manifest, self.rows, self.labels = fixture()
        self.rows[0]['ball'] = None
        with self.assertRaisesRegex(ValueError, 'explicit observed frame'):
            self.run_evaluation()

    def test_cross_segment_support_rejected(self):
        self.rows[5]['ball_flight']['segment_id'] = 1
        with self.assertRaisesRegex(ValueError, 'segment boundary'):
            self.run_evaluation()

    def test_missing_observation_horizon_is_bounded(self):
        original = self.rows[4]['ball_flight']
        for row in self.rows[5:]:
            row['ball'] = None
            flight = row['ball_flight']
            flight['support_frames'] = original['support_frames'][:]
            for key in ('fit_origin', 'fit_time_scale', 'fit_coefficients', 'fit_rmse_px'):
                flight[key] = copy.deepcopy(original[key])
            # No forecast point is supplied beyond the last real observation.
            flight['points'] = []
            if row['source_time'] > original['fit_origin'] + .05 + 1e-9:
                flight['status'] = 'UNAVAILABLE'
                flight['reason'] = 'OBSERVATION_EXPIRED'
                flight['support_frames'] = []
                for key in ('fit_origin', 'fit_time_scale', 'fit_coefficients', 'fit_rmse_px'):
                    flight[key] = None
        result = self.run_evaluation()
        self.assertGreater(result['horizons_source_frames']['2']['accounting']['last_observation_horizon_expired'], 0)
        self.assertTrue(all(sample['elapsed_from_last_observation_s'] <= .05 + 1e-9 for sample in result['samples']))

    def test_unknown_visibility_and_policy_contradiction_rejected(self):
        self.labels[5]['visible'] = 2
        with self.assertRaisesRegex(ValueError, 'Unknown visibility'):
            self.run_evaluation()
        self.manifest, self.rows, self.labels = fixture()
        self.manifest['ball_model']['temporal_policy'] = 'causal_channel_2'
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            self.run_evaluation()

    def test_filtered_experimental_inputs_need_separate_provenance(self):
        self.manifest['experimental_intelligence'] = True
        with self.assertRaisesRegex(ValueError, 'filtered fit inputs need separate provenance'):
            self.run_evaluation()


if __name__ == '__main__':
    unittest.main()
