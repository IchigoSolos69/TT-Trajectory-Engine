import unittest
from score_trajectory_uncertainty import score


class CoverageTests(unittest.TestCase):
    def test_states_coverage_and_unknowns(self):
        def sample(frame, state='MEASURED', xy=(0, 0)):
            return {'frame_id': frame, 'state': state, 'pixel': xy,
                    'pixel_covariance': [1, 0, 0, 1]}
        samples = [sample(1), sample(2, 'PREDICTED', (3, 0)), sample(3, 'LOST'),
                   sample(4), sample(5), sample(6)]
        labels = {'1': {'x': 1, 'y': 1}, '2': {'x': 0, 'y': 0},
                  '3': {'x': 5, 'y': 5}, '4': {'x': -1, 'y': -1},
                  '5': {'x': 0, 'y': -1}}
        report = score(samples, labels)
        self.assertEqual(report['by_state']['MEASURED']['model_95_region_coverage'], 1)
        self.assertEqual(report['by_state']['PREDICTED']['model_95_region_coverage'], 0)
        self.assertEqual(report['known_positive_lost'], 1)
        self.assertEqual(report['known_absent_with_track'], 1)
        self.assertEqual(report['unknown_excluded'], 2)

    def test_correlated_covariance(self):
        samples = [{'frame_id': 1, 'state': 'REACQUIRED', 'pixel': [2, 2],
                    'pixel_covariance': [1, .9, .9, 1]}]
        report = score(samples, {'1': {'x': 0, 'y': 0}})
        self.assertEqual(report['by_state']['REACQUIRED']['model_95_region_coverage'], 1)
        samples[0]['pixel'] = [2, -2]
        report = score(samples, {'1': {'x': 0, 'y': 0}})
        self.assertEqual(report['by_state']['REACQUIRED']['model_95_region_coverage'], 0)


if __name__ == '__main__':
    unittest.main()
