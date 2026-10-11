import unittest

import numpy as np

from yolo_raw_diagnostics import candidate_diagnostic, canonical_input, raw_parity_metrics


class RawDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.contract = dict(source_size=[1920, 1080], letterbox=dict(scale=1, pad_left=0, pad_top=0))
        self.reference = dict(visibility="visible", center_px=[100, 100])

    def raw(self, centers, scores, width=10):
        return np.array([[*[p[0] for p in centers]], [*[p[1] for p in centers]],
                         [width]*len(scores), [width]*len(scores), scores], dtype=np.float32)[None]

    def test_oracle_cannot_replace_selected_candidate(self):
        result = candidate_diagnostic(self.raw([(100, 100), (800, 800)], [.3, .9]), self.contract, self.reference)
        self.assertEqual(result["failure_reason"], "RANKING_OR_ACCEPTANCE_THRESHOLD")
        self.assertEqual(result["selected"]["center_px"], [800, 800])
        self.assertTrue(result["oracle_post_nms_within_4px"])

    def test_pre_nms_oracle_exposes_suppression(self):
        result = candidate_diagnostic(self.raw([(100, 100), (110, 100)], [.3, .9], width=100), self.contract, self.reference)
        self.assertTrue(result["oracle_pre_nms_within_4px"])
        self.assertFalse(result["oracle_post_nms_within_4px"])
        self.assertEqual(result["failure_reason"], "NMS_SUPPRESSION")

    def test_low_score_candidate_is_not_accepted(self):
        result = candidate_diagnostic(self.raw([(100, 100)], [.2]), self.contract, self.reference)
        self.assertIsNone(result["selected"])
        self.assertTrue(result["oracle_post_nms_within_4px"])

    def test_outside_source_proposal_participates_in_native_nms(self):
        reference = dict(visibility="visible", center_px=[1, 100])
        result = candidate_diagnostic(self.raw([(-1, 100), (1, 100)], [.9, .5], width=100), self.contract, reference)
        self.assertEqual(len(result["pre_nms_candidates"]), 2)
        self.assertFalse(result["pre_nms_candidates"][0]["center_inside_source"])
        self.assertIsNone(result["selected"])
        self.assertTrue(result["oracle_pre_nms_within_4px"])
        self.assertEqual(result["failure_reason"], "NMS_SUPPRESSION")

    def test_score_parity_still_fails_without_coordinate_support(self):
        native = self.raw([(100, 100)], [0.])
        exact = self.raw([(100, 100)], [.09])
        result = raw_parity_metrics(native, exact, 1)
        self.assertIsNone(result["coordinate_parity_gate_passed"])
        self.assertFalse(result["score_parity_gate_passed"])
        self.assertFalse(result["parity_gate_passed"])

    def test_empty_support_cannot_claim_coordinate_parity(self):
        raw = self.raw([(100, 100)], [.01])
        result = raw_parity_metrics(raw, raw, 1)
        self.assertTrue(result["score_parity_gate_passed"])
        self.assertIsNone(result["parity_gate_passed"])

    def test_unknown_and_absent_are_not_false_visible_targets(self):
        raw = self.raw([(100, 100)], [.9])
        self.assertEqual(candidate_diagnostic(raw, self.contract, None)["failure_reason"], "REFERENCE_UNKNOWN")
        self.assertEqual(candidate_diagnostic(raw, self.contract, {"visibility": "absent", "center_px": None})["failure_reason"], "ABSENT_FALSE_POSITIVE")

    def test_unletterbox_uses_actual_dimensions(self):
        contract = dict(source_size=[1920, 1080], letterbox=dict(scale=.5, pad_left=0, pad_top=210))
        result = candidate_diagnostic(self.raw([(50, 260)], [.9]), contract, self.reference)
        self.assertEqual(result["selected"]["center_px"], [100, 100])

    def test_rgb_padding_and_normalization_independently_defined(self):
        bgr = np.zeros((3, 7, 3), np.uint8)
        bgr[:, :, 2] = 255
        result = canonical_input(bgr, [1, 3, 8, 8])
        self.assertEqual(result.shape, (1, 3, 8, 8))
        self.assertAlmostEqual(float(result[0, 0, 3, 3]), 1)
        self.assertEqual(float(result[0, 2, 3, 3]), 0)
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), 114/255)


if __name__ == "__main__":
    unittest.main()
