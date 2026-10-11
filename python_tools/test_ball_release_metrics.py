import copy
import unittest

from ball_release_metrics import evaluate_release, score_frames


def row(fid=1, *, visible=True, position=(100., 100.), size=(1920, 1080), state="measured", domain="iphone"):
    return dict(source_id="clip", frame_id=fid, timestamp_s=(fid-1)/60,
                source_size=list(size), domain=domain, rally_id="rally1",
                frame_duration_s=1/60,
                annotation=dict(reviewed=True, visibility="visible" if visible else "absent",
                                center_px=[100., 100.] if visible else None, tags=[]),
                prediction=None if position is None else dict(center_px=list(position), state=state, score=.7, track_id=1))


class BallReleaseMetricsTests(unittest.TestCase):
    def test_wrong_localization_counts_as_false_positive_and_miss(self):
        result = score_frames([row(position=(200, 100))])
        self.assertEqual((result["tp"], result["false_positives"], result["false_negatives"]), (0, 1, 1))

    def test_v1_abstention_cannot_hide_recall(self):
        rows = [row(i, position=(100, 100) if i <= 5 else None) for i in range(1, 271)]
        result = score_frames(rows)
        self.assertEqual(result["precision"], 1)
        self.assertAlmostEqual(result["recall"], 5/270)

    def test_1080_normalization_and_strict_boundary(self):
        self.assertEqual(score_frames([row(position=(103, 100), size=(1280, 720))])["tp"], 0)
        self.assertEqual(score_frames([row(position=(102, 100), size=(1280, 720))])["tp"], 1)
        self.assertEqual(score_frames([row(position=(104, 100))])["tp"], 0)

    def test_predictions_are_not_measured_true_positives(self):
        result = score_frames([row(state="predicted")])
        self.assertEqual((result["tp"], result["false_negatives"], result["predicted_frames"]), (0, 1, 1))
        self.assertIsNone(result["precision"])

    def test_unknown_is_neither_negative_nor_visible(self):
        sample = row()
        sample["annotation"]["reviewed"] = False
        result = score_frames([sample])
        self.assertEqual(result["unknown_reference_frames"], 1)
        self.assertEqual(result["false_positives"], 0)
        self.assertIsNone(result["recall"])

    def test_negative_rate_does_not_invent_precision(self):
        result = score_frames([row(1, visible=False), row(2, visible=False, position=None)])
        self.assertEqual(result["sampled_negative_false_positive_rate"], .5)
        self.assertEqual(result["precision"], 0)
        self.assertIsNone(result["recall"])

    def test_sparse_negatives_cannot_supply_duration_or_episode_rate(self):
        result = score_frames([row(1, visible=False), row(100, visible=False)])
        self.assertIsNone(result["verified_absence_seconds"])
        self.assertIsNone(result["false_track_episodes_per_minute"])

    def test_continuous_absence_duration_and_track_episodes(self):
        rows = [row(i, visible=False, position=(100, 100) if i in (1, 2, 5) else None)
                for i in range(1, 61)]
        result = score_frames(rows)
        self.assertAlmostEqual(result["verified_absence_seconds"], 1)
        self.assertEqual(result["false_track_episodes"], 2)

    def test_missing_track_identity_cannot_merge_false_episodes(self):
        rows = [row(i, visible=False) for i in range(1, 61)]
        for sample in rows:
            sample["prediction"].pop("track_id")
        result = score_frames(rows)
        self.assertAlmostEqual(result["verified_absence_seconds"], 1)
        self.assertIsNone(result["false_track_episodes_per_minute"])
        self.assertIn("track identity", result["episode_rate_reason"])

    def test_declared_duration_cannot_inflate_source_pts_coverage(self):
        rows = [row(i, visible=False, position=None) for i in range(1, 61)]
        for sample in rows:
            sample["frame_duration_s"] = .1
        result = score_frames(rows)
        self.assertIsNone(result["verified_absence_seconds"])
        self.assertFalse(result["absence_coverage_complete"])

    def test_metadata_rounding_tolerance_does_not_add_verified_time(self):
        rows = [row(i, visible=False, position=None) for i in range(1, 61)]
        for sample in rows:
            sample["frame_duration_s"] += .0009
        result = score_frames(rows)
        self.assertTrue(result["absence_coverage_complete"])
        self.assertAlmostEqual(result["verified_absence_seconds"], 1)

    def test_actual_eof_zero_tail_preserves_conservative_pts_coverage(self):
        rows = [row(i, visible=False, position=None) for i in range(1, 61)]
        rows[-1]["frame_duration_s"] = 0
        result = score_frames(rows)
        self.assertTrue(result["absence_coverage_complete"])
        self.assertAlmostEqual(result["verified_absence_seconds"], 59/60)
        interior = copy.deepcopy(rows)
        interior[20]["frame_duration_s"] = 0
        self.assertFalse(score_frames(interior)["absence_coverage_complete"])

    def test_reference_mismatch_rejects_regression_comparison(self):
        candidate, baseline = row(), row()
        baseline["annotation"]["center_px"] = [101, 100]
        with self.assertRaisesRegex(ValueError, "identical"):
            evaluate_release([candidate], [baseline])

    def test_no_camera_data_cannot_pass_and_validation_cannot_promote(self):
        result = evaluate_release([row()], [row()])
        self.assertFalse(result["accuracy_gates_passed"])
        self.assertFalse(result["live_gates_passed"])
        self.assertFalse(result["promotion_approved"])
        self.assertIn("pc_camera:visible_count", result["failed_checks"])

    def test_nonfinite_duplicate_and_source_geometry_rejected(self):
        invalid = row(position=(float("nan"), 100))
        with self.assertRaises(ValueError):
            score_frames([invalid])
        with self.assertRaises(ValueError):
            score_frames([row(), row()])
        with self.assertRaises(ValueError):
            score_frames([row(1), row(2, size=(1280, 720))])

    def test_report_domains_cannot_be_swamped_by_match_count(self):
        samples = [row(i, domain="match") for i in range(1, 101)]
        phone = row(position=(200, 100))
        phone["source_id"] = "phone"
        result = evaluate_release(samples+[phone], copy.deepcopy(samples+[phone]))
        self.assertEqual(result["domains"]["match"]["recall"], 1)
        self.assertEqual(result["domains"]["iphone"]["recall"], 0)
        self.assertFalse(result["checks"]["iphone:recall"])

    def test_offline_timing_cannot_be_live_evidence(self):
        evidence = {d: dict(analyzed_fps=120, supplied_fps=120, capture_frames_dropped=0,
                           p95_capture_to_display_ms=10, duration_s=1800, scope="offline_gpu")
                    for d in ("iphone", "pc_camera")}
        result = evaluate_release([row()], [row()], evidence)
        self.assertFalse(result["live_gates_passed"])
        self.assertFalse(result["live_checks"]["iphone:timing_scope"])

    def test_recorded_iphone_does_not_require_fabricated_live_measurements(self):
        evidence = {"pc_camera": dict(analyzed_fps=60, supplied_fps=60,
                    capture_frames_dropped=0, p95_capture_to_display_ms=80,
                    duration_s=1800, scope="single_session_live_capture_to_display")}
        result = evaluate_release([row()], [row()], evidence)
        self.assertTrue(result["live_gates_passed"])
        self.assertEqual(result["live_domains_required"], ["pc_camera"])
        self.assertFalse(result["release_candidate_eligible"])

    def test_subset_gates_require_counts_and_report_without_new_accuracy_targets(self):
        result = evaluate_release([row()], [row()])
        self.assertIn("iphone:blur_count", result["checks"])
        self.assertNotIn("iphone:blur_precision", result["checks"])
        self.assertIn("precision", result["domains"]["iphone"]["subsets"]["blur"])


if __name__ == "__main__":
    unittest.main()
