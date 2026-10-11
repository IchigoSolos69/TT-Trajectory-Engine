"""Independent v2 journal decisions and fail-closed provenance checks."""
import copy
from pathlib import Path
import tempfile
import unittest

from test_score_paired_ball_candidates import JournalFixture
from score_paired_ball_candidates import (cpp_canonical_json, primary_config,
                                         recompute_selection, score, sha256_bytes,
                                         validate_fusion_config)


class PrimaryScorerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture = JournalFixture(self.root)
        fixture = self.fixture
        fixture.config = primary_config()
        fixture.manifest['fusion'] = fixture.config
        fixture.manifest['fusion_sha256'] = sha256_bytes(cpp_canonical_json(fixture.config).encode())
        fixture.summary['fusion_sha256'] = fixture.manifest['fusion_sha256']
        # Explicit expectations, not derived from the implementation under test.
        decisions = [
            ('BLURBALL_CAUSAL_WARMUP', None),
            ('BLURBALL_CAUSAL_WARMUP', None),
            ('BLURBALL_PRIMARY_SPATIAL_AGREEMENT', ('blurball', 0)),
            ('BLURBALL_PRIMARY_YOLO_DISAGREEMENT', ('blurball', 0)),
            ('AMBIGUOUS_BLURBALL_CANDIDATES', None),
            ('BLURBALL_PRIMARY_AMBIGUOUS_YOLO', ('blurball', 0)),
            ('CONSERVATIVE_YOLO_ONLY_FALLBACK', ('yolo', 0)),
            ('BLURBALL_PRIMARY_NO_ELIGIBLE_YOLO', ('blurball', 0)),
            ('BLURBALL_PRIMARY_NO_ELIGIBLE_YOLO', ('blurball', 0)),
            ('BLURBALL_PRIMARY_NO_ELIGIBLE_YOLO', ('blurball', 0)),
            ('NO_ELIGIBLE_CANDIDATE', None),
            ('YOLO_ONLY_INSUFFICIENT_EVIDENCE', None),
            ('BLURBALL_PRIMARY_NO_ELIGIBLE_YOLO', ('blurball', 0)),
        ]
        for fid, (row, (reason, selected)) in enumerate(zip(fixture.rows, decisions), 1):
            row['fusion_sha256'] = fixture.manifest['fusion_sha256']
            value = row['experimental_selection']
            value['reason'] = reason
            value['selected'] = copy.deepcopy(row[selected[0] + '_candidates'][selected[1]]) if selected else None
            if fid == 6:
                value['yolo_support_index'] = value['blurball_support_index'] = 0
            if fid == 9:
                value['blurball_support_index'] = 0
        fixture.update_summary()
        fixture.write()

    def test_primary_journal_and_legacy_difference(self):
        report = score(self.root, self.fixture.labels)
        self.assertEqual(report['audit']['status'], 'PASS')
        self.assertEqual(report['hypothesis_counts']['experimental_selector']['selected_frames'], 8)
        self.assertEqual(report['selector_reason_histogram']['BLURBALL_PRIMARY_AMBIGUOUS_YOLO'], 1)
        self.assertFalse(report['promotion_approved'])

    def test_disagreement_and_yolo_ambiguity_keep_original_blurball(self):
        for fid in (4, 6):
            row = self.fixture.rows[fid - 1]
            value = recompute_selection(row, primary_config())
            self.assertEqual(value['selected'], row['blurball_candidates'][0])
            self.assertIsNone(value['confidence'])
        self.assertIsNone(recompute_selection(self.fixture.rows[5], primary_config())['agreement_distance_source_px'])

    def test_blurball_ambiguity_remains_abstention(self):
        value = recompute_selection(self.fixture.rows[4], primary_config())
        self.assertIsNone(value['selected'])
        self.assertEqual(value['reason'], 'AMBIGUOUS_BLURBALL_CANDIDATES')

    def test_disabled_yolo_fallback_never_creates_ball(self):
        value = recompute_selection(self.fixture.rows[6], primary_config(False))
        self.assertIsNone(value['selected'])
        self.assertEqual(value['reason'], 'YOLO_FALLBACK_DISABLED')

    def test_primary_accepts_peak_below_old_solo_threshold(self):
        value = recompute_selection(self.fixture.rows[12], primary_config())
        self.assertEqual(value['selected']['raw_score'], self.fixture.rows[12]['blurball_candidates'][0]['raw_score'])
        self.assertEqual(value['selected']['center_px'], [10., 10.])

    def test_changed_selection_is_rejected(self):
        self.fixture.rows[3]['experimental_selection']['selected'] = None
        self.fixture.update_summary()
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'differs from recomputed selector'):
            score(self.root)

    def test_revisions_cannot_silently_change_policy(self):
        self.fixture.manifest['fusion']['selection_policy'] = 'AVERAGE'
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'selection_policy'):
            score(self.root)

    def test_fallback_flag_must_be_boolean(self):
        config = primary_config()
        config['allow_yolo_fallback'] = 1
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            validate_fusion_config(config)

    def test_unknown_config_fields_and_revision_fail_closed(self):
        for field, value in (('blurball_solo_peak', .95), ('revision', 'future-unchecked-v3')):
            config = primary_config()
            config[field] = value
            with self.assertRaises(ValueError):
                validate_fusion_config(config)

    def test_invalid_threshold_order_and_native_precision_fail_closed(self):
        for field, value in (('yolo_fallback_score', .2), ('blurball_minimum_peak', .6),
                             ('yolo_evidence_score', .333333333)):
            config = primary_config()
            config[field] = value
            with self.assertRaises(ValueError):
                validate_fusion_config(config)


if __name__ == '__main__':
    unittest.main()
