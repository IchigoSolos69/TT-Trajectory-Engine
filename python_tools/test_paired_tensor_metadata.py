"""Shape provenance migration must preserve historical 640 journal evidence."""
import copy
import tempfile
from pathlib import Path
import unittest

try:
    from .score_paired_ball_candidates import score, validate_manifest
    from .test_score_paired_ball_candidates import JournalFixture
except ImportError:
    from score_paired_ball_candidates import score, validate_manifest
    from test_score_paired_ball_candidates import JournalFixture


def contracts(size=640, anchors=8400, width=None):
    return {
        'yolo': {'input_shape': [1, 3, size, width or size], 'output_shape': [1, 5, anchors],
                 'classes': 1, 'anchors': anchors, 'io_dtype': 'float32',
                 'input_layout': 'NCHW', 'output_layout': 'CHANNEL_MAJOR',
                 'shape_source': 'TENSORRT_ENGINE_STATIC_BINDINGS'},
        'blurball': {'input_shape': [1, 9, 288, 512], 'output_shape': [1, 3, 288, 512],
                    'io_dtype': 'float32', 'input_layout': 'NCHW', 'output_layout': 'NCHW',
                    'shape_source': 'TENSORRT_ENGINE_STATIC_BINDINGS_VALIDATED_FIXED_EXPORT'},
    }


class TensorMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = JournalFixture(Path(self.directory.name))

    def use_new_contract(self, size=640, anchors=8400, width=None):
        manifest = self.fixture.manifest
        manifest['preprocessing']['yolo'] = 'opencv-linear-letterbox-round-pad114-bgr8-rgb-chw-v1'
        manifest['tensor_contracts'] = contracts(size, anchors, width)
        return manifest

    def test_legacy_report_is_preserved_without_inventing_tensor_dimensions(self):
        report = score(self.fixture.root, self.fixture.labels)
        self.assertIsNone(report['tensor_contracts'])
        self.assertIn('not recorded', report['tensor_contracts_limitation'])
        self.assertEqual(report['preprocessing'], self.fixture.manifest['preprocessing'])

    def test_actual_shape_metadata_does_not_change_640_scores(self):
        before = score(self.fixture.root, self.fixture.labels)
        self.use_new_contract()
        self.fixture.write()
        after = score(self.fixture.root, self.fixture.labels)
        self.assertEqual(before['accuracy'], after['accuracy'])
        self.assertEqual(before['hypothesis_counts'], after['hypothesis_counts'])
        self.assertEqual(before['gates'], after['gates'])
        self.assertEqual(after['tensor_contracts'], contracts())
        self.assertIsNone(after['tensor_contracts_limitation'])
        self.assertFalse(after['promotion_approved'])

    def test_shape_contracts_accept_standard_960_p2_and_rectangular_static_heads(self):
        for height, width, anchors in [(960, 960, 18900), (640, 640, 34000),
                                        (960, 960, 76500), (640, 960, 12600)]:
            with self.subTest(height=height, width=width, anchors=anchors):
                validate_manifest(self.use_new_contract(height, anchors, width))

    def test_new_revision_requires_recorded_shapes(self):
        manifest = self.use_new_contract()
        del manifest['tensor_contracts']
        with self.assertRaisesRegex(ValueError, 'both exact model contracts'):
            validate_manifest(manifest)

    def test_legacy_640_revision_rejects_960_claim(self):
        self.fixture.manifest['tensor_contracts'] = contracts(960, 18900)
        with self.assertRaisesRegex(ValueError, 'legacy 640'):
            validate_manifest(self.fixture.manifest)

    def test_legacy_640_revision_can_carry_explicit_640_shapes(self):
        self.fixture.manifest['tensor_contracts'] = contracts()
        validate_manifest(self.fixture.manifest)

    def test_bad_shape_dtype_layout_and_head_metadata_fail_closed(self):
        base = copy.deepcopy(self.use_new_contract())
        changes = [
            ('yolo', 'input_shape', [1, 3, 641, 640]),
            ('yolo', 'input_shape', [True, 3, 640, 640]),
            ('yolo', 'output_shape', [1, 5, 18900]),
            ('yolo', 'output_shape', [1, 5, True]),
            ('yolo', 'classes', 2),
            ('yolo', 'anchors', 0),
            ('yolo', 'io_dtype', 'float16'),
            ('yolo', 'input_layout', 'NHWC'),
            ('yolo', 'shape_source', 'CONFIGURED_HINT_NOT_ENGINE'),
            ('blurball', 'input_shape', [1, 3, 288, 512]),
            ('blurball', 'output_shape', [True, 3, 288, 512]),
            ('blurball', 'io_dtype', 'float16'),
        ]
        for model, key, value in changes:
            with self.subTest(model=model, key=key, value=value):
                manifest = copy.deepcopy(base)
                manifest['tensor_contracts'][model][key] = value
                with self.assertRaises(ValueError):
                    validate_manifest(manifest)

    def test_new_revision_rejects_missing_model_contract_and_unknown_preprocessing(self):
        manifest = self.use_new_contract()
        del manifest['tensor_contracts']['blurball']
        with self.assertRaises(ValueError):
            validate_manifest(manifest)
        manifest = self.use_new_contract()
        manifest['preprocessing']['yolo'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'unknown YOLO revision'):
            validate_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
