"""Independent decisions and corruption cases for the paired native journal scorer."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    from .score_paired_ball_candidates import (CONFIG_POLICIES, cpp_canonical_json,
                                               f32, regression_gates, score)
except ImportError:
    from score_paired_ball_candidates import (CONFIG_POLICIES, cpp_canonical_json,
                                             f32, regression_gates, score)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def candidate(row, model, index, center, raw_score=.9, mass=10):
    """Exact binary box geometry keeps fixture decisions independent of scorer math."""
    raw_score = f32(raw_score)
    x, y = center
    raw = {'class_id': 0, 'score': raw_score, 'x': (x - 2) / 1024,
           'y': (y - 2) / 1024, 'w': 4 / 1024, 'h': 4 / 1024}
    result = {key: copy.deepcopy(row[key]) for key in
              ('source_sha256', 'decoded_image_sha256', 'frame_id', 'source_pts_s', 'source_size')}
    result.update(model=model, model_sha256=row['_model_hashes'][model], candidate_index=index,
                  valid=True, invalid_reason=None, center_px=list(center), coordinate_basis='SOURCE_PIXELS',
                  center_method='YOLO_BOX_CENTER' if model == 'yolo' else 'HEATMAP_WEIGHTED_CENTROID',
                  raw_score=raw_score, score_kind='DETECTOR_SCORE' if model == 'yolo' else 'HEATMAP_PEAK',
                  confidence=None, confidence_reason='RAW_MODEL_SCORE_NOT_CORRECTNESS_CALIBRATED',
                  raw_detection=raw, blur_evidence=None)
    if model == 'blurball':
        result['blur_evidence'] = {'heatmap_peak': raw_score, 'component_mass': mass,
                                   'extent_source_px': 4, 'axis_deg': 0, 'axis_directed': False,
                                   'endpoints_px': [[x - 2, y], [x + 2, y]]}
    return result


def selection(reason, selected=None, yolo=None, blur=None, distance=None, eligible_yolo=0,
              eligible_blur=0, invalid=0):
    return {'selected': copy.deepcopy(selected), 'reason': reason, 'confidence': None,
            'confidence_reason': 'EXPERIMENTAL_UNCALIBRATED_SELECTOR',
            'yolo_support_index': yolo, 'blurball_support_index': blur,
            'agreement_distance_source_px': distance, 'invalid_candidates': invalid,
            'eligible_yolo': eligible_yolo, 'eligible_blurball': eligible_blur}


class JournalFixture:
    def __init__(self, root):
        self.root = root
        self.config = dict(CONFIG_POLICIES, agreement_radius_source_px=12,
                           blurball_agreement_peak=f32(.7), blurball_solo_peak=f32(.95),
                           within_model_ambiguity_ratio=.8, yolo_agreement_score=f32(.25),
                           yolo_proposal_score=f32(.1), yolo_solo_score=f32(.95))
        hashes = {}
        for name in ('source', 'yolo_model', 'blurball_model'):
            path = root / (name + '.bin')
            path.write_bytes(('fixed synthetic ' + name).encode())
            hashes[name] = digest(path.read_bytes())
        fusion_hash = digest(cpp_canonical_json(self.config).encode())
        self.manifest = {'schema_version': 1, 'evaluation': 'PAIRED_CAUSAL_NATIVE_CANDIDATE_REPLAY',
                         'source_path': str(root / 'source.bin'), 'source_sha256': hashes['source'],
                         'yolo_model_path': str(root / 'yolo_model.bin'), 'yolo_model_sha256': hashes['yolo_model'],
                         'blurball_model_path': str(root / 'blurball_model.bin'), 'blurball_model_sha256': hashes['blurball_model'],
                         'runtime_sha256': digest(b'native runtime'), 'fusion_sha256': fusion_hash,
                         'source_fps': 50, 'source_declared_frames': 13, 'requested_frames': 13,
                         'source_interval': 'COMPLETE_RECORDING',
                         'timestamp_basis': 'ACTUAL_OPENCV_FFMPEG_DECODER_PTS_SECONDS',
                         'invalid_pts_policy': 'FAIL_CLOSED_PARTIAL_JOURNAL', 'maximum_temporal_gap_s': .05,
                         'gap_policy': 'RESET_CAUSAL_HISTORY_AND_STANDALONE_ANCHOR',
                         'fusion': self.config, 'preprocessing': {
                             'yolo': 'opencv-bgr-rgb-letterbox-640-chw-div255-v1',
                             'blurball': 'opencv-affine-rgb-imagenet-512x288-three-frames-v1',
                             'input_transform': 'RAW_SOURCE_NO_FILTER',
                             'decoded_image_hash': 'SHA256_UINT32_LE_WIDTH_HEIGHT_OPENCV_TYPE_THEN_PACKED_BGR_ROWS'},
                         'yolo_proposal_threshold': f32(.1), 'yolo_nms_threshold': f32(.45),
                         'standalone_yolo_threshold': .25, 'blurball_heatmap_threshold': f32(.7),
                         'standalone_blurball_selection': 'HIGHEST_COMPONENT_MASS_STRICT_300_SOURCE_PIXEL_ANCHOR_GATE',
                         'temporal_policy': 'CAUSAL_CHANNEL_2_T_MINUS_2_T_MINUS_1_T',
                         'lookahead_frames': 0, 'warmup_frames': 2, 'source_images_shared': True,
                         'opened_camera': False, 'labels_loaded': False, 'confidence_calibrated': False,
                         'promotion_approved': False, 'experimental': True}
        # Decisions are specified explicitly, not generated by the selector under test.
        # (YOLO specs, BlurBall specs, reason, selected model/index, supports, distance)
        specs = [
            ([(10, .99)], [], 'BLURBALL_CAUSAL_WARMUP', None, None, None, None),
            ([(10, .99)], [], 'BLURBALL_CAUSAL_WARMUP', None, None, None, None),
            ([(12, .9)], [(10, .9, 10)], 'SPATIAL_AGREEMENT_BLURBALL_COORDINATE', ('blurball', 0), 0, 0, 2),
            ([(100, .9)], [(10, .9, 10)], 'DETECTOR_DISAGREEMENT', None, 0, 0, 90),
            ([(10, .9)], [(10, .9, 10), (30, .9, 9)], 'AMBIGUOUS_BLURBALL_CANDIDATES', None, None, None, None),
            ([(10, .9), (50, .85)], [(10, .9, 10)], 'AMBIGUOUS_YOLO_CANDIDATES', None, None, None, None),
            ([(10, .99)], [], 'CONSERVATIVE_YOLO_ONLY_FALLBACK', ('yolo', 0), 0, None, None),
            ([], [(600, .99, 10)], 'STRONG_BLURBALL_ONLY', ('blurball', 0), None, 0, None),
            ([], [(900, .9, 12)], 'BLURBALL_ONLY_INSUFFICIENT_EVIDENCE', None, None, 0, None),
            ([], [(900, .99, 12)], 'STRONG_BLURBALL_ONLY', ('blurball', 0), None, 0, None),
            ([], [], 'NO_ELIGIBLE_CANDIDATE', None, None, None, None),
            ([(10, .5)], [], 'YOLO_ONLY_INSUFFICIENT_EVIDENCE', None, 0, None, None),
            ([], [(10, .8, 10)], 'BLURBALL_ONLY_INSUFFICIENT_EVIDENCE', None, None, 0, None),
        ]
        self.rows = []
        for fid, (yolo_specs, blur_specs, reason, chosen, ys, bs, distance) in enumerate(specs, 1):
            ready = fid >= 3
            row = {'source_sha256': hashes['source'], 'decoded_image_sha256': digest(f'image {fid}'.encode()),
                   'frame_id': fid, 'source_pts_s': (fid - 1) / 50, 'source_size': [1024, 1024],
                   'fusion_sha256': fusion_hash, 'nominal_source_time_s': (fid - 1) / 50,
                   'same_decoded_pixels_as_previous': False, 'temporal_reset_reason': None,
                   'temporal_window_frame_ids': list(range(max(1, fid - 2), fid + 1)),
                   'temporal_window_ready': ready,
                   '_model_hashes': {'yolo': hashes['yolo_model'], 'blurball': hashes['blurball_model']}}
            row['yolo_candidates'] = [candidate(row, 'yolo', i, [x, 10], s) for i, (x, s) in enumerate(yolo_specs)]
            row['blurball_candidates'] = [candidate(row, 'blurball', i, [x, 10], s, m) for i, (x, s, m) in enumerate(blur_specs)]
            row['standalone_yolo'] = copy.deepcopy(row['yolo_candidates'][0]) if yolo_specs else None
            # Frame9 is exactly 300px from anchor and must abstain. Frame10 then reacquires.
            row['standalone_blurball'] = copy.deepcopy(row['blurball_candidates'][0]) if blur_specs and fid != 9 else None
            chosen_candidate = row[chosen[0] + '_candidates'][chosen[1]] if chosen else None
            row['experimental_selection'] = selection(reason, chosen_candidate, ys, bs, distance,
                                                        len(yolo_specs), len(blur_specs))
            row['timings'] = {'yolo_preprocess_wall_ms': 1, 'yolo_inference_wall_ms': 2,
                              'yolo_gpu_ms': None, 'blurball_adapter_wall_ms': 3 if ready else .5,
                              'blurball_engine_wall_ms': 2 if ready else 0, 'blurball_gpu_ms': 1 if ready else None,
                              'blurball_inference_executed': ready, 'frame_before_journal_wall_ms': 5,
                              'live_latency_measured': False}
            del row['_model_hashes']
            self.rows.append(row)
        self.summary = {'status': 'COMPLETE', 'source_frames_decoded': 13, 'analyzed_frames': 13,
                        'source_declared_frames': 13, 'source_count_matches_metadata': True,
                        'entire_recording_decoded': True, 'decoder_eof_reached': True,
                        'source_pts_strictly_increasing': True, 'consecutive_pixel_duplicates': 0,
                        'temporal_resets': 0, 'causal_warmup_frames': 2, 'lookahead_frames': 0,
                        'opened_camera': False, 'accuracy_measured': False, 'live_latency_measured': False,
                        'promotion_approved': False, 'labels_loaded': False,
                        'original_source_mutated': False, 'production_backend_modified': False}
        for key in ('source_sha256', 'yolo_model_sha256', 'blurball_model_sha256', 'fusion_sha256'):
            self.summary[key] = self.manifest[key]
        self.completion = {'status': 'COMPLETE', 'analyzed_frames': 13, 'total_frames': 13}
        self.update_summary()
        self.write()
        self.labels = root / 'labels.csv'
        self.write_labels()

    def update_summary(self):
        self.summary['source_pts_first_s'] = self.rows[0]['source_pts_s']
        self.summary['source_pts_last_s'] = self.rows[-1]['source_pts_s']
        intervals = sorted(b['source_pts_s'] - a['source_pts_s'] for a, b in zip(self.rows, self.rows[1:]))
        self.summary['median_source_interval_s'] = intervals[len(intervals) // 2] if intervals else None
        for model in ('yolo', 'blurball'):
            self.summary['standalone_' + model + '_selected_frames'] = sum(r['standalone_' + model] is not None for r in self.rows)
        self.summary['experimental_selected_frames'] = sum(r['experimental_selection']['selected'] is not None for r in self.rows)
        for target, key in [('yolo_inference_total_wall_ms', 'yolo_inference_wall_ms'),
                            ('blurball_adapter_total_wall_ms', 'blurball_adapter_wall_ms'),
                            ('blurball_gpu_total_ms', 'blurball_gpu_ms')]:
            self.summary[target] = sum(r['timings'][key] or 0 for r in self.rows)

    def write_labels(self, rows=None):
        if rows is None:
            rows = [f'{fid},1,10,10' for fid in range(12)] + ['12,0,,']
        self.labels.write_text('Frame,Visibility,X,Y\n' + '\n'.join(rows) + '\n', encoding='utf-8')

    def write(self):
        for name, value in [('manifest.json', self.manifest), ('diagnostic_summary.json', self.summary),
                            ('completion.json', self.completion)]:
            (self.root / name).write_text(json.dumps(value, allow_nan=False) + '\n', encoding='utf-8')
        lines = [(json.dumps(r, allow_nan=False) + '\n').encode() for r in self.rows]
        offsets, offset = [], 0
        for line in lines:
            offsets.append(offset)
            offset += len(line)
        (self.root / 'frames.jsonl').write_bytes(b''.join(lines))
        (self.root / 'frames.idx').write_bytes(struct.pack('<' + 'Q' * len(offsets), *offsets))


class PairedScorerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture = JournalFixture(self.root)

    def test_known_independent_decisions_warmup_disagreement_no_blend_and_anchor(self):
        report = score(self.root, self.fixture.labels)
        self.assertEqual(report['audit']['status'], 'PASS')
        self.assertEqual(report['hypothesis_counts']['experimental_selector']['selected_frames'], 4)
        self.assertEqual(report['selector_reason_histogram']['BLURBALL_CAUSAL_WARMUP'], 2)
        self.assertEqual(report['selector_reason_histogram']['DETECTOR_DISAGREEMENT'], 1)
        self.assertEqual(report['selector_reason_histogram']['AMBIGUOUS_BLURBALL_CANDIDATES'], 1)
        self.assertEqual(report['selector_reason_histogram']['AMBIGUOUS_YOLO_CANDIDATES'], 1)
        self.assertEqual(report['accuracy']['after_causal_warmup']['experimental_selector']['frames'], 11)
        self.assertEqual(self.fixture.rows[2]['experimental_selection']['selected']['center_px'], [10, 10])
        self.assertEqual(self.fixture.rows[2]['experimental_selection']['selected']['raw_score'], f32(.9))
        self.assertIsNone(self.fixture.rows[8]['standalone_blurball'])
        self.assertEqual(self.fixture.rows[9]['standalone_blurball']['center_px'], [900, 10])
        self.assertFalse(report['promotion_approved'])

    def test_native_canonical_hash_uses_17_digits(self):
        # Authoritative native default config hash from the completed official journal.
        self.assertEqual(digest(cpp_canonical_json(self.fixture.config).encode()),
                         'c580ac8e7f8271264798d500838f2b5d9f80612558b51ae79b745aff1b9f0b1b')
        shortest = json.dumps(self.fixture.config, sort_keys=True, separators=(',', ':'))
        self.assertNotEqual(digest(shortest.encode()), self.fixture.manifest['fusion_sha256'])
        self.fixture.manifest['fusion']['agreement_radius_source_px'] = 13
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'fusion SHA-256'):
            score(self.root)

    def test_cpp_float32_yolo_geometry_matches_real_rounding_case(self):
        row = self.fixture.rows[0]
        box = row['yolo_candidates'][0]
        row['source_size'] = [1280, 720]
        box['source_size'] = [1280, 720]
        box['raw_detection'] = {'class_id': 0, 'score': f32(.99),
                                'x': 0.63125002384185791, 'y': 0.27048611640930176,
                                'w': 0.02031249925494194, 'h': 0.036111112684011459}
        box['center_px'] = [821.00006103515625, 207.75]
        row['standalone_yolo'] = copy.deepcopy(box)
        # Subsequent frame dimensions return to fixture size: reset occurs at frame2.
        self._reset_at(1, 'SOURCE_DIMENSION_CHANGE')
        self.fixture.write()
        self.assertEqual(score(self.root)['audit']['status'], 'PASS')
        naive = (box['raw_detection']['x'] + box['raw_detection']['w'] * .5) * 1280
        self.assertNotEqual(naive, box['center_px'][0])

    def test_candidates_cannot_mix_frames_images_times_sizes_hashes_models_or_indices(self):
        original = copy.deepcopy(self.fixture.rows)
        edits = [('source_sha256', 'a' * 64), ('decoded_image_sha256', 'b' * 64),
                 ('frame_id', 2), ('source_pts_s', .001), ('source_size', [1024, 1000]),
                 ('model_sha256', 'c' * 64), ('model', 'unknown'), ('candidate_index', 1)]
        for key, value in edits:
            with self.subTest(key=key):
                self.fixture.rows = copy.deepcopy(original)
                self.fixture.rows[0]['yolo_candidates'][0][key] = value
                self.fixture.write()
                with self.assertRaises(ValueError):
                    score(self.root)

    def test_selected_candidate_must_retain_all_original_evidence(self):
        original = copy.deepcopy(self.fixture.rows)
        edits = [('center_px', [11, 10]), ('raw_score', .5), ('confidence', .99),
                 ('model', 'yolo'), ('blur_evidence', None)]
        for key, value in edits:
            with self.subTest(key=key):
                self.fixture.rows = copy.deepcopy(original)
                self.fixture.rows[2]['experimental_selection']['selected'][key] = value
                self.fixture.write()
                with self.assertRaisesRegex(ValueError, 'experimental selected'):
                    score(self.root)

    def test_selector_reason_counts_support_and_distance_are_recomputed(self):
        original = copy.deepcopy(self.fixture.rows)
        edits = [('reason', 'STRONG_BLURBALL_ONLY'), ('eligible_yolo', 0),
                 ('eligible_blurball', 0), ('invalid_candidates', 1),
                 ('yolo_support_index', None), ('blurball_support_index', 1),
                 ('agreement_distance_source_px', 1.99), ('confidence', .9)]
        for key, value in edits:
            with self.subTest(key=key):
                self.fixture.rows = copy.deepcopy(original)
                self.fixture.rows[2]['experimental_selection'][key] = value
                self.fixture.write()
                with self.assertRaisesRegex(ValueError, 'experimental'):
                    score(self.root)

    def test_geometry_invalid_flag_and_confidence_are_not_trusted(self):
        original = copy.deepcopy(self.fixture.rows)
        edits = [('valid', False), ('invalid_reason', 'BAD'), ('confidence', .99),
                 ('center_px', [11, 10]), ('score_kind', 'CALIBRATED_PROBABILITY'),
                 ('coordinate_basis', 'HEATMAP_PIXELS')]
        for key, value in edits:
            with self.subTest(key=key):
                self.fixture.rows = copy.deepcopy(original)
                self.fixture.rows[0]['yolo_candidates'][0][key] = value
                self.fixture.write()
                with self.assertRaises(ValueError):
                    score(self.root)

    def test_standalone_policies_are_checked_independently(self):
        # Ambiguous experimental YOLO frame still has one top-score standalone.
        self.fixture.rows[5]['standalone_yolo'] = copy.deepcopy(self.fixture.rows[5]['yolo_candidates'][1])
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'standalone YOLO'):
            score(self.root)
        self.fixture = JournalFixture(self.root)
        # Strict 300px anchor rejects frame9 even though a candidate exists.
        self.fixture.rows[8]['standalone_blurball'] = copy.deepcopy(self.fixture.rows[8]['blurball_candidates'][0])
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'mass/300px'):
            score(self.root)

    def test_invalid_candidate_accounting_filters_class_without_trusting_flag(self):
        row = self.fixture.rows[0]
        invalid = copy.deepcopy(row['yolo_candidates'][0])
        invalid.update(candidate_index=1, valid=False, invalid_reason='NON_BALL_CLASS')
        invalid['raw_detection']['class_id'] = 1
        row['yolo_candidates'].append(invalid)
        row['experimental_selection']['invalid_candidates'] = 1
        self.fixture.write()
        result = score(self.root)
        self.assertEqual(result['audit']['counters']['yolo_invalid_candidates'], 1)
        self.assertEqual(result['hypothesis_counts']['standalone_yolo']['selected_frames'], 8)

    def test_source_and_model_files_verified_when_available(self):
        for name in ('source', 'yolo_model', 'blurball_model'):
            with self.subTest(name=name):
                self.fixture = JournalFixture(self.root)
                (self.root / (name + '.bin')).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'file SHA-256 mismatch'):
                    score(self.root)
        self.fixture = JournalFixture(self.root)
        (self.root / 'source.bin').unlink()
        result = score(self.root)
        self.assertIsNone(result['audit']['source_and_model_files']['source']['sha256_match'])
        self.assertFalse(result['audit']['candidate_ground_truth_verified_by_hashes'])

    def test_runtime_hash_is_recorded_provenance_without_supplied_binary(self):
        result = score(self.root)
        self.assertEqual(result['runtime_binary_sha256_recorded_from_manifest'], digest(b'native runtime'))
        self.assertFalse(result['runtime_binary_current_file_reverified'])
        self.assertFalse(result['audit']['runtime_binary_current_file_reverified'])
        self.assertIsNone(result['runtime_binary_supplied_path'])
        self.assertIsNone(result['runtime_binary_sha256_current_file'])
        self.assertIn('historical executable bytes have not been reverified', result['runtime_binary_limitation_reason'])

    def test_matching_runtime_binary_is_reverified_from_actual_bytes(self):
        runtime = self.root / 'archived native runtime.exe'
        runtime.write_bytes(b'native runtime')
        result = score(self.root, runtime_binary=runtime)
        self.assertTrue(result['runtime_binary_current_file_reverified'])
        self.assertEqual(result['runtime_binary_sha256_current_file'], self.fixture.manifest['runtime_sha256'])
        self.assertEqual(result['runtime_binary_supplied_path'], str(runtime.resolve()))
        self.assertIsNone(result['runtime_binary_limitation_reason'])

    def test_mismatching_or_missing_supplied_runtime_binary_rejects_audit(self):
        runtime = self.root / 'runtime.exe'
        runtime.write_bytes(b'overwritten current runtime')
        with self.assertRaisesRegex(ValueError, 'Runtime binary SHA-256 mismatch'):
            score(self.root, runtime_binary=runtime)
        runtime.unlink()
        with self.assertRaisesRegex(ValueError, 'Runtime binary is missing/unreadable'):
            score(self.root, runtime_binary=runtime)

    def test_partial_missing_and_mismatched_completion_counts_rejected(self):
        for file, field, value in [('completion.json', 'status', 'PARTIAL'),
                                    ('completion.json', 'analyzed_frames', 12),
                                    ('completion.json', 'total_frames', 12),
                                    ('diagnostic_summary.json', 'source_frames_decoded', 12),
                                    ('diagnostic_summary.json', 'experimental_selected_frames', 3),
                                    ('diagnostic_summary.json', 'source_sha256', 'a' * 64),
                                    ('manifest.json', 'requested_frames', 12)]:
            with self.subTest(file=file, field=field):
                self.fixture = JournalFixture(self.root)
                path = self.root / file
                obj = json.loads(path.read_text())
                obj[field] = value
                path.write_text(json.dumps(obj))
                with self.assertRaises(ValueError):
                    score(self.root)
        self.fixture = JournalFixture(self.root)
        (self.root / 'completion.json').unlink()
        with self.assertRaisesRegex(ValueError, 'required journal file'):
            score(self.root)
        self.fixture.write()
        (self.root / 'error.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'error/partial'):
            score(self.root)

    def test_index_offsets_alignment_and_jsonl_full_accounting(self):
        for damage in ('offset', 'extra-index-byte', 'missing-index', 'extra-row', 'missing-row', 'partial-line', 'blank-row'):
            with self.subTest(damage=damage):
                self.fixture.write()
                path = self.root / ('frames.idx' if 'index' in damage or damage == 'offset' else 'frames.jsonl')
                data = path.read_bytes()
                if damage == 'offset':
                    data = struct.pack('<Q', 1) + data[8:]
                elif damage == 'extra-index-byte':
                    data += b'!'
                elif damage == 'missing-index':
                    data = data[:-8]
                elif damage == 'extra-row':
                    data += data.splitlines(keepends=True)[0]
                elif damage == 'missing-row':
                    data = b''.join(data.splitlines(keepends=True)[:-1])
                elif damage == 'partial-line':
                    data = data[:-1]
                else:
                    data += b'\n'
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    score(self.root)

    def test_nonconsecutive_source_frames_duplicate_json_fields_and_nonfinite_json(self):
        self.fixture.rows[3]['frame_id'] = 3
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'frame_id'):
            score(self.root)
        for fragment in ('"source_fps":NaN,', '"unknown_number":1e999,', '"source_fps":50,'):
            with self.subTest(fragment=fragment):
                self.fixture = JournalFixture(self.root)
                manifest = self.root / 'manifest.json'
                manifest.write_text('{' + fragment + manifest.read_text()[1:])
                with self.assertRaises(ValueError):
                    score(self.root)

    def _reset_at(self, start, reason):
        # Replace affected causal observations with warmup, without fabricating padding.
        self.fixture.rows[start]['temporal_reset_reason'] = reason
        for i in range(start, len(self.fixture.rows)):
            row = self.fixture.rows[i]
            row['temporal_window_frame_ids'] = list(range(max(start + 1, i - 1), i + 2))
            ready = i >= start + 2
            row['temporal_window_ready'] = ready
            if not ready:
                row['blurball_candidates'], row['standalone_blurball'] = [], None
                row['experimental_selection'] = selection('BLURBALL_CAUSAL_WARMUP', eligible_yolo=len(row['yolo_candidates']))
                row['timings'].update(blurball_inference_executed=False, blurball_engine_wall_ms=0, blurball_gpu_ms=None)
        self.fixture.summary['temporal_resets'] = 1
        self.fixture.summary['causal_warmup_frames'] = sum(not r['temporal_window_ready'] for r in self.fixture.rows)
        self.fixture.update_summary()

    def test_temporal_gap_reset_and_actual_pts_are_audited(self):
        # Reset late at frame12 to retain the independently specified earlier decisions.
        for row in self.fixture.rows[11:]:
            row['source_pts_s'] += 1
            for c in row['yolo_candidates'] + row['blurball_candidates']:
                c['source_pts_s'] = row['source_pts_s']
            for key in ('standalone_yolo', 'standalone_blurball'):
                if row[key]:
                    row[key]['source_pts_s'] = row['source_pts_s']
        self._reset_at(11, 'SOURCE_PTS_GAP')
        self.fixture.write()
        result = score(self.root, self.fixture.labels)
        self.assertEqual(result['audit']['counters']['temporal_resets'], 1)
        self.assertEqual(result['accuracy']['after_causal_warmup']['experimental_selector']['frames'], 9)
        self.fixture.rows[11]['temporal_reset_reason'] = None
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'temporal_reset_reason'):
            score(self.root)
        self.fixture = JournalFixture(self.root)
        self.fixture.rows[2]['source_pts_s'] = self.fixture.rows[1]['source_pts_s']
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            score(self.root)

    def test_windows_cannot_use_future_or_duplicate_frames(self):
        for window in ([1, 2, 4], [1, 3, 3], [2, 3], [0, 1, 2]):
            with self.subTest(window=window):
                self.fixture.rows[2]['temporal_window_frame_ids'] = window
                self.fixture.write()
                with self.assertRaisesRegex(ValueError, 'causal window'):
                    score(self.root)

    def test_wrong_location_at_exact_four_counts_fp_and_miss_and_strict_boundary(self):
        rows = [f'{fid},1,10,10' for fid in range(13)]
        rows[2] = '2,1,14,10'  # Selected existing BlurBall center is10: exactly4 must fail.
        self.fixture.write_labels(rows)
        result = score(self.root, self.fixture.labels)['accuracy']['full_recording']['experimental_selector']
        self.assertEqual(result['tp'], 1)  # Only frame7's YOLO fallback is correct.
        self.assertEqual(result['wrong_location_predictions'], 3)
        self.assertEqual(result['no_prediction_visible_frames'], 9)
        self.assertEqual(result['missed_visible_reference_frames'], 12)
        self.assertEqual(result['false_positives'], 3)
        self.assertAlmostEqual(result['recall'], 1 / 13)
        rows[2] = '2,1,13.999,10'
        self.fixture.write_labels(rows)
        self.assertEqual(score(self.root, self.fixture.labels)['accuracy']['full_recording']['experimental_selector']['tp'], 2)

    def test_visibility_duplicate_unknown_missing_and_malformed_labels_rejected(self):
        good = [f'{fid},1,10,10' for fid in range(13)]
        cases = [good[:-1], good + ['0,1,10,10'], good + ['13,1,10,10'],
                 ['0,2,10,10'] + good[1:], ['0,unknown,10,10'] + good[1:],
                 ['0,1,,10'] + good[1:], ['0,1,nan,10'] + good[1:],
                 ['0,1,inf,10'] + good[1:], ['0,1,bad,10'] + good[1:],
                 ['0,1,10'] + good[1:], ['0,1,10,10,extra'] + good[1:],
                 ['-1,1,10,10'] + good[1:], ['0.0,1,10,10'] + good[1:]]
        for rows in cases:
            with self.subTest(rows=rows[:1], count=len(rows)):
                self.fixture.write_labels(rows)
                with self.assertRaises(ValueError):
                    score(self.root, self.fixture.labels)
        self.fixture.labels.write_text('Frame,Visibility,X,X\n0,1,10,10\n')
        with self.assertRaisesRegex(ValueError, 'headers'):
            score(self.root, self.fixture.labels)

    def test_positive_finite_tolerance_required(self):
        for tolerance in (0, -1, float('nan'), float('inf'), float('-inf'), True, '4'):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, 'Tolerance'):
                    score(self.root, self.fixture.labels, tolerance)

    def test_optional_reference_source_identity_cannot_mix_recordings(self):
        source = self.fixture.manifest['source_sha256']
        header = 'Frame,Visibility,X,Y,source_sha256\n'
        rows = [f'{fid},1,10,10,{source}' for fid in range(13)]
        self.fixture.labels.write_text(header + '\n'.join(rows) + '\n')
        self.assertEqual(score(self.root, self.fixture.labels)['audit']['status'], 'PASS')
        rows[4] = '4,1,10,10,' + 'a' * 64
        self.fixture.labels.write_text(header + '\n'.join(rows) + '\n')
        with self.assertRaisesRegex(ValueError, 'Mixed/unknown reference source'):
            score(self.root, self.fixture.labels)

    def test_unlabeled_and_declared_no_ball_never_automatic_ground_truth(self):
        self.fixture.manifest['user_declared_no_ball'] = True
        self.fixture.write()
        result = score(self.root)
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['gates'])
        self.assertFalse(result['accuracy_measured'])

    def test_gates_fail_recall_even_with_perfect_precision_and_do_not_invent_negatives(self):
        values = {key: {'precision': precision, 'recall': recall, 'f1': f1,
                        'absent_reference_frames': 0, 'absent_false_positives': 0}
                  for key, precision, recall, f1 in [('standalone_yolo', .7, .6, .64),
                                                    ('standalone_blurball_causal', .98, .94, .96),
                                                    ('experimental_selector', 1., .02, .04)]}
        gates = regression_gates(values)
        self.assertFalse(gates['passed'])
        self.assertEqual(gates['failed_checks'], ['recall_no_regression'])
        self.assertTrue(gates['checks']['precision_no_regression']['passed'])
        self.assertIsNone(gates['checks']['negative_false_positives_no_regression']['passed'])

    def test_cli_writes_new_report_and_refuses_overwrite(self):
        output = self.root / 'metrics.json'
        runtime = self.root / 'runtime.exe'
        runtime.write_bytes(b'native runtime')
        command = [sys.executable, str(Path(__file__).with_name('score_paired_ball_candidates.py')),
                   str(self.root), str(output), '--labelsCSV', str(self.fixture.labels),
                   '--runtime-binary', str(runtime)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(output.is_file())
        self.assertTrue(json.loads(output.read_text())['runtime_binary_current_file_reverified'])
        original = output.read_bytes()
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
