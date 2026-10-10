"""Synthetic fail-closed journal/selection tests; no real labels changed."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from summarize_ball_hypotheses import audit_journal, sha256_file, summarize


class HypothesisSummaryTests(unittest.TestCase):
    def fixture(self, root, count=20):
        source = root/'source.mp4'; source.write_bytes(b'synthetic-source-identity')
        model = root/'model.engine'; model.write_bytes(b'synthetic-engine-identity')
        runtime = root/'runtime.exe'; runtime.write_bytes(b'synthetic-runtime-identity')
        provenance = {'backend': 'blurball', 'experimental': True,
                      'heatmap_threshold': .7, 'lookahead_frames': 0,
                      'model_sha256': sha256_file(model), 'score_calibrated': False,
                      'score_kind': 'uncalibrated_heatmap_peak',
                      'temporal_policy': 'causal_channel_2',
                      'timestamp_basis': 'nominal_source_fps_or_live_capture_clock',
                      'warmup_frames': 2}
        manifest = {'schema_version': 1, 'source': str(source),
                    'source_sha256': sha256_file(source), 'source_declared_frames': count,
                    'source_fps': 10., 'total_frames': count,
                    'source_interval': 'COMPLETE_RECORDING', 'ball_model': provenance,
                    'ball_model_sha256': sha256_file(model), 'runtime_sha256': sha256_file(runtime),
                    'evaluation_policy': 'CAUSAL_SOURCE_REPLAY_NOT_LIVE_LATENCY'}
        rows = [{'frame_id': frame, 'source_time': (frame-1)/10.,
                 'source_width': 1920, 'source_height': 1080,
                 'ball': {'x': .2 if frame < 10 else .8, 'y': .3, 'confidence': .8}
                 if frame in (3, 4, 5, 10, 13, 14, 19, 20) else None,
                 'ball_model': copy.deepcopy(provenance)} for frame in range(1, count+1)]
        journal = root/'journal'; journal.mkdir()
        self.write_json(journal/'manifest.json', manifest)
        self.write_json(journal/'completion.json', {'status': 'COMPLETE', 'analyzed_frames': count, 'total_frames': count})
        self.write_rows(journal, rows)
        return journal, model, runtime, source, manifest, rows

    def write_json(self, path, value):
        path.write_text(json.dumps(value)+'\n', encoding='utf-8')

    def write_rows(self, journal, rows, terminated=True):
        data, offsets = b'', []
        for row in rows:
            offsets.append(len(data))
            data += (json.dumps(row)+'\n').encode()
        if not terminated:
            data = data[:-1]
        (journal/'frames.jsonl').write_bytes(data)
        (journal/'frames.idx').write_bytes(b''.join(struct.pack('<Q', offset) for offset in offsets))

    def test_complete_journal_counts_runs_hashes_and_unreviewed_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal, model, runtime, source, _, _ = self.fixture(Path(temporary))
            report, selection = summarize(journal, model, runtime_path=runtime, max_bursts=3, session_id='session_1')
            self.assertEqual(report['accepted_hypotheses']['count'], 8)
            self.assertEqual(report['accepted_hypotheses']['fraction_of_all_frames'], .4)
            self.assertAlmostEqual(report['accepted_hypotheses']['fraction_of_eligible_frames'], 8/18)
            self.assertEqual(report['observation_runs']['count'], 4)
            self.assertEqual(report['observation_runs']['longest_run_frames'], 3)
            first = report['observation_runs']['runs'][0]
            self.assertEqual((first['start_source_frame'],first['end_source_frame']), (3,5))
            self.assertAlmostEqual(first['timestamp_span_s'], .2)
            self.assertAlmostEqual(first['nominal_frame_coverage_s'], .3)
            self.assertEqual(sum(item['count'] for item in report['spatial_histogram']['bins']), 8)
            self.assertEqual(report['identity']['source_sha256'], sha256_file(source))
            self.assertEqual(report['identity']['journal_sha256'], sha256_file(journal/'frames.jsonl'))
            self.assertTrue(report['identity']['runtime_hash_verified'])
            self.assertEqual(report['truth']['review_status'], 'UNREVIEWED')
            self.assertIsNone(report['accuracy']['confirmed_false_positive_frames'])
            self.assertIsNone(report['accuracy']['precision'])
            self.assertFalse(selection['automatic_labels_created'])
            self.assertFalse(selection['images_imported'])
            self.assertEqual(selection['session_id'], 'session_1')
            self.assertEqual(selection['selected_bursts'], 3)

    def test_selection_is_bounded_nonoverlapping_and_deterministically_diverse(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal, model, _, _, _, rows = self.fixture(Path(temporary), 120)
            for row in rows:
                frame = row['frame_id']
                if frame > 2:
                    row['ball'] = {'x': .1 if frame < 61 else .9,
                                   'y': .1 if frame < 61 else .9, 'confidence': .8}
            self.write_rows(journal, rows)
            _, selection = summarize(journal, model, max_bursts=12)
            _, repeated = summarize(journal, model, max_bursts=12)
            self.assertEqual(selection, repeated)
            bursts = selection['bursts']
            self.assertEqual(len(bursts), 12)
            self.assertGreaterEqual(len({burst['temporal_bin'] for burst in bursts}), 10)
            self.assertEqual(len({tuple(burst['spatial_bin'].values()) for burst in bursts}), 2)
            for index, burst in enumerate(bursts):
                self.assertEqual(burst['end_source_frame'] - burst['start_source_frame'], 4)
                self.assertGreaterEqual(burst['start_source_frame'], 1)
                self.assertLessEqual(burst['end_source_frame'], 120)
                self.assertGreater(burst['accepted_hypotheses_in_burst'], 0)
                if index:
                    self.assertGreater(burst['start_source_frame'], bursts[index-1]['end_source_frame'])

    def test_no_hypotheses_and_short_recording_do_not_invent_candidates(self):
        for count in (3,20):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                journal, model, _, _, _, rows = self.fixture(Path(temporary), count)
                for row in rows: row['ball'] = None
                self.write_rows(journal, rows)
                report, selection = summarize(journal, model)
                self.assertEqual(report['accepted_hypotheses']['count'], 0)
                self.assertEqual(report['observation_runs']['runs'], [])
                self.assertEqual(selection['bursts'], [])
                self.assertIsNone(report['accuracy']['precision'])

    def test_corrupt_frame_rows_fail_closed(self):
        mutations = {
            'duplicate_id': lambda rows: rows[3].update(frame_id=3),
            'boolean_id': lambda rows: rows[0].update(frame_id=True),
            'reordered_ids': lambda rows: rows[2].update(frame_id=5),
            'wrong_time': lambda rows: rows[2].update(source_time=.21),
            'nonfinite_time': lambda rows: rows[2].update(source_time=float('nan')),
            'boolean_time': lambda rows: rows[2].update(source_time=True),
            'wrong_source': lambda rows: rows[2].update(source='different.mp4'),
            'wrong_source_hash': lambda rows: rows[2].update(source_sha256='0'*64),
            'wrong_model_hash': lambda rows: rows[2]['ball_model'].update(model_sha256='0'*64),
            'changed_policy': lambda rows: rows[2]['ball_model'].update(lookahead_frames=1),
            'dimension_change': lambda rows: rows[2].update(source_width=1280),
            'coordinate_outside': lambda rows: rows[2]['ball'].update(x=1.1),
            'negative_score': lambda rows: rows[2]['ball'].update(confidence=-.1),
            'infinite_point': lambda rows: rows[2]['ball'].update(y=float('inf')),
            'missing_ball': lambda rows: rows[2].pop('ball'),
            'warmup_hypothesis': lambda rows: rows[0].update(ball={'x': .5,'y':.5,'confidence':.8}),
            'unrelated_nonfinite': lambda rows: rows[2].update(ball_flight={'fit': [float('inf')]}),
            'missing_last': lambda rows: rows.pop(),
            'extra_row': lambda rows: rows.append(copy.deepcopy(rows[-1])),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                journal, model, _, _, _, rows = self.fixture(Path(temporary))
                mutate(rows); self.write_rows(journal, rows)
                with self.assertRaises(ValueError): audit_journal(journal, model)

    def test_incomplete_manifests_indices_and_changed_identities_fail_closed(self):
        for case in ('partial','bad_count','prefix','source_changed','model_changed','runtime_changed',
                     'index_truncated','index_corrupt','unterminated','manifest_model_disagrees',
                     'noncausal','boolean_schema','duplicate_json_key'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                journal, model, runtime, source, manifest, rows = self.fixture(Path(temporary))
                if case=='partial': self.write_json(journal/'completion.json', {'status':'PARTIAL','analyzed_frames':19,'total_frames':20})
                elif case=='bad_count': self.write_json(journal/'completion.json', {'status':'COMPLETE','analyzed_frames':19,'total_frames':20})
                elif case=='prefix': manifest['source_interval']='CHRONOLOGICAL_PREFIX'
                elif case=='source_changed': source.write_bytes(b'changed')
                elif case=='model_changed': model.write_bytes(b'changed')
                elif case=='runtime_changed': runtime.write_bytes(b'changed')
                elif case=='index_truncated': (journal/'frames.idx').write_bytes((journal/'frames.idx').read_bytes()[:-1])
                elif case=='index_corrupt':
                    index=bytearray((journal/'frames.idx').read_bytes()); index[8]=0
                    (journal/'frames.idx').write_bytes(index)
                elif case=='unterminated': self.write_rows(journal, rows, False)
                elif case=='manifest_model_disagrees': manifest['ball_model']['model_sha256']='0'*64
                elif case=='noncausal': manifest['ball_model']['lookahead_frames']=1
                elif case=='boolean_schema': manifest['schema_version']=True
                elif case=='duplicate_json_key':
                    line=(journal/'frames.jsonl').read_bytes()
                    (journal/'frames.jsonl').write_bytes(line.replace(b'"frame_id": 1', b'"frame_id": 1, "frame_id": 1',1))
                self.write_json(journal/'manifest.json',manifest)
                with self.assertRaises(ValueError): audit_journal(journal,model,runtime_path=runtime)

    def test_selection_limits_reject_boolean_zero_or_unbounded_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal, model, _, _, _, _ = self.fixture(Path(temporary))
            for value in (0,-1,True,1001):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    summarize(journal,model,max_bursts=value)

    def test_missing_evidence_files_and_overflow_numbers_fail_closed(self):
        for filename in ('manifest.json','completion.json','frames.jsonl','frames.idx'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                journal, model, _, _, _, _ = self.fixture(Path(temporary))
                (journal/filename).unlink()
                with self.assertRaises(OSError): audit_journal(journal,model)
        with tempfile.TemporaryDirectory() as temporary:
            journal, model, _, _, _, rows = self.fixture(Path(temporary))
            rows[2]['ball']['x'] = 10**400
            self.write_rows(journal, rows)
            with self.assertRaisesRegex(ValueError,'ball x'): audit_journal(journal,model)


if __name__ == '__main__': unittest.main()
