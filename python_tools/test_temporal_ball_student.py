"""CPU contract tests using the existing local YOLO11n checkpoint; no downloads."""
import copy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from temporal_ball_student import CausalWindow, GEOMETRY, TemporalBallStudent, loss, reviewed_sample

ROOT = Path(__file__).resolve().parents[1]


class TemporalStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(3407)
        torch.set_num_threads(2)
        cls.model = TemporalBallStudent(ROOT / 'yolo11n.pt').eval()

    def data(self, present=True):
        x = torch.rand(1, 3, 3, 64, 96)
        target = torch.zeros(1, 1, 64, 96)
        if present:
            target[0, 0, 20:23, 30:33] = 1
        return x, {'heatmap': target, 'visibility': torch.tensor([[float(present)]])}

    def test_uses_actual_local_yolo_blocks_and_dense_current_frame_heads(self):
        self.assertEqual([type(m).__name__ for m in self.model.encoder], ['Conv', 'Conv', 'C3k2', 'Conv', 'C3k2'])
        self.assertEqual(self.model.provenance['temporal_offsets'], [-2, -1, 0])
        x, target = self.data()
        with torch.inference_mode():
            out = self.model(x)
        self.assertEqual(tuple(out['heatmap_logits'].shape), (1, 1, 64, 96))
        self.assertEqual(tuple(out['visibility_logits'].shape), (1, 1))
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
        self.assertTrue(torch.isfinite(loss(out, target)['total']))

    def test_missing_local_checkpoint_rejected_before_ultralytics_download(self):
        with self.assertRaises(ValueError):
            TemporalBallStudent(ROOT / 'not_a_checkpoint.pt')

    def test_checkpoint_mutation_during_loading_is_rejected_before_provenance(self):
        # Only a disposable local copy is altered. A post-load-only hash would
        # incorrectly accept/report its altered bytes for a different model.
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'yolo11n.pt'
            shutil.copyfile(ROOT / 'yolo11n.pt', checkpoint)
            def changing_loader(path):
                Path(path).write_bytes(Path(path).read_bytes() + b'TEST_ONLY_MUTATION')
                return SimpleNamespace(model=SimpleNamespace())
            with patch('ultralytics.YOLO', side_effect=changing_loader):
                with self.assertRaisesRegex(RuntimeError, 'changed during model loading'):
                    TemporalBallStudent(checkpoint)

    def test_rejects_bad_input_history_values_and_geometry(self):
        bad = [torch.rand(1, 5, 3, 64, 96), torch.rand(1, 9, 64, 96),
               torch.rand(1, 3, 3, 63, 96), torch.zeros(1, 3, 3, 64, 96, dtype=torch.int64),
               torch.full((1, 3, 3, 64, 96), float('nan')), torch.full((1, 3, 3, 64, 96), 1.1)]
        for value in bad:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                self.model(value)

    def test_history_is_used_and_order_matters(self):
        # Same current frame: replacing/reordering history changes model output.
        x = torch.zeros(1, 3, 3, 64, 96)
        x[:, 1] = .7
        x[:, 2] = .2
        swapped = x[:, [1, 0, 2]]
        no_history = x.clone(); no_history[:, :2] = x[:, 2:]
        with torch.inference_mode():
            initial = self.model(x)['heatmap_logits']
            self.assertGreater(float((initial - self.model(swapped)['heatmap_logits']).abs().max()), 1e-7)
            self.assertGreater(float((initial - self.model(no_history)['heatmap_logits']).abs().max()), 1e-7)

    def test_frozen_encoder_stats_unchanged_and_every_new_head_receives_gradient(self):
        self.model.train()
        before = {n: p.detach().clone() for n, p in self.model.encoder.state_dict().items()}
        self.model.zero_grad(set_to_none=True)
        x, target = self.data()
        loss(self.model(x), target)['total'].backward()
        for name, param in self.model.named_parameters():
            if name.startswith('encoder.'):
                self.assertFalse(param.requires_grad)
                self.assertIsNone(param.grad)
            else:
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(float(param.grad.abs().sum()), 0, name)
        for name, value in self.model.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.model.eval()

    def test_verified_blank_target_is_valid_supervision(self):
        x, target = self.data(False)
        parts = loss(self.model(x), target)
        self.assertGreater(float(parts['heatmap'].detach()), 0)
        self.assertGreater(float(parts['visibility'].detach()), 0)

    def test_rejects_unknown_conflicting_nonfinite_and_wrong_target_shape(self):
        x, good = self.data()
        out = self.model(x)
        bad = [None, {'heatmap': good['heatmap']},
               {'heatmap': good['heatmap'], 'visibility': torch.zeros(1, 1)},
               {'heatmap': good['heatmap'], 'visibility': torch.full((1, 1), .5)},
               {'heatmap': torch.full_like(good['heatmap'], float('nan')), 'visibility': good['visibility']},
               {'heatmap': torch.zeros(1, 3, 64, 96), 'visibility': good['visibility']}]
        for target in bad:
            with self.subTest(target_type=type(target).__name__), self.assertRaises(ValueError):
                loss(out, target)

    def test_stream_has_no_future_frame_and_returned_data_cannot_mutate_history(self):
        stream = CausalWindow()
        frames = [torch.full((3, 64, 96), i / 10) for i in range(1, 5)]
        self.assertIsNone(stream.push('session', 1, .01, frames[0]))
        self.assertIsNone(stream.push('session', 2, .02, frames[1]))
        x, context = stream.push('session', 3, .03, frames[2])
        self.assertEqual(context['source_frame_ids'], [1, 2, 3])
        self.assertTrue(torch.equal(x[0], torch.stack(frames[:3])))
        saved = x.clone()
        frames[1].fill_(.9)
        later, context = stream.push('session', 4, .04, frames[3])
        self.assertTrue(torch.equal(x, saved))
        self.assertEqual(context['source_frame_ids'], [2, 3, 4])
        self.assertTrue(torch.all(later[0, 0] == .2))

    def test_stream_clears_on_gap_seek_source_timestamp_or_geometry_change(self):
        rgb = torch.zeros(3, 64, 96)
        cases = [('session', 4, .04, rgb), ('session', 1, .04, rgb),
                 ('other', 3, .03, rgb), ('session', 3, .02, rgb),
                 ('session', 3, 10., rgb),
                 ('session', 3, .03, torch.zeros(3, 96, 96))]
        for source, fid, timestamp, image in cases:
            stream = CausalWindow()
            stream.push('session', 1, .01, rgb); stream.push('session', 2, .02, rgb)
            with self.subTest(source=source, fid=fid, timestamp=timestamp):
                self.assertIsNone(stream.push(source, fid, timestamp, image))
                self.assertEqual(len(stream.frames), 1)
        stream.reset()
        self.assertEqual(len(stream.frames), 0)

    def test_stream_gap_bound_is_explicit_configurable_and_validated(self):
        for gap in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                CausalWindow(max_gap_s=gap)
        stream = CausalWindow(max_gap_s=.03)
        rgb = torch.zeros(3, 64, 96)
        stream.push('source', 1, .01, rgb)
        self.assertIsNone(stream.push('source', 2, .05, rgb))
        self.assertEqual(len(stream.frames), 1)

    def staging_sample(self):
        raw = torch.rand(3, 3, 288, 512)
        mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        std = torch.tensor([.229, .224, .225])[None, :, None, None]
        x = ((raw - mean) / std).reshape(9, 288, 512)
        targets = torch.zeros(3, 288, 512); targets[0, 20, 30] = 1; targets[-1, 25, 40] = 1
        provenance = {'partition': 'unassigned', 'clip': 'reviewed', 'target_method': 'continuous_binary_capsule_v1',
                      'frames': [{'source_frame': i + 10, 'elapsed_ms': float(i + 1),
                                  'output_target': {'visibility': 'absent' if i == 1 else 'visible'}} for i in range(3)]}
        return raw, x, {0: targets}, provenance

    def test_staging_denormalizes_rgb_and_uses_only_latest_reviewed_target(self):
        raw, x, targets, provenance = self.staging_sample()
        original = targets[0].clone()
        rgb, supervision, context = reviewed_sample(x, targets, provenance)
        torch.testing.assert_close(rgb[0], raw, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(supervision['heatmap'][0, 0], original[-1]))
        self.assertTrue(torch.equal(targets[0], original))
        self.assertEqual(context['source_frame_ids'], [[10, 11, 12]])
        self.assertEqual(context['geometry'], GEOMETRY)
        supervision['heatmap'].zero_()
        self.assertTrue(torch.equal(targets[0], original))

    def test_staging_rejects_unreviewed_nonconsecutive_or_bad_normalized_input(self):
        _, x, targets, provenance = self.staging_sample()
        cases = []
        unknown = copy.deepcopy(provenance); unknown['frames'][-1]['output_target']['visibility'] = 'unknown'; cases.append((x, targets, unknown))
        gap = copy.deepcopy(provenance); gap['frames'][-1]['source_frame'] = 20; cases.append((x, targets, gap))
        time_gap = copy.deepcopy(provenance); time_gap['frames'][-1]['elapsed_ms'] = 300.; cases.append((x, targets, time_gap))
        wrong_split = copy.deepcopy(provenance); wrong_split['partition'] = 'test'; cases.append((x, targets, wrong_split))
        cases.append((torch.full_like(x, 99), targets, provenance))
        conflict = copy.deepcopy(provenance); conflict['frames'][-1]['output_target']['visibility'] = 'absent'; cases.append((x, targets, conflict))
        for a, b, c in cases:
            with self.assertRaises(ValueError):
                reviewed_sample(a, b, c)

    def teacher(self, target):
        context = {'source_ids': ['reviewed'], 'source_frame_ids': [[10, 11, 12]], 'geometry': GEOMETRY}
        signal = {'heatmap': target['heatmap'].clone(), 'visibility': target['visibility'].clone(),
                  'model_sha256': 'a' * 64, 'temporal_offsets': [-2, -1, 0], **context}
        return signal, context

    def test_optional_teacher_is_auxiliary_signal_with_nonzero_head_gradient(self):
        self.model.zero_grad(set_to_none=True)
        x, target = self.data()
        teacher, context = self.teacher(target)
        out = self.model(x)
        parts = loss(out, target, teacher=teacher, teacher_weight=.1, context=context)
        self.assertGreater(float(parts['distillation'].detach()), 0)
        self.assertTrue(torch.isfinite(parts['total']))
        parts['total'].backward()
        self.assertGreater(float(self.model.heatmap.weight.grad.norm()), 0)

    def test_distillation_rejects_lookahead_misalignment_missing_or_test_teacher(self):
        x, target = self.data()
        out = self.model(x)
        signal, context = self.teacher(target)
        cases = []
        for key, value in [('source_ids', ['other']), ('source_frame_ids', [[11, 12, 13]]),
                           ('geometry', 'letterbox'), ('temporal_offsets', [0, 1, 2]), ('model_sha256', '')]:
            changed = copy.deepcopy(signal); changed[key] = value; cases.append(changed)
        gradient = copy.deepcopy(signal); gradient['heatmap'].requires_grad_(True); cases.append(gradient)
        bad_probability = copy.deepcopy(signal); bad_probability['heatmap'].fill_(2); cases.append(bad_probability)
        for teacher in cases:
            with self.assertRaises(ValueError):
                loss(out, target, teacher=teacher, teacher_weight=.1, context=context)
        for purpose in ('validation', 'test'):
            with self.assertRaises(ValueError):
                loss(out, target, teacher=signal, teacher_weight=.1, context=context, purpose=purpose)
        with self.assertRaises(ValueError):
            loss(out, target, teacher_weight=.1)
        with self.assertRaises(ValueError):
            loss(out, target, teacher=signal, teacher_weight=.1,
                 context={**context, 'source_frame_ids': [[10, 11, 12], [20, 21, 22]]})


if __name__ == '__main__':
    unittest.main(verbosity=2)
