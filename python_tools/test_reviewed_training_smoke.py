"""CPU-only guards for reviewed smoke selection and supervision contracts.

These tests never load a checkpoint, initialize CUDA, save model weights, or
modify reviewed captures. A sentinel model stops before any training forward.
"""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from smoke_blurball_training import experiment, validate_examples
from smoke_reviewed_blurball_training import select_examples


class SizedDataset:
    def __init__(self, size=5, target=None):
        self.size = size
        self.target = target

    def __len__(self):
        return self.size

    def sample(self, index):
        return (torch.zeros(9, 288, 512), {0: self.target},
                {'target_method': 'TEST_ONLY_SENTINEL'})


class SentinelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        stage = torch.nn.Module()
        stage.fuse_layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        self.stage4 = torch.nn.ModuleList([stage])
        self._out_scales = [0]
        self.forward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        raise RuntimeError('TEST_ONLY_FORWARD_SENTINEL')


def window(*states):
    return {'frames': [{'target': {'visibility': state}} for state in states]}


class ReviewedTrainingSmokeTests(unittest.TestCase):
    def test_selection_is_first_complete_visible_then_first_complete_absent(self):
        rows = [window('visible', 'absent', 'visible'),
                window('partial', 'visible', 'visible'),
                window('unknown', 'unknown', 'unknown'),
                window('absent', 'absent', 'absent'),
                window('visible', 'visible', 'visible'),
                window('visible', 'visible', 'visible'),
                window('absent', 'absent', 'absent')]
        expected = [(4, 'reviewed_visible'), (3, 'reviewed_absent')]
        self.assertEqual(select_examples(rows), expected)
        self.assertEqual(select_examples(rows), expected)

    def test_selection_fails_when_a_complete_class_is_missing(self):
        cases = [[], [window('visible', 'visible', 'visible')],
                 [window('absent', 'absent', 'absent')],
                 [window('visible', 'visible', 'absent'),
                  window('absent', 'partial', 'absent')]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select_examples(rows)

    def test_selection_does_not_mutate_labels(self):
        rows = [window('visible', 'visible', 'visible'),
                window('absent', 'absent', 'absent')]
        import copy
        before = copy.deepcopy(rows)
        select_examples(rows)
        self.assertEqual(rows, before)

    def test_examples_allow_only_one_positive_and_one_absent(self):
        dataset = SizedDataset()
        for values in [[(0, 'sharp'), (3, 'absent')],
                       [(1, 'reviewed_visible'), (4, 'reviewed_absent')],
                       ((4, 'reviewed_absent'), (1, 'reviewed_visible'))]:
            with self.subTest(values=values):
                self.assertIsNone(validate_examples(dataset, values))

    def test_examples_reject_invalid_indices_and_class_combinations(self):
        dataset = SizedDataset()
        cases = [None, [], [(0, 'sharp')],
                 [(0, 'sharp'), (1, 'absent'), (2, 'absent')],
                 [(0, 'sharp'), (0, 'absent')],
                 [(True, 'sharp'), (3, 'absent')],
                 [(1.0, 'sharp'), (3, 'absent')],
                 [(-1, 'sharp'), (3, 'absent')],
                 [(5, 'sharp'), (3, 'absent')],
                 [(0, 'reviewed_visible'), (3, 'sharp')],
                 [(0, 'absent'), (3, 'reviewed_absent')],
                 [(0, 'partial'), (3, 'absent')],
                 [(0, 'sharp'), (3, 'absent', 'extra')]]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_examples(dataset, values)

    def guard_check(self, target, first_kind, expected_message, expected_forward_calls):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            for name in ('src/models/blurball.py', 'src/configs/model/blurball.yaml',
                         'src/configs/train_blur.yaml'):
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test-only source hash fixture\n', encoding='utf-8')
            checkpoint = root / 'checkpoint'
            checkpoint.write_bytes(b'TEST_ONLY_NO_NETWORK_CHECKPOINT')
            args = argparse.Namespace(repo=repo, checkpoint=checkpoint, output=root,
                                      loss='mse')
            model = SentinelModel()
            other = 'reviewed_absent' if first_kind in ('sharp', 'reviewed_visible') else 'reviewed_visible'
            # CUDA entry points and Tensor.cuda are patched to CPU identities.
            # The sentinel prevents forward/backward and any optimizer step.
            with patch('smoke_blurball_training.torch.manual_seed'), \
                 patch('smoke_blurball_training.torch.cuda.is_available', return_value=True), \
                 patch('smoke_blurball_training.torch.cuda.get_device_name', return_value='TEST_ONLY_CPU'), \
                 patch.object(torch.Tensor, 'cuda', lambda tensor: tensor), \
                 patch('smoke_blurball_training.build_network', return_value=model), \
                 self.assertRaisesRegex(RuntimeError, expected_message):
                experiment(args, {}, dataset=SizedDataset(target=target),
                           examples=[(0, first_kind), (1, other)],
                           dataset_scope='TEST ONLY; no data or training')
            self.assertEqual(model.forward_calls, expected_forward_calls)

    def test_positive_requires_nonempty_target_in_every_channel_before_forward(self):
        target = torch.zeros(3, 288, 512)
        target[0, 10, 10] = 1
        target[2, 11, 11] = 1
        for kind in ('sharp', 'reviewed_visible'):
            with self.subTest(kind=kind):
                self.guard_check(target, kind, 'Every reviewed positive frame', 0)

    def test_absence_requires_exactly_blank_targets_before_forward(self):
        target = torch.zeros(3, 288, 512)
        target[1, 10, 10] = 0.001
        for kind in ('absent', 'reviewed_absent'):
            with self.subTest(kind=kind):
                self.guard_check(target, kind, 'Reviewed absent frames must', 0)

    def test_valid_targets_reach_forward_guard_without_gpu_training(self):
        target = torch.zeros(3, 288, 512)
        target[:, 10, 10] = 1
        self.guard_check(target, 'reviewed_visible', 'TEST_ONLY_FORWARD_SENTINEL', 1)
        self.guard_check(torch.zeros_like(target), 'reviewed_absent',
                         'TEST_ONLY_FORWARD_SENTINEL', 1)


if __name__ == '__main__':
    unittest.main()
