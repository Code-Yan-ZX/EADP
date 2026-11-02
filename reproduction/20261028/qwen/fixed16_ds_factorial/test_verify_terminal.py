"""Synthetic CPU tests of the independent terminal verifier; no model or GT."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from verify_terminal import check_continuation, check_prefill, common_prefix_positions, read_archive


def sample(arm):
    ids = np.array([[3, 99, 99, 4]], dtype=np.int64)
    mask = ids == 99
    data = {'processor_input_ids': ids, 'injection_image_mask': mask,
            'visual_features': np.ones((2, 3), dtype=np.uint16)}
    for i in range(3):
        data['visual_deepstack_' + str(i)] = np.full((2, 3), i + 1, dtype=np.uint16)
    for field, value in {'inputs_embeds': np.zeros((1, 4, 3), dtype=np.uint16),
                         'position_ids': np.zeros((3, 1, 4), dtype=np.int64),
                         'attention_mask': np.ones((1, 4), dtype=np.int64),
                         'cache_position': np.arange(4)}.items():
        for prefix in ('prefill_original_', 'prefill_proposed_', 'prefill_'):
            data[prefix + field] = value.copy()
    source_on = arm.startswith('plain')
    output_on = arm in ('plain', 'eadp1024_ds_on')
    for field, value in [('visual_pos_masks', mask)] + [
        ('deepstack_' + str(i), data['visual_deepstack_' + str(i)]) for i in range(3)
    ]:
        if source_on:
            data['prefill_original_' + field] = value.copy()
        if output_on:
            data['prefill_proposed_' + field] = value.copy()
            data['prefill_' + field] = value.copy()
    return data


def trajectory(tokens, nsteps):
    data = {'output_suffix_ids': np.array([tokens], dtype=np.int64)}
    for i in range(nsteps):
        for name in ('position_ids', 'cache_position', 'rope_deltas'):
            data['continuation_%04d_%s' % (i, name)] = np.array([i], dtype=np.int64)
    record = {'continuation_steps': nsteps, 'llm_calls': nsteps + 1,
              'continuation_presence': [dict(visual_mask_present=False, deepstack_present=False) for _ in range(nsteps)]}
    return data, record


class VerifyTest(unittest.TestCase):
    def test_archive_contents_and_declared_dtype_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'item.npz'
            array = np.array([1, 2], dtype=np.int64)
            np.savez_compressed(path, sample=array)
            rec = dict(archive=str(path), archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       arrays={'sample': dict(dtype='torch.int64', shape=[2],
                                             sha256=hashlib.sha256(array.tobytes()).hexdigest())})
            self.assertTrue(np.array_equal(read_archive(rec, path)['sample'], array))
            altered = copy.deepcopy(rec)
            altered['arrays']['sample']['dtype'] = 'torch.float64'
            with self.assertRaises(AssertionError):
                read_archive(altered, path)

    def test_four_valid_prefill_arms(self):
        for arm in ('plain', 'eadp1024', 'plain_ds_off', 'eadp1024_ds_on'):
            check_prefill(sample(arm), arm, 99)

    def test_non_intervention_field_tampering_rejected(self):
        data = sample('plain_ds_off')
        data['prefill_position_ids'][0, 0, 0] = 7
        data['prefill_proposed_position_ids'][0, 0, 0] = 7
        with self.assertRaises(AssertionError):
            check_prefill(data, 'plain_ds_off', 99)

    def test_proposed_and_final_mismatch_rejected(self):
        data = sample('eadp1024_ds_on')
        data['prefill_deepstack_0'][0, 0] = 7
        with self.assertRaises(AssertionError):
            check_prefill(data, 'eadp1024_ds_on', 99)

    def test_injected_values_and_actual_image_mask_verified(self):
        data = sample('eadp1024_ds_on')
        data['processor_input_ids'][0, 1] = 3
        with self.assertRaises(AssertionError):
            check_prefill(data, 'eadp1024_ds_on', 99)
        data = sample('eadp1024_ds_on')
        data['prefill_deepstack_1'][0, 0] = 9
        data['prefill_proposed_deepstack_1'][0, 0] = 9
        with self.assertRaises(AssertionError):
            check_prefill(data, 'eadp1024_ds_on', 99)

    def test_continuation_completeness(self):
        data, record = trajectory([11, 12, 13], 2)
        check_continuation(record, data)
        del data['continuation_0001_rope_deltas']
        with self.assertRaises(AssertionError):
            check_continuation(record, data)

    def test_only_common_prefix_steps_are_compared(self):
        a, ar = trajectory([11, 12, 13, 14], 3)
        b, br = trajectory([11, 12, 99, 14], 3)
        b['continuation_0002_position_ids'][0] = 100
        expected = {'position_ids': [], 'cache_position': [], 'rope_deltas': []}
        self.assertEqual(common_prefix_positions(a, b, ar, br), (2, 2, expected))
        b['continuation_0001_position_ids'][0] = 100
        expected['position_ids'] = [1]
        self.assertEqual(common_prefix_positions(a, b, ar, br), (2, 2, expected))


if __name__ == '__main__':
    unittest.main(verbosity=2)
