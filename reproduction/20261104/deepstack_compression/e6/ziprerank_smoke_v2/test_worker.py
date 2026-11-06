"""CPU-only positive and adversarial checks for the diagnostic acceptance logic."""
import copy
import unittest
from types import SimpleNamespace

import torch
from worker import cache_summary, diagnostic_checks, preflight


def sample(name='keep_256'):
    original, kept, layers = 1052, (284 if name == 'keep_256' else 1052), 36
    trace = [{'position_ids': [[[100 + i]]] * 3} for i in range(33)]
    logs = [{'finite': True, 'all_layer_cache_after': [
        {'key_shape': [1, 8, kept + i, 128], 'value_shape': [1, 8, kept + i, 128]}
        for _ in range(layers)]} for i in range(33)]
    fps = [{'call_index': i, 'layers': [{'key_prefix_sha256': str(j), 'value_prefix_sha256': str(j)}
                                         for j in range(layers)]} for i in (0, 32)]
    return {'name': name, 'status': 'completed', 'forced_tokens_match': True, 'generated_token_count': 33,
            'original_sequence_length': original, 'original_max_position': 100, 'language_model_calls': trace,
            'logits_trace': logs, 'cache_fingerprints': fps,
            'prefill_checks': {'positions_match': True, 'image_main_matches': True, 'text_embeddings_match': True,
                               'ds_matches': [True] * 3, 'ds_count': 3, 'expected_ds_count': 3,
                               'selected_unique': True, 'selected_in_range': True,
                               'selected_count': 256 if name == 'keep_256' else 1024},
            'keep_all_vs_disabled_logit_checks': [{'comparable': True,
                'allclose_atol_0.01_rtol_0.01': True, 'argmax_equal': True} for _ in range(33)]}


class TestDiagnostics(unittest.TestCase):
    def test_complete_paths(self):
        for name in ('disabled', 'keep_all', 'keep_256'):
            self.assertTrue(all(diagnostic_checks(sample(name), 36).values()))

    def test_32_is_not_32_decode_steps(self):
        x = sample(); x['language_model_calls'].pop(); x['logits_trace'].pop()
        self.assertFalse(diagnostic_checks(x, 36)['one_prefill_plus_32_decode_calls'])

    def test_wrong_rope_and_ds(self):
        x = sample(); x['language_model_calls'][7]['position_ids'][0][0][0] = 999
        x['prefill_checks']['ds_matches'][1] = False
        checks = diagnostic_checks(x, 36)
        self.assertFalse(checks['decode_positions']); self.assertFalse(checks['prefill_inputs_and_three_ds_match'])

    def test_wrong_cache_layer_and_changed_prefix(self):
        x = sample(); x['logits_trace'][21]['all_layer_cache_after'][35]['value_shape'][-2] -= 1
        x['cache_fingerprints'][1]['layers'][0]['key_prefix_sha256'] = 'bad'
        checks = diagnostic_checks(x, 36)
        self.assertFalse(checks['all_layer_cache_lengths']); self.assertFalse(checks['all_layer_prefix_unchanged'])

    def test_nonfinite_missing_logits(self):
        x = sample('keep_all'); x['logits_trace'][5]['finite'] = False
        x['keep_all_vs_disabled_logit_checks'].pop()
        checks = diagnostic_checks(x, 36)
        self.assertFalse(checks['all_logits_finite']); self.assertFalse(checks['keep_all_logits_match_disabled'])

    def test_real_tensor_fingerprints(self):
        k = torch.arange(24).reshape(1, 2, 3, 4)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=k, values=k.clone())])
        first = cache_summary(cache, fingerprint_prefix=2)
        cache.layers[0].keys = torch.cat([k, k], dim=-2)
        cache.layers[0].values = torch.cat([k, k], dim=-2)
        second = cache_summary(cache, fingerprint_prefix=2)
        self.assertEqual(first[0]['key_prefix_sha256'], second[0]['key_prefix_sha256'])
        self.assertNotEqual(first[0]['key_shape'], second[0]['key_shape'])
        with self.assertRaises(ValueError): cache_summary(cache, fingerprint_prefix=100)

    def test_preflight_fails_on_source_mismatch(self):
        with self.assertRaises(ValueError): preflight({'sha256': {__file__: 'bad'}})


if __name__ == '__main__': unittest.main()
