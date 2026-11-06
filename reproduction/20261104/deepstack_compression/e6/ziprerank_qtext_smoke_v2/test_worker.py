"""CPU-only positive and adversarial checks for the diagnostic acceptance logic."""
import copy
import unittest
from types import SimpleNamespace

import torch
from worker import cache_summary, configure_rope_state, diagnostic_checks, preflight
from selector_trace import traced_selector


def sample(name='keep_256'):
    original, kept, layers = 1052, (284 if name in ('keep_256','question_256') else 1052), 36
    trace = [{'position_ids': [[[100 + i]]] * 3, 'input_length': kept if i == 0 else 1,
              'cache_position': list(range(original)) if i == 0 else [original+i-1],
              'all_layer_cache_before': [] if i == 0 else [
                  {'key_shape': [1, 8, kept+i-1, 128], 'value_shape': [1, 8, kept+i-1, 128]}
                  for _ in range(layers)]} for i in range(33)]
    logs = [{'finite': True, 'rope_deltas': [[-900]], 'model_rope_deltas': [[-900]], 'all_layer_cache_after': [
        {'key_shape': [1, 8, kept + i, 128], 'value_shape': [1, 8, kept + i, 128]}
        for _ in range(layers)]} for i in range(33)]
    fps = [{'call_index': i, 'layers': [{'key_prefix_sha256': str(j), 'value_prefix_sha256': str(j)}
                                         for j in range(layers)]} for i in (0, 32)]
    return {'name': name, 'status': 'completed', 'forced_tokens_match': True, 'generated_token_count': 33,
            'original_sequence_length': original, 'original_max_position': 100, 'original_rope_deltas': [[-900]], 'language_model_calls': trace, 'selector_restored': True,
            'prompt_ids_unchanged':True, 'prompt_ids_sha256':'p', 'user_positions_sha256':'q', 'user_token_ids_sha256':'r',
            'selection_trace':({'calls':1,'input_ids_sha256':'p','positions_sha256':'q','selected_ids_sha256':'r',
                                'returned_exact_selected_embeddings':True,'returned_embedding_sha256':'a'*64}
                               if name=='question_256' else {'calls':0}),
            'logits_trace': logs, 'cache_fingerprints': fps,
            'prefill_checks': {'positions_match': True, 'image_main_matches': True, 'text_embeddings_match': True,
                               'ds_matches': [True] * 3, 'ds_count': 3, 'expected_ds_count': 3,
                               'selected_unique': True, 'selected_in_range': True,
                               'selected_count': 256 if name in ('keep_256','question_256') else 1024},
            'keep_all_vs_disabled_logit_checks': [{'comparable': True,
                'allclose_atol_0.01_rtol_0.01': True, 'argmax_equal': True} for _ in range(33)]}


class TestDiagnostics(unittest.TestCase):
    def test_complete_paths(self):
        for name in ('disabled', 'keep_all', 'keep_256', 'question_256'):
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

    def test_missing_rope_bad_cache_position_and_before(self):
        x = sample(); x['logits_trace'][7]['model_rope_deltas'] = None
        x['language_model_calls'][9]['cache_position'] = [8]
        x['language_model_calls'][3]['all_layer_cache_before'][2]['key_shape'][-2] -= 1
        checks = diagnostic_checks(x, 36)
        self.assertFalse(checks['model_and_output_rope_deltas'])
        self.assertFalse(checks['input_and_cache_positions'])
        self.assertFalse(checks['all_layer_cache_before_decode'])

    def test_reject_uninvoked_or_wrong_user_selector(self):
        x=sample('question_256');x['selection_trace']['calls']=0
        self.assertFalse(diagnostic_checks(x,36)['user_text_selector_called_once'])
        x=sample('question_256');x['selection_trace']['positions_sha256']='bad'
        self.assertFalse(diagnostic_checks(x,36)['user_text_selector_called_once'])
        x=sample('disabled');x['selection_trace']['calls']=1;x['prompt_ids_unchanged']=False
        result=diagnostic_checks(x,36);self.assertFalse(result['user_text_selector_not_installed']);self.assertFalse(result['prompt_ids_unchanged'])

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

    def test_traced_selector_checks_actual_call_and_restores(self):
        class Tiny:
            def _extract_query_text_embeds(self,*_a,**_k):return 'original'
        model=Tiny();ids=torch.tensor([[8,9,10,11]]);positions=torch.tensor([2,3]);embeds=torch.arange(16).reshape(1,4,4)
        with traced_selector(model,ids,positions) as trace:
            self.assertEqual(trace['calls'],0)
            output=model._extract_query_text_embeds(embeds,ids)
            self.assertTrue(torch.equal(output,embeds[0,positions]));self.assertEqual(trace['calls'],1)
            self.assertTrue(trace['returned_exact_selected_embeddings']);self.assertEqual(len(trace['returned_embedding_sha256']),64)
            with self.assertRaisesRegex(ValueError,'changed'):model._extract_query_text_embeds(embeds,ids+1)
        self.assertEqual(model._extract_query_text_embeds(),'original')

    def test_explicit_rope_state_adapter_is_scoped_and_cloned(self):
        obj = SimpleNamespace(model=SimpleNamespace(rope_deltas='stale'))
        delta = torch.tensor([[-992]])
        configure_rope_state(obj, False, delta)
        self.assertIsNone(obj.model.rope_deltas)
        configure_rope_state(obj, True, delta)
        self.assertTrue(torch.equal(obj.model.rope_deltas, delta))
        self.assertNotEqual(obj.model.rope_deltas.data_ptr(), delta.data_ptr())


if __name__ == '__main__': unittest.main()
