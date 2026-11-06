"""CPU hook contract; not evidence that the real model has been exercised."""
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from model_adapter import InterventionAdapter, BudgetAdapter


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        x = torch.arange(256, dtype=torch.float32).reshape(256, 1)
        self.original = [torch.cat((x + i * 1000, x * 2 + i), 1) for i in range(3)]
        self.mask = torch.tensor([[False, True, True]])
        self.position = torch.tensor([[[0, 1, 2]]])
        self.embed = torch.ones((1, 3, 2))

    def invoke(self, family, arm):
        adapter = InterventionAdapter(torch, None, None, family, arm, '0' * 64)
        adapter.state.update(selected_deepstack=self.original, prefill_position_ids=self.position,
                             intended_ds_mask=self.mask, adjusted_delta=torch.zeros((1, 1)))
        adapter.w = type('W', (), {'model': type('M', (), {'model': type('X', (), {'rope_deltas': torch.zeros((1, 1))})()})()})()
        kwargs = dict(inputs_embeds=self.embed, attention_mask=torch.ones((1, 3)),
                      position_ids=self.position, visual_pos_masks=self.mask,
                      deepstack_visual_embeds=self.original)
        with patch.object(BudgetAdapter, '_prefill', return_value=((), kwargs)):
            _, actual = adapter._prefill(None, (), kwargs)
        return adapter, actual

    def test_all_slot_patterns_and_observed_injection(self):
        for bits in map(''.join, itertools.product('01', repeat=3)):
            a, kw = self.invoke('E2', bits)
            for i, bit in enumerate(bits):
                expected = self.original[i] if bit == '1' else torch.zeros_like(self.original[i])
                self.assertTrue(torch.equal(kw['deepstack_visual_embeds'][i], expected))
                self.assertTrue(torch.equal(a.state['selected_before_intervention'][i], self.original[i]))
            a._observe_prefill(None, (), kw)
            self.assertTrue(a.state['actual_injection_verified'])
            self.assertTrue(torch.equal(kw['position_ids'], self.position))

    def test_perm_mean_and_corrupted_actual_stream(self):
        for mode in ('OFF', 'MATCH', 'PERM', 'MEAN'):
            a, kw = self.invoke('E3', mode)
            if mode == 'PERM':
                indices = a.state['intervention_metadata']['permutation']
                self.assertTrue(all(torch.equal(actual, before[indices]) for actual, before in zip(kw['deepstack_visual_embeds'], self.original)))
            if mode == 'MEAN':
                self.assertTrue(all(torch.equal(actual, before.mean(0, keepdim=True).expand_as(before)) for actual, before in zip(kw['deepstack_visual_embeds'], self.original)))
            kw['deepstack_visual_embeds'] = [x.clone() for x in kw['deepstack_visual_embeds']]
            kw['deepstack_visual_embeds'][1][0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'actual DS selected features mismatch'):
                a._observe_prefill(None, (), kw)

    def test_invalid_arm(self):
        for family, arm in (('E2', '11'), ('E3', 'BOGUS'), ('E4', '111')):
            with self.assertRaises(ValueError):
                InterventionAdapter(torch, None, None, family, arm, '0' * 64)

    def test_registered_hooks_with_fake_model(self):
        class Merger(torch.nn.Module):
            def __init__(self, offset=0):
                super().__init__(); self.offset = offset
            def forward(self, x): return x + self.offset
        class Visual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
                self.merger = Merger()
                self.deepstack_merger_list = torch.nn.ModuleList([Merger(i) for i in (10, 20, 30)])
            def forward(self, x):
                for block in self.blocks: x = block(x)
                return self.merger(x), [merger(x) for merger in self.deepstack_merger_list]
        native = types.SimpleNamespace(_greed_select_impl=lambda: (torch.arange(255, -1, -1), torch.tensor([256])))
        class Pruner(torch.nn.Module):
            def forward(self, x):
                ids, _ = native._greed_select_impl()
                return x.index_select(0, ids.sort().values), [256]
        class Language(torch.nn.Module):
            def forward(self, **kw):
                return {k: v for k, v in kw.items()}
        language = Language()
        inner = types.SimpleNamespace(language_model=language, rope_deltas=None)
        def rope(ids, grid, video, attention_mask):
            return torch.arange(ids.shape[1]).reshape(1, 1, -1).expand(3, 1, -1), torch.zeros((1, 1), dtype=torch.long)
        inner.get_rope_index = rope
        visual = Visual()
        model = types.SimpleNamespace(visual=visual, model=inner, config=types.SimpleNamespace(image_token_id=9999),
                                      get_input_embeddings=lambda: lambda ids: torch.zeros((1, ids.shape[1], 2)))
        inputs = {'input_ids': torch.tensor([[1] + [9999] * 1024 + [2]]),
                  'attention_mask': torch.ones((1, 1026), dtype=torch.long),
                  'pixel_values': torch.zeros((1, 2)), 'image_grid_thw': torch.tensor([[1, 32, 32]])}
        wrapper = types.SimpleNamespace(model=model, pruner=Pruner(), _processor_inputs=lambda: inputs)
        full_visual = torch.arange(2048, dtype=torch.float32).reshape(1024, 2)
        with InterventionAdapter(torch, wrapper, native, 'E2', '101', '0' * 64) as adapter:
            wrapper._processor_inputs()
            main, streams = visual(full_visual)
            selected, _ = wrapper.pruner(main)
            embeds = torch.cat((torch.zeros((1, 1, 2)), selected.unsqueeze(0), torch.zeros((1, 1, 2))), 1)
            result = language(inputs_embeds=embeds, attention_mask=torch.ones((1, 258), dtype=torch.long))
            self.assertTrue(adapter.verify_calls())
            self.assertTrue(torch.equal(result['deepstack_visual_embeds'][0], streams[0][:256]))
            self.assertEqual(torch.count_nonzero(result['deepstack_visual_embeds'][1]).item(), 0)
            self.assertTrue(torch.equal(result['deepstack_visual_embeds'][2], streams[2][:256]))
            self.assertTrue(torch.equal(adapter.state['selected_before_intervention'][1], streams[1][:256]))
        self.assertEqual(len(language._forward_pre_hooks), 0)

    def test_transitive_dependency_import_poison_fails_closed(self):
        here = Path(__file__).resolve().parent
        script = "import sys,types; m=types.ModuleType('skip_deepstack'); m.__file__='/tmp/evil.py'; m.SkipDeepStackMergers=object; m.require_skip_visual_result=lambda x:x; sys.modules['skip_deepstack']=m; import model_adapter"
        env = dict(os.environ, PYTHONPATH=str(here))
        result = subprocess.run([sys.executable, '-c', script], cwd=here, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unexpected skip_deepstack module', result.stderr)


if __name__ == '__main__':
    unittest.main()
