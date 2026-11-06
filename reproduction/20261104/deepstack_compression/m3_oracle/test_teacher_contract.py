"""Tiny random Qwen3-VL text model, CPU only; not an 8B feasibility result."""
import unittest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from teacher_contract import compact_prompt, teacher_inputs, mean_reference_nll
from oracle_teacher import loss_and_gradients, loss_only


class TeacherContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        cfg = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                head_dim=8, use_cache=False,
                                rope_scaling={'rope_type':'default', 'mrope_section':[1, 1, 1]})
        cfg._attn_implementation = 'eager'
        self.model = Qwen3VLTextModel(cfg).eval()
        self.head = torch.nn.Linear(32, 64, bias=False).eval()
        for p in list(self.model.parameters()) + list(self.head.parameters()): p.requires_grad_(False)

    def test_four_actual_stream_gradients_and_teacher_alignment(self):
        prompt = torch.randn(1, 9, 32)
        visual = torch.randn(4, 32, requires_grad=True)
        ds = [torch.randn(4, 32, requires_grad=True) for _ in range(3)]
        target = torch.tensor([[11, 22, 33]])
        answer_embeds = self.model.embed_tokens(target[:, :-1]).detach()
        positions = torch.arange(9).reshape(1, 1, -1).expand(3, 1, -1)
        inputs, pos, mask = teacher_inputs(prompt, visual, 2, answer_embeds, positions, -2)
        self.assertEqual(pos[:, 0, 9:].tolist(), [[7, 8]] * 3)
        self.assertEqual(mask.nonzero()[:, 1].tolist(), [2, 3, 4, 5])
        out = self.model(inputs_embeds=inputs, position_ids=pos, attention_mask=torch.ones((1, 11)),
                         visual_pos_masks=mask, deepstack_visual_embeds=ds, use_cache=False)
        hidden = out.last_hidden_state
        loss = mean_reference_nll(hidden, self.head, 9, target)
        expected = F.cross_entropy(self.head(hidden[:, 8:]).reshape(-1, 64), target.reshape(-1))
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        for stream in [visual, *ds]:
            self.assertIsNotNone(stream.grad)
            self.assertTrue(torch.isfinite(stream.grad).all())
            self.assertGreater(torch.count_nonzero(stream.grad).item(), 0)
        self.assertTrue(all(p.grad is None for p in list(self.model.parameters()) + list(self.head.parameters())))

    def test_invalid_teacher_geometry_and_target_length(self):
        prompt = torch.zeros(1, 9, 32); visual = torch.zeros(4, 32)
        positions = torch.zeros(3, 1, 9, dtype=torch.long)
        with self.assertRaises(ValueError): teacher_inputs(prompt, visual, 7, torch.zeros(1, 2, 32), positions, 0)
        with self.assertRaises(ValueError): mean_reference_nll(torch.zeros(1, 9, 32), self.head, 9, torch.zeros(1, 0, dtype=torch.long))
        with self.assertRaises(ValueError): mean_reference_nll(torch.zeros(1, 9, 32), self.head, 9, torch.zeros(1, 2, dtype=torch.long))

    def test_compact_positions_preserve_decode_origin_and_selected_rows(self):
        ids = torch.tensor([[1, 2, 9, 9, 9, 9, 3, 4, 5]])
        p = torch.tensor([[[0, 1, 2, 2, 2, 2, 3, 4, 5]]]).expand(3, 1, -1)
        compact_ids, positions, delta, kept = compact_prompt(ids, p, 2, 4, [0, 3], -3)
        self.assertEqual(kept, [0, 1, 2, 5, 6, 7, 8])
        self.assertEqual(compact_ids.tolist(), [[1, 2, 9, 9, 3, 4, 5]])
        self.assertEqual(delta, -1)
        self.assertEqual(compact_ids.shape[1] + delta, ids.shape[1] - 3)
        x, pos, mask = teacher_inputs(torch.zeros(1, 7, 32), torch.ones(2, 32), 2,
                                      torch.zeros(1, 2, 32), positions, delta)
        self.assertEqual(pos[:, 0, 7:].tolist(), [[6, 7]]*3)
        self.assertEqual(mask.nonzero()[:, 1].tolist(), [2, 3])
        with self.assertRaises(ValueError): compact_prompt(ids, p, 2, 4, [3, 0], -3)
        with self.assertRaises(ValueError): compact_prompt(ids, p, 2, 4, [0, 0], -3)

    def test_teacher_logits_match_incremental_cached_decode(self):
        prompt = torch.randn(1, 9, 32)
        visual = torch.randn(4, 32)
        ds = [torch.randn(4, 32) for _ in range(3)]
        target = torch.tensor([[11, 22, 33]])
        answer_embeds = self.model.embed_tokens(target[:, :-1]).detach()
        positions = torch.tensor([[[0, 1, 2, 2, 2, 2, 3, 4, 5]]]).expand(3, 1, -1)
        inputs, pos, mask = teacher_inputs(prompt, visual, 2, answer_embeds, positions, -3)
        with torch.no_grad():
            full = self.model(inputs_embeds=inputs, position_ids=pos, attention_mask=torch.ones((1, 11)),
                              visual_pos_masks=mask, deepstack_visual_embeds=ds, use_cache=False)
            first = self.model(inputs_embeds=inputs[:, :9], position_ids=pos[:, :, :9], attention_mask=torch.ones((1, 9)),
                               visual_pos_masks=mask[:, :9], deepstack_visual_embeds=ds, use_cache=True)
            hidden = [first.last_hidden_state[:, -1:]]
            cache = first.past_key_values
            for step in range(2):
                out = self.model(inputs_embeds=answer_embeds[:, step:step+1], position_ids=pos[:, :, 9+step:10+step],
                                 attention_mask=torch.ones((1, 10+step)), past_key_values=cache, use_cache=True)
                hidden.append(out.last_hidden_state)
                cache = out.past_key_values
            cached = torch.cat(hidden, dim=1)
            self.assertTrue(torch.allclose(self.head(cached), self.head(full.last_hidden_state[:, 8:]), atol=2e-6, rtol=2e-6))

    def test_four_stream_gradient_matches_directional_finite_difference(self):
        prompt = torch.randn(1, 9, 32)
        streams = [torch.randn(4, 32, requires_grad=True) for _ in range(4)]
        directions = [torch.randn_like(x) for x in streams]
        targets = torch.tensor([[11, 22, 33]])
        prefix = self.model.embed_tokens(targets[:, :-1]).detach()
        pos = torch.arange(9).reshape(1, 1, -1).expand(3, 1, -1)
        def evaluate(values):
            x, p, mask = teacher_inputs(prompt, values[0], 2, prefix, pos, 0)
            out = self.model(inputs_embeds=x, position_ids=p, attention_mask=torch.ones((1, 11)),
                             visual_pos_masks=mask, deepstack_visual_embeds=values[1:], use_cache=False)
            return mean_reference_nll(out.last_hidden_state, self.head, 9, targets)
        loss = evaluate(streams)
        gradients = torch.autograd.grad(loss, streams)
        analytic = sum((g * d).sum().item() for g, d in zip(gradients, directions))
        eps = 1e-3
        with torch.no_grad():
            plus = evaluate([x + eps*d for x, d in zip(streams, directions)]).item()
            minus = evaluate([x - eps*d for x, d in zip(streams, directions)]).item()
        numeric = (plus - minus) / (2 * eps)
        self.assertAlmostEqual(analytic, numeric, delta=7e-4 + abs(numeric)*0.02)

    def test_frozen_model_wrapper_requires_grad_only_for_visual_streams(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, lm, head):
                super().__init__();self.model = torch.nn.Module();self.model.language_model = lm;self.lm_head = head
            def get_input_embeddings(self): return self.model.language_model.embed_tokens
        wrapper = Wrapper(self.model, self.head).eval()
        ids = torch.randint(0, 64, (1, 9)); targets = torch.tensor([[11, 22, 33]])
        positions = torch.arange(9).reshape(1, 1, -1).expand(3, 1, -1)
        streams = [torch.randn(4, 32) for _ in range(4)]
        loss, leaves, gradients = loss_and_gradients(wrapper, ids, 2, streams[0], streams[1:],
                                                     positions, 0, targets, expected_visual_count=4)
        score = loss_only(wrapper, ids, 2, streams[0], streams[1:], positions, 0, targets, expected_visual_count=4)
        self.assertTrue(torch.allclose(loss, score, atol=1e-6, rtol=1e-6))
        self.assertFalse(score.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([tuple(g.shape) for g in gradients], [(4, 32)]*4)
        self.assertTrue(all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients))
        self.assertTrue(all(not x.requires_grad for x in leaves))
        wrapper.train()
        with self.assertRaisesRegex(ValueError, 'fully frozen'):
            loss_and_gradients(wrapper, ids, 2, streams[0], streams[1:], positions, 0, targets, expected_visual_count=4)

if __name__ == '__main__':
    unittest.main()
