import ast
from pathlib import Path
import unittest

import torch
from torch import nn

from skip_deepstack import SkipDeepStackMergers, require_skip_visual_result


class Counted(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * self.scale


class FakeVision(nn.Module):
    """CPU stand-in mirroring the source's block/DS/main control flow."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Counted(i + 1) for i in range(5)])
        self.deepstack_visual_indexes = [0, 2, 4]
        self.deepstack_merger_list = nn.ModuleList([Counted(11), Counted(12), Counted(13)])
        self.merger = Counted(17)

    def forward(self, hidden_states):
        streams = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if layer_num in self.deepstack_visual_indexes:
                i = self.deepstack_visual_indexes.index(layer_num)
                streams.append(self.deepstack_merger_list[i](hidden_states))
        return self.merger(hidden_states), streams


class RejectForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.reject_patch = True

    def __setattr__(self, name, value):
        if name == 'forward' and self.__dict__.get('reject_patch'):
            raise RuntimeError('synthetic patch failure')
        super().__setattr__(name, value)

    def forward(self, x): return x


class SkipTests(unittest.TestCase):
    def test_fake_main_path_and_real_ds_calls(self):
        visual = FakeVision()
        image = torch.arange(4, dtype=torch.float32)
        drop_primary, drop_streams = visual(image)
        self.assertTrue(all(x is not None for x in drop_streams))
        self.assertEqual([x.calls for x in visual.deepstack_merger_list], [1, 1, 1])
        with SkipDeepStackMergers(visual, audit_calls=True) as skip:
            skip_primary, skipped_streams = visual(image)
            self.assertIs(require_skip_visual_result((skip_primary, skipped_streams)), skip_primary)
            self.assertEqual(skip.skipped_calls, [1, 1, 1])
            self.assertEqual([x.calls for x in visual.deepstack_merger_list], [1, 1, 1])
            self.assertTrue(torch.equal(drop_primary, skip_primary))
            self.assertTrue(all(x is None for x in skipped_streams))
        self.assertEqual([x.calls for x in visual.blocks], [2] * 5)
        self.assertEqual(visual.merger.calls, 2)
        restored_primary, restored_streams = visual(image)
        self.assertTrue(torch.equal(drop_primary, restored_primary))
        self.assertTrue(all(x is not None for x in restored_streams))
        self.assertEqual([x.calls for x in visual.deepstack_merger_list], [2, 2, 2])

    def test_cleanup_on_exception_and_nested_context_rejected(self):
        visual = FakeVision()
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with SkipDeepStackMergers(visual):
                with self.assertRaisesRegex(RuntimeError, 'another DS skip'):
                    with SkipDeepStackMergers(visual): pass
                raise RuntimeError('synthetic failure')
        self.assertTrue(all('forward' not in x.__dict__ for x in visual.deepstack_merger_list))
        self.assertTrue(all(x is not None for x in visual(torch.ones(1))[1]))

    def test_preexisting_instance_forward_restored_exactly(self):
        visual = FakeVision()
        prior = lambda x: x + 5
        visual.deepstack_merger_list[0].forward = prior
        with SkipDeepStackMergers(visual):
            self.assertIsNone(visual(torch.ones(1))[1][0])
        self.assertIs(visual.deepstack_merger_list[0].__dict__['forward'], prior)

    def test_partial_patch_failure_rolls_back(self):
        visual = FakeVision()
        visual.deepstack_merger_list[1] = RejectForward()
        with self.assertRaisesRegex(RuntimeError, 'synthetic patch failure'):
            with SkipDeepStackMergers(visual): pass
        self.assertTrue(all('forward' not in x.__dict__ for x in visual.deepstack_merger_list))
        self.assertTrue(all(SkipDeepStackMergers._OWNER not in x.__dict__ for x in visual.deepstack_merger_list))
        self.assertTrue(all(x is not None for x in visual(torch.ones(1))[1]))

    def test_rejects_alias_invalid_indexes_and_unexpected_output(self):
        visual = FakeVision()
        visual.deepstack_merger_list[0] = visual.merger
        with self.assertRaisesRegex(ValueError, 'aliases'):
            with SkipDeepStackMergers(visual): pass
        visual = FakeVision()
        visual.deepstack_merger_list[0] = visual.blocks[1]
        with self.assertRaisesRegex(ValueError, 'aliases'):
            with SkipDeepStackMergers(visual): pass
        visual = FakeVision(); visual.deepstack_visual_indexes[-1] = 6
        with self.assertRaisesRegex(ValueError, 'outside'):
            with SkipDeepStackMergers(visual): pass
        for result in [(None, [None] * 3), (torch.ones(1), [None] * 2), (torch.ones(1), [None, None, 1])]:
            with self.assertRaises(ValueError): require_skip_visual_result(result)

    def test_fixed_author_vision_structure_remains_compatible(self):
        root = Path('/media/disk2/YZX/research/EADP_v2')
        source = root / '_reproduction/envs/qwen_author/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py'
        module = ast.parse(source.read_text())
        cls = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == 'Qwen3VLVisionModel')
        forward = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'forward')
        attributes = {x.attr for x in ast.walk(forward) if isinstance(x, ast.Attribute)}
        self.assertTrue({'blocks', 'deepstack_merger_list', 'deepstack_visual_indexes', 'merger'} <= attributes)
        returns = [x for x in ast.walk(forward) if isinstance(x, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertIsInstance(returns[0].value, ast.Tuple)
        self.assertEqual([x.id for x in returns[0].value.elts], ['hidden_states', 'deepstack_feature_lists'])
        model = root / '_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'
        tree = ast.parse(model.read_text())
        unwrap = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'unwrap_visual_output')
        self.assertTrue(any(isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and x.value.id == 'output' for x in ast.walk(unwrap)))

    def test_random_tiny_actual_qwen_vision_cpu_only(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = Qwen3VLVisionConfig(depth=3, hidden_size=32, intermediate_size=64, num_heads=4,
            in_channels=3, patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=16, num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2],
            attn_implementation='sdpa')
        visual = Qwen3VLVisionModel(config).cpu().eval()
        pixels = torch.randn(16, 24)
        grid = torch.tensor([[1, 4, 4]])
        calls = {'blocks': [0, 0, 0], 'main': 0}
        handles = []
        def count_block(i):
            def hook(*_args): calls['blocks'][i] += 1
            return hook
        def count_main(*_args): calls['main'] += 1
        try:
            for i, block in enumerate(visual.blocks): handles.append(block.register_forward_hook(count_block(i)))
            handles.append(visual.merger.register_forward_hook(count_main))
            with torch.inference_mode():
                reference, streams = visual(pixels, grid_thw=grid)
                with SkipDeepStackMergers(visual, audit_calls=True) as skip:
                    result = visual(pixels, grid_thw=grid)
                again, restored = visual(pixels, grid_thw=grid)
            self.assertTrue(torch.equal(reference, require_skip_visual_result(result)))
            self.assertTrue(torch.equal(reference, again))
            self.assertEqual(skip.skipped_calls, [1, 1, 1])
            self.assertTrue(all(x is not None for x in streams + restored))
            self.assertEqual(calls, {'blocks': [3, 3, 3], 'main': 3})
        finally:
            for handle in handles: handle.remove()


if __name__ == '__main__': unittest.main()
