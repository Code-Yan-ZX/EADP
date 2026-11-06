"""CPU synthetic alignment test of the original PriorTR DeepStack handler."""
import importlib.util
from pathlib import Path
import torch

file = Path(__file__).resolve().parents[1] / 'PriorTR/image/Qwen3-VL/visual_token_pruning/model/deepstack_handler.py'
spec = importlib.util.spec_from_file_location('unmodified_deepstack_handler', file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
handler = module.DeepStackSyncHandler(3)
embeds = [torch.arange(12).view(6, 2) + i * 100 for i in range(3)]
visual_mask = torch.tensor([[False, True, True, True, True, True, True, False]])
kept_visual = torch.tensor([0, 2, 5])
kept_full = torch.tensor([0, 1, 3, 6, 7])
out, mask = handler.sync_after_pruning(embeds, visual_mask, kept_visual, kept_full, 1)
assert torch.equal(out[0], embeds[0])
assert all(torch.equal(out[i], embeds[i][kept_visual]) for i in (1, 2))
assert torch.equal(mask, torch.tensor([[False, True, True, True, False]]))
print('PASS CPU synthetic sync: old DS retained, future DS rows 0,2,5 and visual mask aligned')
print('not a full model forward, generation, GPU, cache or mRoPE test')
