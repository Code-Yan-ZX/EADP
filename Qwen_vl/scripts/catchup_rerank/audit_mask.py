"""Verify the native single-prune reference's sliced mask is causal on CPU."""
import json
from pathlib import Path
import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", local_files_only=True).text_config
config._attn_implementation = "sdpa"
cache = DynamicCache(config=config)
x = torch.zeros(1, 4, 1024, 128)
cache.update(x, x, 0)
n = 280
mask = create_causal_mask(config, input_embeds=torch.zeros(1, n, 4096),
    attention_mask=torch.ones(1, n, dtype=torch.long), cache_position=torch.arange(n),
    past_key_values=cache, position_ids=torch.arange(n)[None])
trimmed = mask[0, 0, :, :n]
ok = torch.equal(trimmed, torch.ones(n, n, dtype=torch.bool).tril())
result = dict(passed=bool(ok), returned_shape=list(mask.shape), returned_dtype=str(mask.dtype),
              actual_key_length=n, after_sdpa_slice_shape=list(trimmed.shape),
              comparison="identical to causal lower triangle; unpadded prefill only")
out = Path(__file__).resolve().parents[2] / "outputs/catchup_rerank/audit_mask.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not ok:
    raise AssertionError("Mask mismatch")
