"""CPU-only tiny random Qwen3-VL text-model path, no model weights or GPU."""
import json
import traceback
import torch
import transformers
from transformers import Qwen3VLTextConfig
from hilo_prune.models.qwen3_vl.modeling_hilo_qwen3_vl import Qwen3VLTextModel

torch.manual_seed(0)
config = Qwen3VLTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=4, num_attention_heads=4,
                           num_key_value_heads=4, head_dim=8, use_cache=False)
config.rope_parameters = {'rope_type': 'default', 'rope_theta': 5000000.0,
                          'mrope_section': [2, 1, 1]}
config._attn_implementation = 'eager'
config.hilo_keep_ratio = .5
config.hilo_candidate_ratio = .75
config.hilo_prune_layer = 2
model = Qwen3VLTextModel(config).eval()
embeds = torch.randn(1, 8, 32)
ids = torch.tensor([[1, 9, 9, 9, 9, 9, 9, 2]])
mask = ids.eq(9)
deep = [torch.randn(6, 32) for _ in range(3)]
patches = torch.randn(6, 12)
print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__,
                  'device': str(embeds.device), 'weights': 'random tiny model',
                  'config': 'eager, 4 decoder layers, prune layer 2, 6 visual tokens, 3 DS streams'}))
for label, image in [('no_selection_control', None), ('selection_on', patches)]:
    print('CASE', label, flush=True)
    try:
        with torch.no_grad():
            out = model(inputs_embeds=embeds.clone(), input_ids_=ids, image_token_id=9,
                        image_tensor=image, visual_pos_masks=mask,
                        deepstack_visual_embeds=deep, use_cache=False)
        print('OK shape=', tuple(out.last_hidden_state.shape), flush=True)
    except Exception:
        traceback.print_exc()
