"""Isolated CPU compatibility diagnostic using a tiny random Qwen3-VL.

No real weights, images, predictions or accuracy. Patch is confined to this
short-lived process. A pass cannot establish production compatibility.
"""
import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PIXELPRUNE_ENABLED'] = 'false'
os.environ['PIXELPRUNE_METHOD'] = 'pred_2d'
os.environ['PIXELPRUNE_THRESHOLD'] = '0.0'
BASE = Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression')
HERE = BASE / 'e6/pixelprune_cpu_decode'
EXTERNAL = BASE / 'e6/external_sources/pixelprune_8d347eb'
sys.path.insert(0, str(EXTERNAL))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def main():
    plan = json.loads((HERE / 'plan001.json').read_text())
    for p, expected in plan['sources'].items():
        if sha(p) != expected: raise ValueError('source mismatch: ' + p)
    import torch
    import transformers
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, LogitsProcessor
    from transformers.models.qwen3_vl import modeling_qwen3_vl as m
    from pixelprune import apply_pixelprune
    torch.manual_seed(20261104)
    cfg = Qwen3VLConfig(
        text_config={'vocab_size': 128, 'hidden_size': 32, 'intermediate_size': 64,
                     'num_hidden_layers': 3, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'head_dim': 8, 'rope_scaling': {'rope_type': 'default', 'mrope_section': [1, 1, 2]}},
        vision_config={'depth': 3, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 4,
                       'out_hidden_size': 32, 'num_position_embeddings': 16,
                       'deepstack_visual_indexes': [0, 1, 2]},
        image_token_id=121, video_token_id=122, vision_start_token_id=123, vision_end_token_id=124,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    cfg._attn_implementation = 'eager'
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    inputs = {'input_ids': torch.tensor([[3, 123, 121, 121, 121, 121, 124, 4, 5]]),
              'attention_mask': torch.ones(1, 9, dtype=torch.long),
              'pixel_values': torch.zeros(16, 3*2*16*16),
              'image_grid_thw': torch.tensor([[1, 4, 4]])}
    names = [('vision', m.Qwen3VLVisionModel, 'forward'), ('image_features', m.Qwen3VLModel, 'get_image_features'),
             ('model_forward', m.Qwen3VLModel, 'forward'), ('cg_forward', m.Qwen3VLForConditionalGeneration, 'forward')]
    before = {n: str(inspect.signature(getattr(c, a))) for n, c, a in names}
    records = []
    outputs = {}
    original_pos, delta = model.model.get_rope_index(inputs['input_ids'], inputs['image_grid_thw'], None, inputs['attention_mask'])
    original_max = int(original_pos.max())
    class Force(LogitsProcessor):
        def __call__(self, input_ids, scores):
            out = torch.full_like(scores, -float('inf')); out[:, 7] = 0; return out
    def attempt(label):
        model.model.rope_deltas = None
        prefill = []
        post = []
        def pre_hook(_m, _args, kw):
            pos = kw.get('position_ids'); vmask = kw.get('visual_pos_masks'); ds = kw.get('deepstack_visual_embeds')
            prefill.append({'positions': None if pos is None else pos.tolist(),
                'visual_count': None if vmask is None else int(vmask.sum()),
                'ds_rows': None if ds is None else [int(x.shape[0]) for x in ds]})
        def post_hook(_m, _args, _kw, out):
            cache = out.past_key_values
            post.append([{'key': list(layer.keys.shape), 'value': list(layer.values.shape)} for layer in cache.layers])
        handles = [model.model.language_model.register_forward_pre_hook(pre_hook, with_kwargs=True),
                   model.model.language_model.register_forward_hook(post_hook, with_kwargs=True)]
        try:
            with torch.inference_mode():
                result = model.generate(**inputs, do_sample=False, max_new_tokens=33, min_new_tokens=33,
                                        eos_token_id=[], use_cache=True, return_dict_in_generate=True,
                                        logits_processor=[Force()])
            ids = result.sequences.tolist()
            outputs[label] = ids
            enabled = label == 'patched_enabled_threshold_zero'
            selector_mask = torch.tensor([True, True, True, False, False, False, True, True, True])
            expected = original_pos[:, :, selector_mask] if enabled else original_pos
            first_len = 6 if enabled else 9
            checks = {'exactly_33_calls': len(prefill) == len(post) == 33,
                      'forced_tokens_33': ids[0][9:] == [7] * 33,
                      'prefill_original_positions': bool(prefill and prefill[0]['positions'] == expected.tolist()),
                      'prefill_ds_rows': bool(prefill and prefill[0]['ds_rows'] == [1 if enabled else 4] * 3),
                      'decode_positions': len(prefill) == 33 and all(
                          item['positions'] == [[[original_max + i]]] * 3 for i, item in enumerate(prefill) if i),
                      'three_layer_cache_lengths': len(post) == 33 and all(
                          len(layers) == 3 and all(x['key'][-2] == first_len+i and x['value'][-2] == first_len+i
                                                   for x in layers) for i, layers in enumerate(post))}
            records.append({'arm': label, 'status': 'completed', 'output_ids': ids,
                            'cache_length': result.past_key_values.get_seq_length(),
                            'checks': checks, 'language_model_inputs': prefill, 'layer_cache_shapes': post})
        except Exception as exc:
            records.append({'arm': label, 'status': 'failed', 'error_type': type(exc).__name__,
                            'error': str(exc), 'traceback': traceback.format_exc()})
        finally:
            for handle in handles: handle.remove()
    attempt('unpatched')
    apply_pixelprune(model='qwen3_vl')
    after = {n: str(inspect.signature(getattr(c, a))) for n, c, a in names}
    attempt('patched_explicit_disabled')
    os.environ['PIXELPRUNE_ENABLED'] = 'true'
    from pixelprune import compute_merged_keep_indices
    from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector
    keep = compute_merged_keep_indices(normalize_pixel_values_for_selector(inputs['pixel_values']),
                                       inputs['image_grid_thw'], spatial_merge_size=2)
    attempt('patched_enabled_threshold_zero')
    if torch.cuda.is_initialized(): raise ValueError('unexpected CUDA initialization')
    result = {'scope': 'tiny random CPU model and synthetic image, 33 forced synthetic tokens; no real data',
              'plan_sha256': sha(HERE/'plan001.json'), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'before_signatures': before, 'after_signatures': after,
              'synthetic_image_grid': [1, 4, 4], 'original_merged_tokens': 4,
              'selector_kept_indices': [x.tolist() for x in keep],
              'original_rope_delta': delta.tolist(), 'original_max_position': original_max,
              'records': records, 'disabled_output_equals_unpatched': outputs.get('unpatched') == outputs.get('patched_explicit_disabled')
                  if 'unpatched' in outputs and 'patched_explicit_disabled' in outputs else None,
              'cuda_initialized': False}
    with (HERE/'result001.json').open('x') as f: json.dump(result, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


if __name__ == '__main__': main()
