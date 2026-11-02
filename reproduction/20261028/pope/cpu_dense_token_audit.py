#!/usr/bin/env python3
"""CPU-only native CLIP construction and paper/source dense-token diagnostic.

No LLM, generation, GT labels, scoring, GPU context, source edits, or parameter search.
The image panel follows a fixed, label-free filename rule declared in this script.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SOURCE = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/LLaVA')
CLIP = Path('/home/dell/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1')
IMAGE_ROOT = Path('/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval/pope/val2014')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPTokenizerFast
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    started = datetime.now(timezone.utc).isoformat()
    arch_path = SOURCE / 'llava/model/llava_arch.py'
    clip_path = SOURCE / 'llava/model/multimodal_encoder/clip_encoder.py'
    tree = ast.parse(arch_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlavaMetaForCausalLM')
    names = ['sim_cross', 'entrpy_filter', 'local_aggregation', 'spatial_smoothing']
    extracted = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(extracted, str(arch_path), 'exec'), namespace)
    methods = types.SimpleNamespace()
    for name in names:
        setattr(methods, name, types.MethodType(namespace[name], methods))
    spec = importlib.util.spec_from_file_location('author_clip_cpu_diagnostic', clip_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Native builder calls both tower loads without torch_dtype and does not cast
    # vision_tower when its device_map argument is the default 'auto'. CPU is the
    # only declared deviation here; source CLIP loaders and weight payloads are used.
    tower = module.CLIPVisionTower(str(CLIP), types.SimpleNamespace(mm_vision_select_layer=-2,
        mm_vision_select_feature='patch'), delay_load=False)
    tower.load_text_tower(device_map='cpu')
    tower.eval()
    dtypes = {name: sorted({str(p.dtype) for p in sub.parameters()}) for name, sub in {
        'vision_tower': tower.vision_tower, 'visual_projection': tower.vision_tower.visual_projection,
        'text_tower': tower.text_tower}.items()}
    questions = [json.loads(line) for line in (ROOT / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    unique = list(dict.fromkeys(q['text'] for q in questions))
    tokenizer = CLIPTokenizerFast.from_pretrained(CLIP, local_files_only=True)
    token_rows, feature_cache = [], {}
    with torch.inference_mode():
        for text in unique:
            guidance = text.replace('\nAnswer the question using a single word or phrase.', '')
            encoded = tokenizer(text=guidance, return_tensors='pt')
            length = encoded.input_ids.shape[1]
            assert length <= 77
            inputs = {k: torch.cat([v, v.new_zeros((v.shape[0], 77 - length))], dim=1)
                      for k, v in encoded.items()}
            outputs = tower.text_tower(**inputs)
            seq = tower.text_tower.text_projection(outputs.last_hidden_state)
            ids = inputs['input_ids'][0]
            mask = inputs['attention_mask'][0].bool()
            non_eos = mask & (ids != tokenizer.eos_token_id)
            content = non_eos & (ids != tokenizer.bos_token_id)
            feature_cache[text] = (outputs.text_embeds, seq, mask, non_eos, content)
            token_rows.append({'prompt': text, 'unpadded_length': length, 'input_ids': ids.tolist(),
                'attention_mask': inputs['attention_mask'][0].tolist(),
                'tokens': tokenizer.convert_ids_to_tokens(ids.tolist()),
                'nonpadding_nonEOS_count': int(non_eos.sum()), 'content_count': int(content.sum()),
                'source_dense_candidates': seq.shape[1], 'source_q02_keeps': int(.2 * seq.shape[1]),
                'source_sequence_dtype': str(seq.dtype), 'source_global_dtype': str(outputs.text_embeds.dtype)})
            print(json.dumps({'stage': 'text', 'done': len(token_rows), 'total': len(unique)}), flush=True)
        names = sorted({q['image'] for q in questions})
        panel = [names[round(i * (len(names) - 1) / 15)] for i in range(16)]
        panel_rows = []
        for name in panel:
            q = next(q for q in questions if q['image'] == name)
            image = Image.open(IMAGE_ROOT / name).convert('RGB')
            # Source process_images uses image_aspect_ratio='pad'; preserve that
            # exact RGB padding and CLIP processor rather than a direct crop.
            w, h = image.size
            padded = Image.new(image.mode, (max(w, h), max(w, h)), tuple(int(x * 255) for x in tower.image_processor.image_mean))
            padded.paste(image, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
            pixels = tower.image_processor.preprocess(padded, return_tensors='pt')['pixel_values']
            # Native loader first casts image_tensors to FP16, then the standalone
            # FP32 CLIP tower promotes them back. Preserve that rounding on CPU.
            pixels = pixels.half().float()
            forward = tower.vision_tower(pixels, output_hidden_states=True)
            image_outputs = tower.feature_select(forward)
            image_embeds = tower.vision_tower.visual_projection(tower.vision_tower.vision_model.post_layernorm(image_outputs))
            global_emb, seq_emb, mask, non_eos, content = feature_cache[q['text']]
            global_sim, dense = methods.sim_cross(global_emb, seq_emb, image_embeds)
            probs = torch.softmax(dense * 100., dim=1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)[0, 0]
            kept = torch.topk(entropy, int(.2 * dense.shape[-1]), largest=False).indices
            local, ent = methods.entrpy_filter(dense, T=100., entropy_keep_ratio=.2)
            source_local = methods.local_aggregation(local, ent, M_temp=.01, strategy='negative_entropy')
            source_weights = torch.softmax(-ent / (.01 + 1e-6), dim=-1)[0, 0]
            comparisons = {}
            for variant, valid in [('nonpadding_nonEOS', non_eos), ('content_only', content)]:
                filtered, low_ent = methods.entrpy_filter(dense[..., valid], T=100., entropy_keep_ratio=.2)
                masked_local = methods.local_aggregation(filtered, low_ent, M_temp=.01, strategy='negative_entropy')
                alpha_results = {}
                for alpha in [.5, 1.]:
                    def score(loc):
                        scores = (alpha * global_sim + (1. - alpha) * loc).mean(dim=-1)
                        importance = (scores - scores.min(dim=-1, keepdim=True).values + 1e-6) / (scores.max(dim=-1, keepdim=True).values - scores.min(dim=-1, keepdim=True).values + 1e-6)
                        return methods.spatial_smoothing(importance, kernel_size=3, sigma=1.) ** (2. if alpha == .5 else 1.)
                    a, b = score(source_local), score(masked_local)
                    alpha_results[str(alpha)] = {'importance_exact_equal': torch.equal(a, b),
                        'importance_max_abs_difference': float((a - b).abs().max()),
                        'top32_importance_set_overlap': len(set(a.topk(32).indices[0].tolist()) & set(b.topk(32).indices[0].tolist()))}
                comparisons[variant] = {'available_tokens': int(valid.sum()), 'kept_tokens': filtered.shape[-1],
                    'source_vs_masked_local_max_abs_difference': float((source_local - masked_local).abs().max()),
                    'importance_comparison': alpha_results}
            selected = [{'position': int(i), 'token_id': int(token_rows[unique.index(q['text'])]['input_ids'][i]),
                'token': token_rows[unique.index(q['text'])]['tokens'][i], 'is_padding': not bool(mask[i]),
                'is_EOS': int(token_rows[unique.index(q['text'])]['input_ids'][i]) == tokenizer.eos_token_id,
                'entropy': float(entropy[i]), 'aggregation_weight': float(source_weights[j])} for j, i in enumerate(kept)]
            row = {'image': name, 'image_sha256': sha(IMAGE_ROOT / name), 'question_id': q['question_id'], 'prompt': q['text'],
                'cross_similarity_dtype': str(dense.dtype), 'global_dtype': str(global_sim.dtype),
                'entropy_nan_count': int(torch.isnan(entropy).sum()), 'probability_zero_count': int((probs == 0).sum()),
                'selected_dense_tokens': selected, 'selected_padding_count': sum(r['is_padding'] for r in selected),
                'selected_EOS_count': sum(r['is_EOS'] for r in selected),
                'padding_aggregation_weight_mass': sum(r['aggregation_weight'] for r in selected if r['is_padding']),
                'EOS_aggregation_weight_mass': sum(r['aggregation_weight'] for r in selected if r['is_EOS']),
                'comparisons': comparisons}
            panel_rows.append(row)
            print(json.dumps({'stage': 'image', 'done': len(panel_rows), 'total': 16,
                              'padding_selected': row['selected_padding_count'], 'nan': row['entropy_nan_count']}), flush=True)
    assert not torch.cuda.is_initialized()
    result = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {str(arch_path): sha(arch_path), str(clip_path): sha(clip_path)},
        'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
        'cpu_native_loader_parameter_dtypes': dtypes,
        'scope': 'CPU FP32 native standalone CLIP tower construction; source native auto branch skips explicit tower FP16 cast. No LLM, no generation, no GT or score. Device placement differs; cannot claim bit identity to finished GPU intermediates.',
        'all_unique_prompt_count': len(token_rows), 'canonical_prompt_rows': len(questions),
        'unpadded_length_distribution_unique_prompts': dict(Counter(r['unpadded_length'] for r in token_rows)),
        'token_rows': token_rows, 'fixed_panel_rule': '16 evenly spaced rounded indices round(i*499/15) in sorted 500 actually-used image filenames; rule declared before feature computation; first canonical question per image, no GT read. This endpoint-spaced panel differs from the older midpoint-spaced archive SHA panel.',
        'native_image_rounding': 'source processor FP32 -> loader FP16 -> standalone CLIP FP32, reproduced by pixels.half().float()',
        'panel_rows': panel_rows,
        'summary': {'panel_images': len(panel_rows), 'any_padding_selected_images': sum(r['selected_padding_count'] > 0 for r in panel_rows),
            'any_EOS_selected_images': sum(r['selected_EOS_count'] > 0 for r in panel_rows),
            'total_selected_padding': sum(r['selected_padding_count'] for r in panel_rows),
            'total_selected_EOS': sum(r['selected_EOS_count'] for r in panel_rows),
            'total_entropy_nans': sum(r['entropy_nan_count'] for r in panel_rows),
            'all_alpha1_importance_exact_equal': all(r['comparisons']['nonpadding_nonEOS']['importance_comparison']['1.0']['importance_exact_equal'] for r in panel_rows)},
        'cuda_initialized': torch.cuda.is_initialized()}
    (ROOT / 'cpu_dense_token_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
