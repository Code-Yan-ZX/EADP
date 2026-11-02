#!/usr/bin/env python3
"""Two single-formula CPU diagnostics on the already fixed16-image panel.

Never generates answers or reads labels. No GPU and no approximate projector.
"""
import ast
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
IMAGES = Path('/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval/pope/val2014')
OUT = ROOT / 'cpu_formula_real_panel'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    sys.path.insert(0, str(ROOT))
    from alpha_control import validate_frozen
    frozen_path = ROOT / 'run03_single_alpha05_beta1_wait_k32/run_plan.json'
    frozen_sha = sha(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    validate_frozen(frozen)
    prior_path = ROOT / 'cpu_dense_token_audit.json'
    prior = json.loads(prior_path.read_text())
    panel = prior['panel_rows']
    assert len(panel) == 16
    # Preserve the original candidates (all77 including padding), q.2, source
    # negative cosine and temperatures. Only each declared formula changes.
    arch = SOURCE / 'llava/model/llava_arch.py'
    tree = ast.parse(arch.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlavaMetaForCausalLM')
    names = ['sim_cross', 'entrpy_filter', 'local_aggregation', 'spatial_smoothing']
    selected = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = {'torch': torch}
    exec(compile(selected, str(arch), 'exec'), scope)
    methods = types.SimpleNamespace()
    for name in names:
        setattr(methods, name, types.MethodType(scope[name], methods))
    half_linear = {'attempted_dtype': 'torch.float16'}
    try:
        F.linear(torch.ones(1, 4, dtype=torch.float16), torch.ones(3, 4, dtype=torch.float16))
        half_linear.update(native_CPU_supported=True)
    except RuntimeError as error:
        half_linear.update(native_CPU_supported=False, error=str(error))
    # Do not substitute an FP32 MLP plus rounding: such a facility result would
    # change the arithmetic and cannot exclude effects on real GPU selection.
    assert not half_linear['native_CPU_supported'], 'Review newly available native projector support separately'
    cache_path, metadata_path = OUT / 'cross_modal_intermediates.npz', OUT / 'feature_cache_manifest.json'
    if not cache_path.exists():
        clip_path = SOURCE / 'llava/model/multimodal_encoder/clip_encoder.py'
        spec = importlib.util.spec_from_file_location('author_clip_formula_cpu', clip_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tower = module.CLIPVisionTower(str(CLIP), types.SimpleNamespace(mm_vision_select_layer=-2, mm_vision_select_feature='patch'), delay_load=False)
        tower.load_text_tower(device_map='cpu')
        tower.eval()
        text_cache, cache, metadata = {}, {}, []
        with torch.inference_mode():
            for index, row in enumerate(panel):
                prompt = row['prompt']
                if prompt not in text_cache:
                    guidance = prompt.replace('\nAnswer the question using a single word or phrase.', '')
                    tokens = tower.text_tokenizer(text=guidance, return_tensors='pt')
                    padding = 77 - tokens.input_ids.shape[1]
                    inputs = {key: torch.cat([value, value.new_zeros(value.shape[0], padding)], dim=1) for key, value in tokens.items()}
                    text_out = tower.text_tower(**inputs)
                    seq = tower.text_tower.text_projection(text_out.last_hidden_state)
                    text_cache[prompt] = (text_out.text_embeds, seq, inputs)
                global_text, seq, inputs = text_cache[prompt]
                image = Image.open(IMAGES / row['image']).convert('RGB')
                w, h = image.size
                padded = Image.new('RGB', (max(w, h), max(w, h)), tuple(int(v * 255) for v in tower.image_processor.image_mean))
                padded.paste(image, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
                pixels = tower.image_processor.preprocess(padded, return_tensors='pt')['pixel_values'].half().float()
                image_out = tower.vision_tower(pixels, output_hidden_states=True)
                features = tower.feature_select(image_out)
                image_embeds = tower.vision_tower.visual_projection(tower.vision_tower.vision_model.post_layernorm(features))
                global_sim, dense = methods.sim_cross(global_text, seq, image_embeds)
                cache[f'global_{index:02d}'] = global_sim.numpy()
                cache[f'dense_{index:02d}'] = dense.numpy()
                metadata.append({'index': index, 'image': row['image'], 'image_sha256': sha(IMAGES / row['image']),
                    'question_id': row['question_id'], 'prompt': prompt,
                    'CLIP_input_ids': inputs['input_ids'][0].tolist(), 'CLIP_attention_mask': inputs['attention_mask'][0].tolist(),
                    'input_FP16_roundtrip_FP32_sha256': hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
                    'global_dtype': str(global_sim.dtype), 'dense_dtype': str(dense.dtype)})
                print(json.dumps({'CPU_image_feature_extractions': index + 1, 'maximum': 16}), flush=True)
        np.savez_compressed(cache_path, **cache)
        manifest = {'started_utc': start, 'panel_source': str(prior_path), 'panel_source_sha256': sha(prior_path),
            'scope': 'Same16images and first canonical question per image as prior fixed endpoint panel. No GT, no generated answers. CLIPFP32 CPU native loaders, source input FP32->FP16->FP32 rounding; no claim of GPU bit identity.',
            'source_files': {str(path): sha(path) for path in [arch, clip_path]},
            'model_payload_identity_reference': str(ROOT / 'model_identity/model_identity_audit.json'),
            'model_payload_identity_reference_sha256': sha(ROOT / 'model_identity/model_identity_audit.json'),
            'cache_sha256': sha(cache_path), 'bytes': cache_path.stat().st_size,
            'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'images': metadata}
        metadata_path.write_text(json.dumps(manifest, indent=2) + '\n')
        del tower, text_cache
    manifest = json.loads(metadata_path.read_text())
    assert sha(cache_path) == manifest['cache_sha256'] and sha(prior_path) == manifest['panel_source_sha256']
    cache = np.load(cache_path)
    rows = []
    with torch.inference_mode():
        for index, previous in enumerate(panel):
            global_sim = torch.from_numpy(cache[f'global_{index:02d}'])
            dense = torch.from_numpy(cache[f'dense_{index:02d}'])
            probabilities = torch.softmax(dense * 100., dim=1)
            entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=1)
            source_indices = entropy.topk(15, largest=False).indices
            source_local, source_ent = methods.entrpy_filter(dense, T=100., entropy_keep_ratio=.2)
            source_aggregated = methods.local_aggregation(source_local, source_ent, M_temp=.01, strategy='negative_entropy')
            threshold = torch.quantile(entropy, .2, dim=-1, keepdim=True, interpolation='linear')
            quantile_indices = torch.nonzero((entropy <= threshold)[0, 0], as_tuple=False).flatten()
            quantile_dense = dense[..., quantile_indices]
            quantile_entropy = entropy[..., quantile_indices]
            quantile_aggregated = methods.local_aggregation(quantile_dense, quantile_entropy, M_temp=.01, strategy='negative_entropy')
            quantile_weights = torch.softmax(-quantile_entropy / (.01 + 1e-6), dim=-1)[0, 0]
            added = [int(i) for i in quantile_indices if int(i) not in source_indices[0, 0].tolist()]
            prior_selected = previous['selected_dense_tokens']
            repeat_indices_equal = source_indices[0, 0].tolist() == [r['position'] for r in prior_selected]
            repeat_entropy_error = max(abs(float(entropy[0, 0, r['position']]) - r['entropy']) for r in prior_selected)
            recipe_rows = {}
            for alpha, beta in [(1., 1.), (.5, 2.)]:
                def importance(local, numerator_epsilon):
                    raw = (alpha * global_sim + (1 - alpha) * local).mean(dim=-1)
                    minimum = raw.min(dim=-1, keepdim=True).values
                    span = raw.max(dim=-1, keepdim=True).values - minimum
                    normalized = (raw - minimum + numerator_epsilon) / (span + 1e-6)
                    return methods.spatial_smoothing(normalized, kernel_size=3, sigma=1.) ** beta, span
                source_importance, span = importance(source_aggregated, 1e-6)
                eq10, _ = importance(source_aggregated, 0.)
                eq6, _ = importance(quantile_aggregated, 1e-6)
                variants = {}
                for name, value in [('Eq10_remove_numerator_epsilon_only', eq10), ('Eq6_linear_quantile_ties_only', eq6)]:
                    delta = (value - source_importance).abs()
                    variants[name] = {'importance_exact_equal': torch.equal(value, source_importance),
                        'importance_max_abs_delta': float(delta.max()), 'importance_mean_abs_delta': float(delta.mean()),
                        'importance_elements_changed': int((delta != 0).sum()),
                        'source_importance_sha256': hashlib.sha256(source_importance.numpy().tobytes()).hexdigest(),
                        'variant_importance_sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                        'importance_top32_set_overlap_PROXY_NOT_FACILITY': len(set(value.topk(32).indices[0].tolist()) & set(source_importance.topk(32).indices[0].tolist())),
                        'facility_retained_set_comparison': None}
                recipe_rows[f'alpha{alpha}_beta{beta}'] = {'alpha': alpha, 'beta': beta,
                    'fused_raw_score_span': float(span[0, 0]), 'span_is_degenerate_zero': bool(span[0, 0] == 0),
                    'normalization_offset_removed': float(1e-6 / (span[0, 0] + 1e-6)), 'variants': variants}
            row = {'index': index, 'image': previous['image'], 'question_id': previous['question_id'],
                'source_kept_text_count': 15, 'quantile_kept_text_count': len(quantile_indices),
                'linear_quantile_threshold': float(threshold[0, 0, 0]),
                'quantile_threshold_ties': int((entropy == threshold).sum()),
                'new_text_positions': added,
                'new_text_total_aggregation_weight': sum(float(quantile_weights[(quantile_indices == position).nonzero()[0, 0]]) for position in added),
                'dense_local_max_abs_delta_from_quantile': float((quantile_aggregated - source_aggregated).abs().max()),
                'same_panel_repeat_source_indices_identical': repeat_indices_equal,
                'same_panel_repeat_source_selected_entropy_max_abs_delta': repeat_entropy_error,
                'recipes': recipe_rows}
            rows.append(row)
    summary = {}
    for recipe in ['alpha1.0_beta1.0', 'alpha0.5_beta2.0']:
        summary[recipe] = {}
        for variant in ['Eq10_remove_numerator_epsilon_only', 'Eq6_linear_quantile_ties_only']:
            values = [row['recipes'][recipe]['variants'][variant] for row in rows]
            summary[recipe][variant] = {'importance_changed_images': sum(not x['importance_exact_equal'] for x in values),
                'max_importance_abs_delta': max(x['importance_max_abs_delta'] for x in values),
                'minimum_top32_importance_overlap_PROXY_NOT_FACILITY': min(x['importance_top32_set_overlap_PROXY_NOT_FACILITY'] for x in values),
                'facility_retained_set_comparison': 'not measured; nativeFP16 projector unsupported on thisCPU runtime; no FP32 simulation substituted'}
    validate_frozen(frozen)
    assert sha(frozen_path) == frozen_sha and not torch.cuda.is_initialized()
    result = {'started_utc': start, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Bounded real-feature CPU diagnostics on original16images only, no labels, generation or benchmark accuracy. Two independent variants; q.2 and all source settings unchanged.',
        'requested_visual_budget': 32, 'facility_selection_executed': False,
        'implementation': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
        'quantile_convention': 'torch.quantile(q=.2,dim=-1,interpolation=linear); retain every entropy<=threshold, including exact ties. All77 original sequence candidates remain; no padding/EOS mask added.',
        'feature_cache_manifest': str(metadata_path), 'feature_cache_manifest_sha256': sha(metadata_path),
        'half_precision_projector_probe': half_linear,
        'facility_limit': 'Actual source usesFP16 LLM mm_projector outputs for pairwise visual similarity. CPUtorch2.1.2 does not support nativeFP16 linear. Facility retained sets are unmeasured; top32importance is only a ranked-score proxy and cannot rule out realGPU selection changes.',
        'frozen_run03_plan_sha256_before_and_after': frozen_sha, 'frozen_run03_guards_unchanged': True,
        'degenerate_fused_maps': sum(row['recipes'][recipe]['span_is_degenerate_zero'] for row in rows for recipe in row['recipes']),
        'all_repeated_source_top15_indices_identical_to_prior_CPU_panel': all(row['same_panel_repeat_source_indices_identical'] for row in rows),
        'max_repeated_source_selected_entropy_delta': max(row['same_panel_repeat_source_selected_entropy_max_abs_delta'] for row in rows),
        'summary': summary, 'rows': rows, 'cuda_initialized': False}
    (OUT / 'real_formula_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'frozen_run03_unchanged': True, 'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
