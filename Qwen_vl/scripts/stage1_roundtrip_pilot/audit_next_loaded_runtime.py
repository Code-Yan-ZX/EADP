"""CPU-only inspection of the real NeXT loader and dense-guidance token validity.

Does not install a model patch, initialize CUDA, or calculate benchmark accuracy.
The activation probe uses FP32 on CPU after recording original loaded dtypes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / 'LLaVA'))
    import torch
    import transformers
    from safetensors import safe_open
    from PIL import Image
    from llava.model.builder import load_pretrained_model
    from llava.model.llava_arch import LlavaMetaForCausalLM
    from llava.mm_utils import get_model_name_from_path, process_images
    from llava.utils import disable_torch_init

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20261009)
    source = root / 'LLaVA/playground/data/eval/anchorzip_p3/next_textvqa_official_20261008/EGATHER_K32.jsonl.protocol.json'
    base = json.loads(source.read_text())
    questions = [json.loads(line) for line in Path(base['question_file']).open()]
    assert len(questions) == 5000 and 0 < args.samples <= len(questions)
    indices = [((2*i+1)*len(questions))//(2*args.samples) for i in range(args.samples)]
    result = dict(cpu_only=True, cpu_activation_dtype='float32',
                  limitation='CPU loader/device placement; not a GPU activation or accuracy measurement',
                  source_protocol=str(source), source_protocol_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch=torch.__version__, transformers=transformers.__version__,
                  sample_selection='Equal-width midpoint indices fixed without labels or predictions', indices=indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result['cuda_initialized'] = torch.cuda.is_initialized()
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        tmp.replace(args.output)

    disable_torch_init()
    print('Loading exact builder on CPU with device_map=auto; no CUDA visible', flush=True)
    _, model, processor, context = load_pretrained_model(
        base['model_path'], None, get_model_name_from_path(base['model_path']),
        visual_token_num=32, alpha=.5, beta=2., device_map='auto')
    tower = model.get_vision_tower()
    config = model.config
    projector = model.get_model().mm_projector
    result['loaded'] = dict(
        class_name=f'{type(model).__module__}.{type(model).__name__}',
        alpha=model.alpha, beta=model.beta, visual_token_num=model.visual_token_num,
        context_len=context, attention_backend=config._attn_implementation,
        hf_device_map=model.hf_device_map,
        dtypes={name:dict(Counter(str(p.dtype) for p in mod.parameters())) for name,mod in
                [('model',model),('vision',tower.vision_tower),('text',tower.text_tower),('projector',projector)]},
        config={k:getattr(config,k,None) for k in ['mm_vision_tower','unfreeze_mm_vision_tower','mm_vision_select_layer','image_aspect_ratio','image_grid_pinpoints','mm_patch_merge_type']})
    state = model.state_dict()
    checked = []
    for shard in sorted(Path(base['model_path']).glob('model-*.safetensors')):
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            for key in f.keys():
                if not (key.startswith('model.vision_tower.') or key.startswith('model.mm_projector.')):
                    continue
                expected = f.get_tensor(key).to(state[key].dtype)
                checked.append(dict(key=key, same=torch.equal(state[key],expected),
                                    loaded_dtype=str(state[key].dtype), saved_dtype=str(f.get_slice(key).get_dtype())))
    result['embedded_vision_projector_weights'] = dict(n=len(checked), mismatches=[c for c in checked if not c['same']], all_identical=all(c['same'] for c in checked))
    save()
    print(json.dumps(result['loaded']), flush=True)
    print(json.dumps(result['embedded_vision_projector_weights']), flush=True)
    del state, model
    gc.collect()
    # The production CPU half path cannot execute all CLIP ops in torch 2.1.
    # Explicit FP32 probe isolates token-validity semantics from GPU precision.
    tower.float().eval()
    result['samples'] = []
    suffix = '\nAnswer the question using a single word or phrase.'
    with torch.inference_mode():
        for position in indices:
            row = questions[position]
            prompt = row['text'].replace(suffix,'')
            image = Image.open(Path(base['image_folder']) / row['image']).convert('RGB')
            pixels = process_images([image], processor, config)[0].float()
            visual = tower.vision_tower(pixels, output_hidden_states=True)
            features = tower.feature_select(visual)
            embeddings = tower.vision_tower.visual_projection(tower.vision_tower.vision_model.post_layernorm(features))
            raw = tower.text_tokenizer(text=prompt, return_tensors='pt')
            length = raw.input_ids.shape[1]
            pad = (-length) % tower.max_position_embeddings
            inputs = {k:torch.cat([v,v.new_zeros((v.shape[0],pad))],1).reshape(-1,tower.max_position_embeddings) for k,v in raw.items()}
            outputs = tower.text_tower(**inputs)
            seq = tower.text_tower.text_projection(outputs.last_hidden_state)
            _, local = LlavaMetaForCausalLM.sim_cross(None, outputs.text_embeds, seq, embeddings)
            logits = local*100.
            probs = logits.softmax(1)
            entropy = -(probs*(probs+1e-12).log()).sum(1)
            selected = entropy.topk(15, dim=-1, largest=False).indices
            valid = inputs['attention_mask'].bool()
            special = (inputs['input_ids'] == tower.text_tokenizer.bos_token_id) | (inputs['input_ids'] == tower.text_tokenizer.eos_token_id)
            content = valid & ~special
            chosen_padding = (~valid).unsqueeze(0).expand(features.shape[0],-1,-1).gather(-1,selected)
            chosen_special = (valid & special).unsqueeze(0).expand(features.shape[0],-1,-1).gather(-1,selected)
            selected_entropy = entropy.gather(-1, selected)
            weights = (-selected_entropy/(.01+1e-6)).softmax(-1)
            record = dict(position=position,question_id=row['question_id'],text_length=length,
                          segments=inputs['input_ids'].shape[0], crops=features.shape[0],
                          selected_total=selected.numel(), selected_padding=int(chosen_padding.sum()),
                          selected_special=int(chosen_special.sum()),
                          invalid_local_weight_per_crop_segment=(weights*(chosen_padding|chosen_special)).sum(-1).tolist(),
                          content_tokens_per_segment=content.sum(-1).tolist(),
                          entropy_finite=bool(torch.isfinite(entropy).all()),
                          pooled_eos_locations=(inputs['input_ids'].int().argmax(-1)).tolist(),
                          proper_eos_per_segment=((inputs['input_ids']==tower.text_tokenizer.eos_token_id)&valid).sum(-1).tolist())
            result['samples'].append(record)
            save()
            print(json.dumps(record),flush=True)
    result['complete'] = True
    save()


if __name__ == '__main__':
    main()
