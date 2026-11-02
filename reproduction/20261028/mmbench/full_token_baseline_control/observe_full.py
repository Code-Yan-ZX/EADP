"""Post-generation observers around the unchanged author FULL entry."""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import time

import torch
import transformers
import llava.model.builder as builder

OUT = Path(sys.argv[1]).resolve()
NATIVE = Path(sys.argv[2]).resolve()
NATIVE_ARGV = [str(NATIVE)] + sys.argv[3:]
OUT.mkdir(exist_ok=True, parents=True)
real_load = builder.load_pretrained_model


def observed_load(*args, **kwargs):
    plan=json.loads((OUT.parent/'protocol.json').read_text())
    assert torch.__version__==plan['runtime_versions']['torch']
    assert transformers.__version__==plan['runtime_versions']['transformers']
    result = real_load(*args, **kwargs)
    tokenizer, model, processor, context_len = result
    assert model.visual_token_num == 0 and model.alpha == .5 and model.beta == 1
    tower = model.get_vision_tower()
    encode = model.encode_images
    arch = Path(inspect.getfile(encode.__func__)).resolve()
    expected_arch = NATIVE.parents[1] / 'model/llava_arch.py'
    assert arch == expected_arch.resolve()
    dtype = lambda m: sorted({str(p.dtype) for p in m.parameters()})
    details = dict(native_argv=NATIVE_ARGV,python=sys.version,torch=torch.__version__,
                   transformers=transformers.__version__,model_class=type(model).__name__,
                   model_class_source=inspect.getfile(type(model)),model_dtype=str(model.dtype),
                   model_device=str(model.device),model_config=model.config.to_dict(),
                   visual_token_num=model.visual_token_num,alpha=model.alpha,beta=model.beta,
                   clip_class_source=inspect.getfile(type(tower)),clip_dtype=str(tower.dtype),
                   clip_device=str(tower.device),image_processor=processor.to_dict(),context_len=context_len,
                   actual_encode_images_source=str(arch),actual_encode_images_source_sha256=hashlib.sha256(arch.read_bytes()).hexdigest(),
                   attention_backend=getattr(model.config,'_attn_implementation',None),
                   parameter_dtypes={k:dtype(v) for k,v in dict(vision_body=tower.vision_tower,
                       text=tower.text_tower,visual_projection=tower.vision_tower.visual_projection,
                       mm_projector=model.get_model().mm_projector).items()},
                   tokenizer_length=len(tokenizer),default_dtype=str(torch.get_default_dtype()),
                   observer_math_changes=False,pre_generation_tensor_hashing_or_synchronization=False)
    assert details['attention_backend']=='sdpa' and model.dtype==torch.float16
    (OUT/'loaded_model.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
    capture=[]
    @functools.wraps(encode)
    def observed_encode(*arguments,**keywords):
        features,mask=encode(*arguments,**keywords)
        # Retain references only; all reads/copies/synchronization happen after
        # native generation, not between CLIP/projector and the language model.
        capture.append((features,mask))
        return features,mask
    model.encode_images=observed_encode
    generate=model.generate
    calls=0
    @functools.wraps(generate)
    def observed_generate(*arguments,**keywords):
        nonlocal calls
        capture.clear();start=time.monotonic()
        answer=generate(*arguments,**keywords)
        calls+=1
        assert len(capture)==1
        features,mask=capture[0]
        masks=mask.detach().cpu()
        assert list(masks.shape)==[1,576] and bool(masks.all())
        assert list(features.shape)==[1,576,4096] and features.dtype==torch.float16
        assert bool(torch.isfinite(features).all()) and int(answer[1])==576
        images=keywords['images'];text=keywords.get('texts','')
        input_ids=arguments[0].detach().cpu().contiguous()
        ids=answer[0].detach().cpu().tolist()
        record=dict(call=calls,seconds=time.monotonic()-start,input_shape=list(input_ids.shape),
                    input_ids_sha256=hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
                    image_shape=list(images.shape),image_dtype=str(images.dtype),image_sizes=keywords.get('image_sizes'),
                    guidance_text_sha256=hashlib.sha256(text.encode()).hexdigest(),guidance_text=text,
                    do_sample=keywords.get('do_sample'),temperature=keywords.get('temperature'),
                    num_beams=keywords.get('num_beams'),max_new_tokens=keywords.get('max_new_tokens'),
                    returned_visual_token_num=int(answer[1]),output_shape=list(answer[0].shape),
                    generated_token_ids=ids,encode_calls=1,actual_mask_shape=list(masks.shape),
                    actual_mask_all_true=bool(masks.all()),actual_retained_tokens=int(masks.sum()),
                    projected_features_shape=list(features.shape),projected_features_finite=True)
        with (OUT/'runtime.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        return answer
    model.generate=observed_generate
    return result


builder.load_pretrained_model=observed_load
sys.argv=NATIVE_ARGV
runpy.run_path(str(NATIVE),run_name='__main__')
