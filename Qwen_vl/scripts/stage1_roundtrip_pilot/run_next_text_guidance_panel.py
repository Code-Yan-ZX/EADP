"""Fixed four-arm NeXT K32 mechanism panel, separate from published scores."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
ARMS = {'wait_released':(False,False), 'valid_dense':(True,False),
        'bounded_chunks':(False,True), 'both':(True,True)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'records.jsonl').exists():
        raise FileExistsError('Preserve existing generation records')
    panel=ROOT/'Qwen_vl/outputs/audit_followup_20261008/next_gap_diagnosis_20261008/stability/next_text_stream_panel.py'
    spec=importlib.util.spec_from_file_location('frozen_wait_panel',panel)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    base=json.loads(helper.BASE.read_text())
    questions=[json.loads(l) for l in Path(base['question_file']).open()]
    assert len(questions)==5000 and 0<args.n<=5000
    indices=[((2*i+1)*5000)//(2*args.n) for i in range(args.n)]
    samples=[dict(position=i,question_id=questions[i]['question_id'],image=questions[i]['image'],
                  prompt=questions[i]['text'],image_sha256=sha(Path(base['image_folder'])/questions[i]['image'])) for i in indices]
    sources={str(ROOT/'LLaVA'/k):v for k,v in base['source_sha256'].items()}
    for f in [Path(__file__),Path(__file__).with_name('next_text_guidance_control.py'),panel,helper.HELPER,
              helper.BASE,Path(base['question_file']),ROOT/'LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json']:
        sources[str(f)]=sha(f)
    for f,digest in sources.items():
        if sha(f)!=digest:
            raise ValueError('Source drift: '+f)
    protocol=dict(version=1,n=args.n,arms=ARMS,samples=samples,source_sha256=sources,
                  parameters={k:base[k] for k in ['model_path','alpha','beta','visual_token_num','conv_mode','max_new_tokens','temperature']},
                  selection='Fixed equal-width midpoint samples without labels or predictions',
                  interpretation='Mechanism panel only; not a full TextVQA reproduction score',
                  changes='All arms share wait-only fix; orthogonal content-mask and BOS/EOS chunk controls; q=.2, alpha=.5, beta=2 unchanged')
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Prepared protocol drift')
    dump(protocol_path,protocol)
    if not args.run:
        print(json.dumps(dict(prepared=True,n=args.n,generations=args.n*4)),flush=True)
        return
    gate=helper.safe_slot()
    for name in ['USE_LLAVA_ARCH_CDPRUNER','USE_LLAVA_ARCH_DIVPRUNE','USE_LLAVA_ARCH_HIPRUNE','USE_LLAVA_ARCH_ABLATION']:
        if os.environ.get(name)=='1':
            raise ValueError('Wrong architecture environment')
    sys.path.insert(0,str(ROOT/'LLaVA'))
    import torch
    from PIL import Image
    from llava.model.builder import load_pretrained_model
    from llava.model.multimodal_encoder import clip_encoder
    from llava.mm_utils import get_model_name_from_path,process_images,tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN,DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.utils import disable_torch_init
    from llava.eval.eval_textvqa import prompt_processor
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    import next_text_guidance_control as control
    torch.set_num_threads(4)
    torch.manual_seed(20261009)
    module,validation=helper.patch_module()
    ns=dict(clip_encoder.__dict__)
    exec(compile(module,'independent_wait_only','exec'),ns)
    clip_encoder.CLIPVisionTower.forward=ns['forward']
    disable_torch_init()
    state=dict(status='loading',complete=False,pid=os.getpid(),n=args.n,completed=0,gpu_gate=gate,
               protocol_sha256=sha(protocol_path),wait_patch_validation=validation)
    dump(out/'state.json',state)
    tokenizer,model,processor,_=load_pretrained_model(base['model_path'],None,get_model_name_from_path(base['model_path']),
                                                     visual_token_num=32,alpha=.5,beta=2.)
    tower=model.get_vision_tower()
    state['actual_loaded']=dict(class_name=f'{type(model).__module__}.{type(model).__name__}',
        alpha=model.alpha,beta=model.beta,visual_token_num=model.visual_token_num,
        dtype=str(model.dtype),vision_dtype=str(tower.dtype),text_dtype=str(tower.text_tower.dtype),
        vision_projection_dtype=str(tower.vision_tower.visual_projection.weight.dtype),
        attention_backend=model.config._attn_implementation,torch=torch.__version__,
        transformers=__import__('transformers').__version__)
    state['status']='generating'
    dump(out/'state.json',state)
    annotations=json.loads((ROOT/'LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json').read_text())['data']
    annotations={(r['image_id'],r['question'].lower()):r for r in annotations}
    evaluator=TextVQAAccuracyEvaluator()
    records=[]
    with (out/'records.jsonl').open('x') as stream:
        for j,sample in enumerate(samples):
            prompt=sample['prompt']
            image=Image.open(Path(base['image_folder'])/sample['image']).convert('RGB')
            pixels=process_images([image],processor,model.config)[0]
            image_prefix=DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
            conv=conv_templates[base['conv_mode']].copy()
            conv.append_message(conv.roles[0],image_prefix+'\n'+prompt)
            conv.append_message(conv.roles[1],None)
            ids=tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
            order=list(ARMS)
            order=order[j%4:]+order[:j%4]
            row=dict(position=sample['position'],question_id=sample['question_id'],prompt=prompt,order=order,arms={})
            for arm in order:
                mask,chunks=ARMS[arm]
                restore=control.install(model,mask_content=mask,bounded_chunks=chunks)
                try:
                    with torch.inference_mode():
                        output,count=model.generate(ids,images=pixels.unsqueeze(0).half().cuda(),image_sizes=[image.size],
                            texts=prompt.replace(helper.SUFFIX,''),do_sample=False,temperature=0,top_p=None,num_beams=1,
                            max_new_tokens=128,use_cache=True)
                    text=tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip()
                    annotation=annotations[(sample['question_id'],prompt_processor(prompt))]
                    score=float(evaluator.eval_pred_list([dict(pred_answer=text,gt_answers=annotation['answers'])]))
                    row['arms'][arm]=dict(text=text,score=score,visual_tokens=int(count))
                finally:
                    restore()
            records.append(row)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            state.update(completed=len(records),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            dump(out/'state.json',state)
            print(json.dumps(dict(completed=len(records),position=sample['position'],scores={a:r['score'] for a,r in row['arms'].items()})),flush=True)
    summary={a:dict(panel_accuracy=100*sum(r['arms'][a]['score'] for r in records)/args.n,
                   changed_predictions=sum(r['arms'][a]['text']!=r['arms']['wait_released']['text'] for r in records)) for a in ARMS}
    result=dict(complete=True,n=args.n,summary=summary,records_sha256=sha(out/'records.jsonl'),
                protocol_sha256=sha(protocol_path),limitation=protocol['interpretation'])
    dump(out/'result.json',result)
    state.update(status='complete',complete=True)
    dump(out/'state.json',state)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
