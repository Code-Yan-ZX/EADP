"""All5349 Doc prompts through the real CPU processor; no image-output cache."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import types

import processor_input_identity as probe

HERE = Path(__file__).resolve().parent
EXPECTED_CANONICAL = "79091c0178d228237519c87a50f74f766ca8b2ea6c593d99790d3d0034a45ebc"


def save(path, record):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--environment-label',required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU only'
    args.output_dir.mkdir(parents=True,exist_ok=False)
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    identity=dict(pid=os.getpid(),start_ticks=int(fields[19]),started_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.executable,argv=sys.argv,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_only=True)
    save(args.output_dir/'process_identity.json',identity)
    state=dict(status='initializing_real_CPU_processor',identity=identity,rows=0,target_rows=5349,
               target_unique_images=1286,unique_images=0,model_loaded=False,CUDA_initialized=False)
    save(args.output_dir/'state.json',state)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert not torch.cuda.is_initialized()
    processor=AutoProcessor.from_pretrained(probe.MODEL,local_files_only=True)
    namespace,proof=probe.source_preprocessing(args.source_root,torch,Image)
    context=types.SimpleNamespace(processor=processor,min_pixels=1048576,max_pixels=1048576,system_prompt=None,fps=2)
    for name in ('_prepare_content','_build_messages','_processor_inputs'):
        setattr(context,name,types.MethodType(namespace[name],context))
    prompt_path=args.source_root/'VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py'
    prompt_namespace={};exec(compile(prompt_path.read_text(),str(prompt_path),'exec'),prompt_namespace)
    prompt_builder=prompt_namespace['Qwen3VLPromptMixin'](use_custom_prompt=True)
    image_root=Path('/media/disk2/YZX/LMUData/images/DocVQA_VAL')
    prompt_builder.set_dump_image(lambda row:[str(image_root/row['image_path'])])
    canonical_path=HERE/'independent_audit/DocVQA_VAL.canonical.jsonl'
    assert probe.sha(canonical_path)==EXPECTED_CANONICAL
    canonical=[json.loads(line) for line in canonical_path.read_text().splitlines()]
    assert len(canonical)==5349 and len({r['index'] for r in canonical})==5349
    manifest_path=HERE/'independent_audit/DocVQA_VAL.canonical_manifest.json'
    manifest=json.loads(manifest_path.read_text());stat=Path(manifest['source_stat']['path']).stat()
    assert (stat.st_size,stat.st_mtime_ns)==(manifest['source_stat']['bytes'],manifest['source_stat']['mtime_ns'])
    cache_audit_path=HERE/'dataset_cache_audit.json'
    cache_audit=json.loads(cache_audit_path.read_text())['datasets']['DocVQA_VAL']
    expected_images={r['image_path']:r['cached_sha256'] for r in cache_audit['images']}
    assert len(expected_images)==1286
    smoke=[json.loads(line) for line in probe.SMOKE.read_text().splitlines()]
    smoke={r['position']:r for r in smoke if r['dataset']=='DocVQA_VAL'}
    small_files={str(p):probe.sha(p) for p in probe.MODEL.iterdir() if p.is_file() and p.suffix in ('.json','.jinja') and p.stat().st_size<30*1024*1024}
    small_files.update({str(prompt_path):probe.sha(prompt_path),proof['source']:proof['source_sha256'],
                        str(args.source_root/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model.py'):proof['model_source_sha256'],
                        str(canonical_path):EXPECTED_CANONICAL,str(manifest_path):probe.sha(manifest_path),
                        str(cache_audit_path):probe.sha(cache_audit_path),str(Path(probe.__file__)):probe.sha(probe.__file__)})
    preparation=dict(environment_label=args.environment_label,identity=identity,source_AST_proof=proof,
                     current_dependency_stack_only=True,historical_September22_environment_reconstructed=False,
                     real_full_processor_calls=5349,image_output_cache_reused=False,
                     versions={name:importlib.metadata.version(name) for name in ('torch','transformers','qwen-vl-utils','Pillow','tokenizers','numpy')},
                     processor_class=type(processor).__name__,image_processor_class=type(processor.image_processor).__name__,
                     image_processor_config=processor.image_processor.to_dict(),frozen_small_files=small_files,
                     processor_source={name:dict(path=inspect.getfile(type(obj)),sha256=probe.sha(inspect.getfile(type(obj)))) for name,obj in [('processor',processor),('image_processor',processor.image_processor),('tokenizer',processor.tokenizer)]})
    save(args.output_dir/'preparation.json',preparation)
    def array_record(tensor):
        actual=tensor.detach().cpu().contiguous()
        array=actual.view(torch.uint16).numpy() if actual.dtype==torch.bfloat16 else actual.numpy()
        return dict(dtype=str(actual.dtype),shape=list(actual.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())
    def image_record(image):
        rgb=np.ascontiguousarray(np.asarray(image.convert('RGB')))
        return dict(size=list(image.size),mode=image.mode,RGB_shape=list(rgb.shape),RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    images={};row_path=args.output_dir/'per_question.jsonl'
    with row_path.open('x') as stream:
        for position,row in enumerate(canonical):
            message=prompt_builder._build_vqa_prompt(row,'DocVQA_VAL')
            paths=[s['value'] for s in message if s['type']=='image'];assert len(paths)==1
            image_path=paths[0]
            if image_path not in images:
                assert probe.sha(image_path)==expected_images[image_path]
            if position in smoke:assert message==smoke[position]['message']
            messages=context._build_messages(message,dataset='DocVQA_VAL');stages={}
            # Every question follows the normal complete source processor path.
            # Repeated images are recomputed, never supplied by a fake/cache processor.
            inputs,text=context._processor_inputs(messages,stages)
            actual_keys={k for k,v in inputs.items() if isinstance(v,torch.Tensor)}
            assert actual_keys=={'pixel_values','image_grid_thw','input_ids','attention_mask'}
            raw={k:array_record(inputs[k]) for k in sorted(actual_keys)}
            ids=inputs['input_ids'].tolist();mask=inputs['attention_mask'].tolist();grid=inputs['image_grid_thw'].tolist()
            assert grid==[[1,64,64]] and raw['pixel_values']['shape']==[4096,1536] and raw['pixel_values']['dtype']=='torch.float32'
            native=inputs.to('cpu').to(torch.bfloat16)
            native_meta={k:array_record(native[k]) for k in sorted(actual_keys)}
            image_fingerprint=dict(image_path=image_path,source_sha256=expected_images[image_path],
                                   stages={key:[image_record(image) for image in values] for key,values in stages.items()},
                                   raw_pixel_values=raw['pixel_values'],native_BF16_pixel_values=native_meta['pixel_values'],
                                   image_grid_thw=grid)
            if image_path in images:
                assert images[image_path]['fingerprint']==image_fingerprint, 'Repeated real image processing changed'
                images[image_path]['question_count']+=1
            else:images[image_path]=dict(fingerprint=image_fingerprint,question_count=1,first_position=position)
            item=dict(position=position,index=row['index'],image_path=image_path,
                      message_sha256=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                      text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      raw_processor_tensors=raw,native_CPU_BF16_tensors=native_meta,
                      image_grid_thw=grid,input_ids=ids,attention_mask=mask)
            stream.write(json.dumps(item,ensure_ascii=False)+'\n')
            if (position+1)%32==0 or position+1==5349:
                stream.flush();state.update(status='processing_real_full_path',rows=position+1,unique_images=len(images),observed_utc=datetime.now(timezone.utc).isoformat())
                save(args.output_dir/'state.json',state)
                print(json.dumps(dict(rows=position+1,unique_images=len(images))),flush=True)
    assert len(images)==1286 and set(images)==set(expected_images)
    assert sum(r['question_count'] for r in images.values())==5349
    for path,expected in small_files.items():assert probe.sha(path)==expected,path
    fa2=[n for n in sys.modules if n.startswith('flash_attn') or n=='flash_attn_2_cuda']
    assert not torch.cuda.is_initialized() and not fa2
    image_output=args.output_dir/'unique_images.json';save(image_output,images)
    report=dict(status='complete',rows=5349,unique_images=1286,real_full_processor_calls=5349,
                current_dependency_stack_only=True,historical_environment_reconstruction=False,
                model_loaded=False,CUDA_initialized=False,FA2_modules_imported=[],image_output_cache_reused=False,
                identity=identity,preparation_sha256=probe.sha(args.output_dir/'preparation.json'),
                per_question=dict(path=str(row_path),sha256=probe.sha(row_path)),
                unique_image_fingerprints=dict(path=str(image_output),sha256=probe.sha(image_output)),
                source_input_processor_hashes_unchanged=True,script_sha256=probe.sha(__file__),completed_utc=datetime.now(timezone.utc).isoformat())
    save(args.output_dir/'report.json',report)
    state.update(status='complete',rows=5349,unique_images=1286,report_sha256=probe.sha(args.output_dir/'report.json'),completed_utc=report['completed_utc'])
    save(args.output_dir/'state.json',state)
    print(json.dumps(dict(status='complete',rows=5349,unique_images=1286,report=str(args.output_dir/'report.json'))),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        # Keep all partial metadata without a retry or any model fallback.
        parser_args=sys.argv
        if '--output-dir' in parser_args:
            folder=Path(parser_args[parser_args.index('--output-dir')+1])
            if folder.exists():save(folder/'failure.json',dict(status='terminal_failure',error=repr(error),automatic_retry=False,model_loaded=False))
        raise
