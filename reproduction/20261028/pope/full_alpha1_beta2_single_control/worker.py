"""Unmodified fresh-author eval_model, beta2 builder argument and postcall observer."""
import argparse
import inspect
import json
import marshal
import os
from pathlib import Path
import runpy
import sys
import common as c


def worker(note_path):
    plan=c.read(c.PLAN);c.check(plan);claim=c.read(c.HERE/'launch.claim.json');note=c.read(note_path)
    from run import identity,allocation
    assert identity(os.getppid())==claim['controller']
    assert c.sha(c.PLAN)==claim['plan_sha256']==note['plan_sha256']
    assert c.sha(note_path)==claim['allocation_note_sha256'];allocation(note,plan)
    c.OUT.mkdir(exist_ok=False)
    c.save(c.OUT/'process_identity.json',dict(**identity(os.getpid()),session_id=os.getsid(0),
        plan_sha256=c.sha(c.PLAN),allocation_note_sha256=c.sha(note_path)))
    sys.path.insert(0,plan['llava_root']);sys.path.insert(0,str(c.POPE))
    from wait_only_patch import install
    patch=install()
    for key,value in plan['wait_patch_prepared_proof'].items(): assert patch[key]==value,key
    assert patch['runtime_patch_installed'];c.save(c.OUT/'wait_patch_proof.json',patch)
    import torch
    assert torch.__version__=='2.1.2+cu118' and __import__('transformers').__version__=='4.37.2'
    assert not torch.cuda.is_initialized()
    questions=[json.loads(line) for line in Path(plan['question_file']).read_text().splitlines()]
    assert len(questions)==8910
    anchors={(str(r['question_id']),r['prompt']):r['expected'] for r in c.read(c.HERE/'beta2_panel_anchors.json')['rows']}
    assert len(anchors)==16
    entry=Path(plan['entry']);namespace=runpy.run_path(str(entry),run_name='fresh_author_pope_beta2')
    evaluate=namespace['eval_model'];load=evaluate.__globals__['load_pretrained_model']
    actual_code_sha256=__import__('hashlib').sha256(marshal.dumps(evaluate.__code__)).hexdigest()
    assert actual_code_sha256==plan['eval_model_code_sha256']
    finish_observer=None;runtime={}
    def loader(*args,**kwargs):
        nonlocal finish_observer
        assert args[0]==plan['model_path'] and kwargs==dict(visual_token_num=32,beta=2.,alpha=1.)
        result=load(*args,**kwargs);tokenizer,model,processor,context=result
        assert [model.alpha,model.beta,model.visual_token_num]==[1.,2.,32]
        assert str(model.dtype)=='torch.float16' and model.config._attn_implementation=='sdpa'
        assert model.config.image_aspect_ratio=='pad' and not model.config.mm_use_im_start_end
        assert not torch.is_autocast_enabled() and torch.get_default_dtype()==torch.float32
        assert not torch.backends.cuda.matmul.allow_tf32
        tower=model.get_vision_tower();dtype=lambda m:sorted({str(p.dtype) for p in m.parameters()})
        dtypes={k:dtype(v) for k,v in dict(vision_body=tower.vision_tower.vision_model,
            visual_projection=tower.vision_tower.visual_projection,text_tower=tower.text_tower,
            mm_projector=model.get_model().mm_projector).items()}
        assert dtypes==plan['expected_component_dtypes']
        bound={}
        for name in ['generate','encode_images','greed_select','prepare_inputs_labels_for_multimodal']:
            path=inspect.getfile(inspect.unwrap(getattr(model,name).__func__))
            assert c.sha(path)==plan['source_sha256'][path];bound[name]=dict(path=path,sha256=c.sha(path))
        runtime.update(actual_alpha=1.,actual_beta=2.,actual_K=32,model_class=type(model).__module__+'.'+type(model).__name__,
            model_dtype=str(model.dtype),backend=model.config._attn_implementation,component_dtypes=dtypes,
            torch=torch.__version__,transformers=__import__('transformers').__version__,CUDA=torch.version.cuda,
            GPU=torch.cuda.get_device_name(0),tokenizer_class=type(tokenizer).__name__,processor_class=type(processor).__name__,
            bos_token_id=model.generation_config.bos_token_id,eos_token_id=model.generation_config.eos_token_id,
            bound_methods=bound,source_sha256=plan['source_sha256'],entry_source_sha256=c.sha(entry),
            original_eval_model_code_filename=evaluate.__code__.co_filename,
            actual_eval_model_code_sha256=actual_code_sha256,
            original_eval_model_AST_sha256=plan['eval_model_AST_sha256'],unmodified_fresh_eval_model=True,
            backend_flags=dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic),
            observer=dict(path=str(c.HERE/'observe.py'),sha256=c.sha(c.HERE/'observe.py'),pre_generation_tensor_reads=False,
                native_numerical_operations_replaced=False),plan_sha256=c.sha(c.PLAN),worker_sha256=c.sha(__file__))
        from observe import install
        finish_observer=install(model,tokenizer,questions,anchors,runtime)
        c.save(c.OUT/'loaded_runtime.json',runtime)
        return result
    evaluate.__globals__['load_pretrained_model']=loader
    args=argparse.Namespace(model_path=plan['model_path'],model_base=None,image_folder=plan['image_folder'],
        question_file=plan['question_file'],answers_file=str(c.OUT/'predictions.jsonl'),conv_mode='vicuna_v1',num_chunks=1,chunk_idx=0,
        temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,visual_token_num=32,alpha=1.,beta=2.)
    evaluate.__globals__['args']=args
    c.save(c.OUT/'native_args.json',vars(args))
    evaluate(args)
    observation=finish_observer()
    raw=[json.loads(line) for line in (c.OUT/'predictions.jsonl').read_text().splitlines()]
    assert len(raw)==8910 and [r['question_id'] for r in raw]==[r['question_id'] for r in questions]
    assert all(r['prompt']==q['text'] for r,q in zip(raw,questions))
    c.check(plan)
    c.save(c.OUT/'native_complete.json',dict(success=True,n_predictions=8910,**observation,
        predictions=c.describe(c.OUT/'predictions.jsonl'),runtime_observations=c.describe(c.OUT/'runtime_observations.jsonl'),
        plan_sha256=c.sha(c.PLAN),GT_read=False,scoring_started=False,all8910_generation_completed_before_scoring=True))


if __name__=='__main__':
    assert len(sys.argv)==2
    try: worker(Path(sys.argv[1]).resolve())
    except BaseException as error:
        c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error)));raise
