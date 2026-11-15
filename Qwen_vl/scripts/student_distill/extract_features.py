"""Freeze provenance then generate only fit/val native gold-gradient labels.

This process must load the decoder outside torch.inference_mode so its weights
remain ordinary tensors. No held-out test image or test answer is opened.
"""
from __future__ import annotations
import argparse, json, os, platform, sys, time
from pathlib import Path
import torch
import transformers

HERE=Path(__file__).resolve().parent
QWEN=HERE.parents[1]
ROOT=QWEN.parent
sys.path.insert(0,str(HERE.parent/'multistream_oracle'))
from experiment_common import NativeEngine, common, message, save_json, sha256, MAX_TOKENS
from oracle import answer_token_ids, score_full_streams, tensor_sha256

OUT=QWEN/'outputs/student_distill'
DATA=OUT/'data'
TARGETS=('magnitude_v','magnitude_joint','signed_v','signed_joint')

def sources():
    local=sorted(HERE.glob('*.py'))
    paths=local+[ROOT/'docs/student_distill_prereg.md',
        QWEN/'scripts/multistream_oracle/oracle.py',
        QWEN/'scripts/multistream_oracle/gates.py',
        QWEN/'scripts/multistream_oracle/test_oracle.py',
        QWEN/'scripts/multistream_oracle/experiment_common.py',
        QWEN/'scripts/discovery/common.py',QWEN/'model/native_qwen3.py',
        QWEN/'model/e0_selectors.py',QWEN/'model/pruner.py',
        QWEN/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py',
        QWEN/'VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py',
        QWEN/'scripts/cf_mining/cf_mine.py']
    return paths

def digest_sources():
    return {str(p):sha256(p) for p in sources()}

def check_gate():
    gate_path=OUT/'gates.json'; gate=json.loads(gate_path.read_text())
    if not gate.get('pass_all') or gate.get('cpu_only') or gate.get('checkpointing'):
        raise RuntimeError('need passing GPU native no-checkpoint gate')
    adapter=DATA/'dev_gate_manifest.json'; dev=DATA/'dev_manifest.json'
    adapted=json.loads(adapter.read_text()); original=json.loads(dev.read_text())
    if adapted['role']!='development' or original['role']!='dev' or adapted['items']!=original['items']:
        raise RuntimeError('development gate adapter changed sample')
    if gate['environment']['manifest_sha256']!=sha256(adapter):
        raise RuntimeError('gate used different development input')
    for path,digest in gate['source_sha256'].items():
        if sha256(path)!=digest: raise RuntimeError('native gate source changed: '+path)
    return gate_path

def verify_freeze():
    p=OUT/'train_freeze.json'; freeze=json.loads(p.read_text())
    if freeze['source_sha256']!=digest_sources(): raise RuntimeError('source file set/hash changed after train freeze')
    for role,digest in freeze['manifest_sha256'].items():
        if sha256(DATA/f'{role}_manifest.json')!=digest: raise RuntimeError('data manifest changed: '+role)
    gate_path=check_gate()
    if freeze['gates_sha256']!=sha256(gate_path): raise RuntimeError('native gate changed')
    return freeze

def build_freeze(eng, manifests):
    output=OUT/'train_freeze.json'
    if output.exists(): raise RuntimeError('train freeze already exists; use --resume')
    gate_path=check_gate()
    snaproot=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct'
    revision=(snaproot/'refs/main').read_text().strip(); snap=snaproot/'snapshots'/revision
    if not snap.is_dir() or eng.text.config._attn_implementation!='sdpa':
        raise RuntimeError('unexpected model revision/backend')
    model_path=str(eng.vlm.model_path); loaded_commit=getattr(eng.model.config,'_commit_hash',None)
    if model_path=='Qwen/Qwen3-VL-8B-Instruct':
        if loaded_commit!=revision: raise RuntimeError('loaded model commit differs from cached revision')
    elif Path(model_path).resolve()!=snap.resolve() or (loaded_commit is not None and loaded_commit!=revision):
        raise RuntimeError('loaded model snapshot differs from cached revision')
    rendered={}
    # Only fit/val answers are touched before the teacher labels are frozen.
    for role in ('fit','val'):
        for i,item in enumerate(manifests[role]['items']):
            if sha256(item['image_path'])!=item['image_sha256']: raise ValueError('image hash mismatch')
            prep=eng.prepare(message(item),'DocVQA_VAL')
            if prep['n_vis']!=1024: raise AssertionError('wrong native token count')
            target=answer_token_ids(eng.tok,item['answers'][0],device='cpu').tolist()
            rendered[item['item_id']]=dict(input_ids=prep['ids'][0].cpu().tolist(),target_ids=target,
                image_sha256=item['image_sha256'],role=role)
            print(f'freeze {role} {i+1}/{len(manifests[role]["items"])}',flush=True)
    payload=dict(schema_version=1,pre_teacher=True,
        manifest_sha256={role:sha256(DATA/f'{role}_manifest.json') for role in ('fit','val','test','dev')},
        gate_adapter_sha256=sha256(DATA/'dev_gate_manifest.json'),gates_sha256=sha256(gate_path),
        source_sha256=digest_sources(),targets=list(TARGETS),seeds=[11,23,37],
        model=dict(name=common.BASELINE_MODEL,revision=revision,snapshot=str(snap),loaded_model_path=model_path,loaded_commit=loaded_commit,
            metadata_sha256={name:sha256(snap/name) for name in ('config.json','model.safetensors.index.json','preprocessor_config.json') if (snap/name).exists()},
            dtype=str(next(eng.model.parameters()).dtype)),
        environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,attention_backend=eng.text.config._attn_implementation),
        generation=dict(max_new_tokens=MAX_TOKENS,eos_ids=sorted(eng.eos_ids),do_sample=False,
            prompt_suffix='\nAnswer with the value only.'),rendered_inputs=rendered,
        teacher=dict(checkpointing=False,K=256,gold_alias=0,append_eos=False,streams=['V','DS0','DS1','DS2']))
    save_json(output,payload);print('frozen',sha256(output),flush=True)
    return payload

def atomic_torch_save(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);tmp.replace(path)

def run(eng,manifests,freeze):
    out=OUT/'cache'; out.mkdir(parents=True,exist_ok=True)
    log_path=OUT/'cache_index.json'
    if log_path.exists():
        index=json.loads(log_path.read_text())
        if index['train_freeze_sha256']!=sha256(OUT/'train_freeze.json'):
            raise RuntimeError('cache belongs to different train freeze')
    else: index=dict(train_freeze_sha256=sha256(OUT/'train_freeze.json'),items={},complete=False)
    started=time.time()
    for role in ('fit','val'):
        for i,item in enumerate(manifests[role]['items']):
            item_id=item['item_id']; path=out/(item_id+'.pt')
            if item_id in index['items']:
                if not path.exists() or sha256(path)!=index['items'][item_id]['sha256']:
                    raise RuntimeError('existing cached artifact corrupt: '+item_id)
                continue
            if path.exists(): raise RuntimeError('orphan cache file; inspect before resuming: '+str(path))
            if sha256(item['image_path'])!=item['image_sha256']: raise RuntimeError('image changed')
            t0=time.perf_counter(); msg=message(item); prep=eng.prepare(msg,'DocVQA_VAL')
            frozen=freeze['rendered_inputs'][item_id]
            if prep['ids'][0].cpu().tolist()!=frozen['input_ids'] or prep['n_vis']!=1024:
                raise RuntimeError('frozen native prompt changed')
            torch.cuda.synchronize(); tv=time.perf_counter(); V,DS=eng.encode(prep)
            q,_=eng.instruction_embeds(msg,'DocVQA_VAL');torch.cuda.synchronize();te=time.perf_counter()
            if V.shape!=(1024,4096) or len(DS)!=3 or any(d.shape!=V.shape for d in DS) or q.shape!=(1,4096):
                raise AssertionError('unexpected model feature shapes')
            if not all(t.dtype==torch.bfloat16 for t in [V,*DS,q]):
                raise AssertionError('unexpected cached feature dtype')
            torch.cuda.reset_peak_memory_stats()
            teacher=score_full_streams(eng,prep,V,DS,item['answers'][0],K=256,checkpointing=False)
            if teacher['gold_token_ids'].tolist()!=frozen['target_ids']:
                raise RuntimeError('frozen gold target changed')
            maps=teacher['scores']
            if set(maps)!=set(TARGETS): raise AssertionError('incorrect teacher map definitions')
            features=torch.stack([V,*DS]).detach().cpu().contiguous()
            question=q[0].detach().cpu().contiguous()
            artifact=dict(item_id=item_id,role=role,image_sha256=item['image_sha256'],
                features=features,question=question,teacher_scores=maps,
                teacher_loss=teacher['mean_answer_nll'],target_ids=teacher['gold_token_ids'],
                teacher_timing=teacher['timings'],feature_input_sha256=dict(streams=[tensor_sha256(v) for v in [V,*DS]],
                    question=tensor_sha256(q)),freeze_sha256=sha256(OUT/'train_freeze.json'))
            atomic_torch_save(artifact,path)
            index['items'][item_id]=dict(role=role,path=str(path),sha256=sha256(path),
                nll=teacher['mean_answer_nll'],extract_seconds=te-t0,vision_and_question_seconds=te-tv,
                teacher_seconds=teacher['timings']['total_label_seconds'],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),bytes=path.stat().st_size)
            save_json(log_path,index)
            print(f'{role} {i+1}/{len(manifests[role]["items"])} elapsed={time.time()-started:.1f}s nll={teacher["mean_answer_nll"]:.5g}',flush=True)
    index['complete']=True;index['elapsed_seconds_this_process']=time.time()-started
    save_json(log_path,index)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    check_gate()
    manifests={role:json.loads((DATA/f'{role}_manifest.json').read_text()) for role in ('fit','val')}
    for role,m in manifests.items():
        if not m.get('complete') or m['role']!=role: raise ValueError('incomplete data: '+role)
    model=common.load_model(common.BASELINE_MODEL,max_new_tokens=MAX_TOKENS)
    eng=NativeEngine(model)
    if any(p.is_inference() for p in eng.model.parameters()): raise RuntimeError('model loaded as inference tensors')
    freeze=verify_freeze() if args.resume else build_freeze(eng,manifests)
    run(eng,manifests,freeze)

if __name__=='__main__': main()
