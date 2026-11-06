"""CPU validation of E1 32-question four-arm diagnostic, no scoring or GT."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import numpy as np

from contract import ARMS, read_jsonl, sha, validate_quartet, validate_decode_trace
from runner import ROOT, PLAN, validate_gate, validate_legacy_gate, verified_inputs


def main():
    parser=argparse.ArgumentParser()
    for name in ('run-dir','e0-proof','manifest-dir','inputs','input-proof','terminal','container-exit','launch','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    for key in vars(a): setattr(a,key,getattr(a,key).resolve())
    errors=[]; budget={}; verified=0
    try:
        validate_gate(a.e0_proof); validate_legacy_gate()
        inputs=verified_inputs(a.manifest_dir,a.inputs,a.input_proof)
        panel=[r for task,n in (('DocVQA_VAL',11),('TextVQA_VAL',11),('AI2D_TEST',10)) for r in [x for x in inputs if x['task']==task][:n]]
        if len(panel)!=32: raise ValueError('smoke panel count')
        frozen=json.loads((a.run_dir/'frozen.json').read_text()); done=json.loads((a.run_dir/'worker_complete.json').read_text())
        if frozen['stage']!='smoke' or frozen['n_questions']!=32 or done['n']!=32: raise ValueError('run identity/incomplete worker')
        launch=json.loads(a.launch.read_text()); ex=json.loads(a.container_exit.read_text()); terminal=json.loads(a.terminal.read_text())
        if terminal['reason']!='completed' or terminal['worker_returncode']!=0 or terminal['remaining_process_group']:
            raise ValueError('supervisor did not exit cleanly')
        if ex['returncode']!=0 or ex['timed_out'] or ex['interrupted'] is not None or ex['client_exited_while_container_running'] or not ex['container_stopped_verified'] or ex['gpu_pids_after']:
            raise ValueError('container exit incomplete')
        if launch['controller']!=ex['controller'] or launch['controller']['pid']!=terminal['worker_pid']:
            raise ValueError('launcher/supervisor identity mismatch')
        ci=ex['container_identity']; host=ci['host_init']; proc=Path('/proc')/str(host['pid'])/'stat'
        if proc.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==host['boot_id'] and int(proc.read_text().rsplit(')',1)[1].split()[19])==host['start_ticks']:
            raise ValueError('original worker still running')
        probe=subprocess.run(['docker','inspect',ci['container_id']],capture_output=True,text=True,timeout=10)
        if probe.returncode==0:
            if json.loads(probe.stdout)[0]['State']['Running']: raise ValueError('container still running')
        elif 'No such object' not in probe.stderr and 'No such container' not in probe.stderr: raise ValueError('cannot verify container state')
        source_dir=a.run_dir.parent/(a.run_dir.name+'_sources')
        for path,digest in launch['source_sha256'].items():
            if sha(source_dir/Path(path).name)!=digest: raise ValueError('source snapshot mismatch: '+path)
        if frozen['runner_sha256']!=launch['source_sha256'][str(a.run_dir.parent/'runner.py')] or frozen['position_adapter_sha256']!=launch['source_sha256'][str(a.run_dir.parent/'position_adapter.py')]:
            raise ValueError('worker and launch sources differ')
        records=read_jsonl(a.run_dir/'raw_predictions.jsonl'); generated=read_jsonl(a.run_dir/'generated_arms.jsonl'); quartets=read_jsonl(a.run_dir/'validated_quartets.jsonl')
        if len(records)!=256 or len(generated)!=256 or len(quartets)!=64: raise ValueError('journal count mismatch')
        from transformers import AutoTokenizer
        recipe=json.loads(PLAN.read_text())['recipe']; tok=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True)
        fixed=tok.encode(' the',add_special_tokens=False)[0]
        if recipe.get('post_process',False): raise ValueError('unsupported response postprocessing')
        times={}
        for i,p in enumerate(panel):
            for f,forced in enumerate((False,True)):
                batch=records[(i*2+f)*4:(i*2+f+1)*4]
                q=quartets[i*2+f]
                if q!={'task':p['task'],'qid':p['qid'],'group_id':p['group_id'],'forced':forced}: raise ValueError('quartet identity mismatch')
                validate_quartet(batch)
                for j,r in enumerate(batch):
                    if any(r[k]!=p[k] for k in ('task','qid','group_id','message_sha256')) or r['arm']!=ARMS[j] or r['forced']!=forced:
                        raise ValueError('arm identity mismatch')
                    gen=generated[(i*2+f)*4+j]
                    if gen!={k:r[k] for k in ('task','qid','group_id','arm','forced','output_ids','eos_ids')}:
                        raise ValueError('generated token journal mismatch')
                    if tok.batch_decode([r['output_ids']],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=[r['answer']]:
                        raise ValueError('token re-decode mismatch')
                    if forced:
                        if r['output_ids']!=[fixed]*33 or r['forced_token']!=fixed: raise ValueError('forced token mismatch')
                        trace=r['decode_trace']; n=len(r['retained_ids'])
                        validate_decode_trace(trace,n,r['rope_delta'][0][0])
                        if len(trace)!=33: raise ValueError('decode count')
                        for step,t in enumerate(trace):
                            if t['rope_delta']!=r['rope_delta'] or len(t['layer_cache_lengths_after'])!=36 or any(z!=n+step for z in t['layer_cache_lengths_after']):
                                raise ValueError('cache or delta trace mismatch')
                    else:
                        eos=bool(r['output_ids'] and r['output_ids'][-1] in r['eos_ids'])
                        truncated=not eos and len(r['output_ids'])>=recipe['max_new_tokens']
                        if r['eos']!=eos or r['truncated']!=truncated or not (eos or truncated): raise ValueError('natural EOS/length')
                    path=Path(r['array_path'])
                    if path.parent!=a.run_dir or sha(path)!=r['array_sha256']: raise ValueError('array SHA/path')
                    with np.load(path,allow_pickle=False) as z:
                        if sorted(z.files)!=r['array_keys']: raise ValueError('array keys')
                        checks={**{'processor_'+k:v for k,v in r['processor'].items()},
                                'visual_main':r['visual_sha256'],'pruned_main':r['pruned_sha256'],
                                'prefill_inputs_embeds':r['prefill_embed_sha256'],'prefill_attention_mask':r['attention_mask_sha256'],
                                **{'visual_ds_'+str(k):v for k,v in enumerate(r['deepstack_full_sha256'])},
                                **{'selected_ds_'+str(k):v for k,v in enumerate(r['selected_deepstack'])},
                                **{'logits_'+str(k):v for k,v in enumerate(r['logits'])}}
                        for key,m in checks.items():
                            if hashlib.sha256(z[key].tobytes()).hexdigest()!=m['sha256']: raise ValueError('tensor hash mismatch: '+key)
                        selected=np.asarray(r['spatial_ids'],dtype=np.int64)
                        if not np.array_equal(z['visual_main'][selected],z['pruned_main']): raise ValueError('selection/main reconstruction')
                        for k in range(3):
                            if not np.array_equal(z['visual_ds_'+str(k)][selected],z['selected_ds_'+str(k)]): raise ValueError('selection/DS reconstruction')
                            if r['arm'].endswith('D1') and not np.array_equal(z['prefill_ds_'+str(k)],z['selected_ds_'+str(k)]): raise ValueError('injected DS mismatch')
                        start=r['image_start']
                        if not np.array_equal(z['prefill_inputs_embeds'][0,start:start+256],z['pruned_main']): raise ValueError('LLM visual input mismatch')
                        if r['arm'].endswith('D0') and ('prefill_visual_pos_masks' in z or any('prefill_ds_'+str(k) in z for k in range(3))): raise ValueError('D0 injection')
                    key=p['task']+'|'+r['arm']+'|'+('forced' if forced else 'natural')
                    times.setdefault(key,[]).append(r['wall_seconds_through_archive'])
            verified+=1
        natural_total=sum(statistics.mean(times[f'{task}|{arm}|natural'])*128 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for arm in ARMS)
        budget={'smoke_launcher_elapsed_seconds':terminal['elapsed_seconds'],
                'per_task_arm_seconds':{key:{'count':len(v),'mean':statistics.mean(v),'min':min(v),'max':max(v)} for key,v in sorted(times.items())},
                'dev_natural_extrapolated_gpu_hours_mean':natural_total/3600,
                'qualification':'exploratory extrapolation from 11/11/10 questions per task; per-arm times include observer and diagnostic NPZ persistence, model startup separately, 32-step teacher-forced diagnostics add overhead; not E4 performance or K128/K512/K1024 estimates'}
    except Exception as exc:
        traceback.print_exc(); errors.append(repr(exc))
    proof={'status':'failed' if errors else 'passed','errors':errors,'n_expected':32,'n_verified':verified,
           'verifier_sha256':sha(__file__),'frozen_sha256':sha(a.run_dir/'frozen.json'),
           'raw_predictions_sha256':sha(a.run_dir/'raw_predictions.jsonl'),'generated_arms_sha256':sha(a.run_dir/'generated_arms.jsonl'),
           'validated_quartets_sha256':sha(a.run_dir/'validated_quartets.jsonl'),'worker_complete_sha256':sha(a.run_dir/'worker_complete.json'),
           'launch_sha256':sha(a.launch),'container_exit_sha256':sha(a.container_exit),'supervisor_terminal_sha256':sha(a.terminal),
           'budget':budget,'verified_unix':time.time()}
    with a.out.open('x') as f: json.dump(proof,f,indent=2); f.write('\n')
    print(json.dumps(proof,ensure_ascii=False)); return not errors

if __name__=='__main__': raise SystemExit(0 if main() else 1)
