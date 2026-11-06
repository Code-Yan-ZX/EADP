"""Independent CPU acceptance for the DEV-only legacy-vs-P1 anchor run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
import time
from contract import sha, read_jsonl, choose_legacy_dev, validate_decode_trace, validate_anchor_exclusion
from runner import E0, PLAN, validate_gate, verified_inputs
from legacy_anchor import PANEL


def main():
    p=argparse.ArgumentParser()
    for name in ('run-dir','e0-proof','manifest-dir','inputs','input-proof','terminal','container-exit','launch','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-n',type=int,choices=(1,16),required=True)
    a=p.parse_args(); errors=[]
    for name in ('run_dir','e0_proof','manifest_dir','inputs','input_proof','terminal','container_exit','launch','out'):
        setattr(a,name,getattr(a,name).resolve())
    try:
        e0=validate_gate(a.e0_proof)
        records=verified_inputs(a.manifest_dir,a.inputs,a.input_proof)
        panel,old_groups=choose_legacy_dev(records,read_jsonl(a.manifest_dir/'full_manifest.jsonl'),json.loads(PANEL.read_text())['panel'])
        panel=panel[:a.expected_n]
        frozen=json.loads((a.run_dir/'frozen.json').read_text())
        expected={'e0_proof_sha256':sha(a.e0_proof),'e0_adapter_sha256':e0['adapter_sha256'],
                  'e0_verifier_terminal_sha256':sha(E0/'monitor_verify_array16/terminal.json'),
                  'position_adapter_sha256':sha(Path(__file__).with_name('position_adapter.py')),
                  'worker_sha256':sha(Path(__file__).with_name('legacy_anchor.py')),
                  'runner_sha256':sha(Path(__file__).with_name('runner.py')),
                  'contract_sha256':sha(Path(__file__).with_name('contract.py')),
                  'anchor_exclusion':validate_anchor_exclusion(a.manifest_dir,PANEL),
                  'old_anchor_panel_sha256':sha(PANEL),'panel_sha256':sha(a.manifest_dir/'panel_manifest.jsonl'),
                  'full_manifest_sha256':sha(a.manifest_dir/'full_manifest.jsonl'),
                  'input_sha256':sha(a.inputs),'input_proof_sha256':sha(a.input_proof), 'recipe_sha256':sha(PLAN)}
        if any(frozen.get(k)!=v for k,v in expected.items()): raise ValueError('source or input identity changed')
        selected=[{k:r[k] for k in ('task','qid','group_id')} for r in panel]
        if frozen['selected_dev']!=selected or frozen['excluded_old_anchor_groups']!=sorted(old_groups): raise ValueError('DEV selection changed')
        complete=json.loads((a.run_dir/'worker_complete.json').read_text())
        if complete.get('n')!=a.expected_n or complete.get('comparisons')!=2*a.expected_n or complete.get('forced_decode_steps')!=32: raise ValueError('worker did not finish')
        terminal=json.loads(a.terminal.read_text()); ex=json.loads(a.container_exit.read_text()); launch=json.loads(a.launch.read_text())
        if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group'):
            raise ValueError('supervisor did not finish cleanly')
        if (ex.get('returncode')!=0 or ex.get('timed_out') or ex.get('interrupted') is not None or
            ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after') or not ex.get('container_identity')):
            raise ValueError('container exit evidence incomplete')
        identity=ex['container_identity']; process=identity.get('host_init')
        if not process or not identity.get('container_id'): raise ValueError('container process identity missing')
        stat=Path('/proc')/str(process['pid'])/'stat'
        if stat.exists() and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']:
            if int(stat.read_text().rsplit(')',1)[1].split()[19])==process['start_ticks']:
                raise ValueError('original container worker still alive')
        probe=subprocess.run(['docker','inspect',identity['container_id']],capture_output=True,text=True,timeout=10)
        if probe.returncode==0:
            if json.loads(probe.stdout)[0]['State']['Running']: raise ValueError('container still running')
        elif 'No such object' not in probe.stderr and 'No such container' not in probe.stderr:
            raise ValueError('cannot confirm container state')
        if launch.get('source_sha256',{}).get(str(Path(__file__).with_name('legacy_anchor.py')))!=expected['worker_sha256']:
            raise ValueError('launch does not bind worker source')
        observations=read_jsonl(a.run_dir/'comparisons.jsonl')
        if len(observations)!=2*a.expected_n: raise ValueError('missing comparisons')
        arms=read_jsonl(a.run_dir/'arms.jsonl')
        if len(arms)!=4*a.expected_n: raise ValueError('missing individual arm records')
        import numpy as np
        from transformers import AutoTokenizer
        recipe=json.loads(PLAN.read_text())['recipe']; model_path=Path(recipe['model_path'])
        if recipe.get('post_process',False): raise ValueError('independent decoding requires post_process disabled')
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        forced_token=tokenizer.encode(' the',add_special_tokens=False)[0]
        model_config=json.loads((model_path/'config.json').read_text())
        generation_config=json.loads((model_path/'generation_config.json').read_text())
        eos=generation_config['eos_token_id']; eos=[eos] if isinstance(eos,int) else eos
        natural_endings=[]
        for i,row in enumerate(panel):
            for k,forced in enumerate((False,True)):
                rec=observations[i*2+k]
                if rec['qid']!=row['qid'] or rec['group_id']!=row['group_id'] or rec['forced']!=forced:
                    raise ValueError('wrong sample order or split')
                old,new=rec['legacy'],rec['p1']
                for value in (old,new):
                    decoded=tokenizer.batch_decode(value['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    if len(decoded)!=1 or decoded[0]!=value['answer']: raise ValueError('independent token re-decode mismatch')
                for j,(label,value) in enumerate((('legacy',old),('p1',new))):
                    if arms[(i*2+k)*2+j]!={'qid':row['qid'],'group_id':row['group_id'],'forced':forced,'arm':label,'observation':value}:
                        raise ValueError('individual arm record differs from comparison')
                base_keys=('processor','ordered_ids','llm','output_ids','answer','logits','array_keys')
                if any(old[k]!=new[k] for k in base_keys): raise ValueError('actual legacy and P1 observations differ')
                arrays=[]
                for label,value in (('legacy',old),('p1',new)):
                    path=Path(value['array_path'])
                    if path.parent!=a.run_dir or path.name!=f'{row["qid"]}_{"forced" if forced else "natural"}_{label}.npz' or sha(path)!=value['array_sha256']:
                        raise ValueError('tensor archive path or hash mismatch')
                    archive=np.load(path,allow_pickle=False)
                    if sorted(archive.files)!=value['array_keys']: raise ValueError('tensor archive inventory mismatch')
                    required=['processor_'+k for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')]
                    required += ['visual_main','visual_ds_0','visual_ds_1','visual_ds_2','pruned_main',
                                 'prefill_inputs_embeds','prefill_attention_mask','prefill_position_ids','prefill_cache_position']
                    required += ['logits_'+str(j) for j in range(33 if forced else 1)]
                    if sorted(required)!=value['array_keys']: raise ValueError('diagnostic tensor inventory incomplete')
                    checks={**{'processor_'+k:v for k,v in value['processor'].items()},
                            **{'prefill_'+k:value['llm'][0][k] for k in ('inputs_embeds','attention_mask')},
                            **{'logits_'+str(j):v for j,v in enumerate(value['logits'])}}
                    for key,expected_meta in checks.items():
                        if hashlib.sha256(archive[key].tobytes()).hexdigest()!=expected_meta['sha256']:
                            raise ValueError('tensor bytes differ from recorded metadata: '+key)
                    pos=np.frombuffer(archive['prefill_position_ids'].tobytes(),dtype=np.int64).reshape(3,1,-1).tolist()
                    cache=np.frombuffer(archive['prefill_cache_position'].tobytes(),dtype=np.int64).tolist()
                    if pos!=value['llm'][0]['position_ids'] or cache!=value['llm'][0]['cache_position']:
                        raise ValueError('prefill position/cache bytes differ from trace')
                    arrays.append(archive)
                try:
                    if any(not np.array_equal(arrays[0][key],arrays[1][key]) for key in required): raise ValueError('legacy and P1 tensor bytes differ')
                    ids=np.frombuffer(arrays[0]['processor_input_ids'].tobytes(),dtype=np.int64)
                    image_positions=np.flatnonzero(ids==model_config['image_token_id'])
                    if len(image_positions)!=1024 or not np.array_equal(image_positions,np.arange(image_positions[0],image_positions[0]+1024)):
                        raise ValueError('processor visual span incorrect')
                    ordered=np.asarray(old['ordered_ids'],dtype=np.int64)
                    selected=arrays[0]['visual_main'][np.sort(ordered)]
                    if not np.array_equal(selected,arrays[0]['pruned_main']): raise ValueError('actual selected IDs do not reconstruct pruned vision')
                    prefill=arrays[0]['prefill_inputs_embeds']
                    start=int(image_positions[0])
                    if not np.array_equal(prefill[0,start:start+256],selected): raise ValueError('selected features do not match LLM prefill')
                finally:
                    for archive in arrays: archive.close()
                ordered=old['ordered_ids']
                if len(ordered)!=256 or len(set(ordered))!=256 or any(v<0 or v>=1024 for v in ordered): raise ValueError('invalid visual selection')
                if len(old['logits'])!=(33 if forced else 1): raise ValueError('logit observation count')
                if forced:
                    if len(old['llm'])!=33: raise ValueError('forced decode step count')
                    if old['output_ids']!=[[forced_token]*33]: raise ValueError('forced token prefix differs from frozen rule')
                    length=len(old['llm'][0]['position_ids'][0][0]); delta=old['llm'][0]['rope_delta'][0][0]
                    if delta!=0: raise ValueError('legacy delta is not zero')
                    validate_decode_trace(old['llm'],length,delta)
                else:
                    tokens=old['output_ids'][0]
                    ended=bool(tokens and tokens[-1] in eos)
                    truncated=bool(len(tokens)>=recipe['max_new_tokens'] and not ended)
                    if not ended and not truncated: raise ValueError('natural output neither EOS nor max length')
                    natural_endings.append({'qid':row['qid'],'eos':ended,'truncated':truncated,'output_tokens':len(tokens)})
        for path,before in frozen['weight_stats_before'].items():
            st=Path(path).stat()
            if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=before: raise ValueError('weight stat changed after run')
    except Exception as exc:
        traceback.print_exc()
        errors.append(repr(exc))
    proof={'status':'failed' if errors else 'passed','errors':errors,'n_expected':a.expected_n,
           'n_verified':0 if errors else a.expected_n,'verifier_sha256':sha(__file__),
           'e0_proof_sha256':sha(a.e0_proof),'comparisons_sha256':sha(a.run_dir/'comparisons.jsonl') if (a.run_dir/'comparisons.jsonl').exists() else None,
           'frozen_sha256':sha(a.run_dir/'frozen.json') if (a.run_dir/'frozen.json').exists() else None,
           'worker_complete_sha256':sha(a.run_dir/'worker_complete.json') if (a.run_dir/'worker_complete.json').exists() else None,
           'terminal_sha256':sha(a.terminal),'container_exit_sha256':sha(a.container_exit), 'launch_sha256':sha(a.launch),
           'natural_endings':natural_endings if not errors else [],'verified_unix':time.time()}
    with a.out.open('x') as f: json.dump(proof,f,indent=2); f.write('\n')
    print(json.dumps(proof,ensure_ascii=False))
    return not errors

if __name__=='__main__': raise SystemExit(0 if main() else 1)
