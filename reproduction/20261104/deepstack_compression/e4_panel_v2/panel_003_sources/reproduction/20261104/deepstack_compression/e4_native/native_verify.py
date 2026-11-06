"""Independent acceptance for 3 native/K1024 DEV diagnostic pairs, no GT."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from native_contract import HERE,E4,select_anchors,compare_pair
from native_gate import check_gates
sys.path.insert(0,str(E4))
from identity_worker import PLAN,sha,read_jsonl
from completion import alive
from decode_observer import validate_trace

def terminal(run,started,ended,launch,exited):
    if (ended.get('reason')!='completed' or ended.get('worker_returncode')!=0 or ended.get('remaining_process_group')!=[] or
        ended.get('worker_pid')!=started.get('worker_pid')):raise ValueError('supervisor did not exit cleanly')
    controller=launch.get('controller');container=exited.get('container_identity')
    if (not isinstance(controller,dict) or controller!=exited.get('controller') or controller.get('pid')!=ended.get('worker_pid') or
        not isinstance(container,dict) or not container.get('container_id') or not isinstance(container.get('host_init'),dict)):
        raise ValueError('process identity mismatch')
    for ident in (controller,container['host_init']):
        if (not isinstance(ident.get('pid'),int) or ident['pid']<=0 or not isinstance(ident.get('start_ticks'),int) or
            ident['start_ticks']<=0 or not ident.get('boot_id') or alive(ident)):raise ValueError('process still alive or identity incomplete')
    if (exited.get('returncode')!=0 or exited.get('container_exit_code')!=0 or exited.get('timed_out') is not False or
        exited.get('interrupted') is not None or exited.get('container_stopped_verified') is not True or
        exited.get('client_exited_while_container_running') is not False or exited.get('gpu_pids_after')!=[]):
        raise ValueError('container exit is not clean')
    cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    scripts=[(cwd/x).resolve() for x in cmd if Path(x).name=='native_launch.py']
    if scripts!=[(HERE/'native_launch.py').resolve()] or cmd.count('--run-id')!=1:raise ValueError('wrong native launcher')
    i=cmd.index('--run-id')
    if i+1>=len(cmd) or run.name!='native_'+cmd[i+1]:raise ValueError('wrong native run directory')
    dock=launch.get('command',[])
    if (dock.count(str(HERE/'native_worker.py'))!=1 or dock.count('--out')!=1 or Path(dock[dock.index('--out')+1]).resolve()!=run or
        launch.get('worker_sha256')!=sha(HERE/'native_worker.py') or launch.get('launcher_sha256')!=sha(HERE/'native_launch.py')):
        raise ValueError('wrong worker/launcher identity')
    from native_launch import check_approval
    if launch.get('approval_bundle')!=check_approval():raise ValueError('native approval changed')
    ins=subprocess.run(['docker','inspect',container['container_id']],capture_output=True,text=True,timeout=10,check=True)
    state=json.loads(ins.stdout)[0]
    if state['Id']!=container['container_id'] or state['State']['Running'] or state['State']['ExitCode']!=0:
        raise ValueError('independent container inspection failed')


def verify_array(record,path,arrays):
    import numpy as np
    names={'visual','prefill','prefill_logits','ds0','ds1','ds2','injected0','injected1','injected2',
           'processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw'}
    if set(arrays)!=names:raise ValueError('native diagnostic array inventory mismatch')
    values={'visual':record['main_visual'],'prefill':record['prefill_embeds'],'prefill_logits':record['prefill_logits'],
        **{'ds'+str(i):x for i,x in enumerate(record['full_deepstack'])},
        **{'injected'+str(i):x for i,x in enumerate(record['injected_deepstack'])},
        **{'processor_'+k:record['processor'][k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}}
    sizes={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8,'torch.int64':8,'torch.int32':4,'torch.uint8':1,'torch.bool':1}
    for name,meta in values.items():
        a=arrays[name];n=1
        for d in meta.get('shape',[]):n*=d
        size=sizes.get(meta.get('dtype'))
        if size is None or a.dtype!=np.uint8 or a.size!=n*size or hashlib.sha256(a.tobytes()).hexdigest()!=meta.get('sha256'):
            raise ValueError('array does not match actual metadata: '+name)
    meta=values['prefill_logits'];raw=arrays['prefill_logits'].tobytes();dtype=meta['dtype']
    if dtype=='torch.bfloat16':v=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
    elif dtype in ('torch.float16','torch.float32','torch.float64'):v=np.frombuffer(raw,dtype={'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8'}[dtype])
    else:raise ValueError('invalid logits dtype')
    if not v.size or not np.isfinite(v).all():raise ValueError('nonfinite actual prefill logits')


def main():
    p=argparse.ArgumentParser()
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal',
                 'run-dir','run-started','run-terminal','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();e1=check_gates(a);anchors=select_anchors(e1);run=a.run_dir.resolve()
    launch=json.loads((run/'launch.json').read_text());exited=json.loads((run/'exit.json').read_text())
    terminal(run,json.loads(a.run_started.read_text()),json.loads(a.run_terminal.read_text()),launch,exited)
    frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text())
    fields=('task','qid','group_id','message_sha256','image_sha256')
    if (frozen.get('stage')!='stock_native_k1024' or frozen.get('n_questions')!=3 or
        frozen.get('anchors')!=[{k:r[k] for k in fields} for r in anchors] or frozen.get('paths')!=['stock_native','budget1024'] or
        frozen.get('modes')!=['natural','forced33'] or frozen.get('e1_acceptance_sha256')!=sha(a.e1_acceptance) or
        frozen.get('smoke_proof_sha256')!=sha(a.smoke_proof) or frozen.get('plan_sha256')!=sha(PLAN) or
        frozen.get('gt_read') is not False or frozen.get('scoring') is not False or frozen.get('timing_claims') is not False or
        done.get('n_questions')!=3 or done.get('n_generations')!=12 or done.get('n_pairs')!=6):
        raise ValueError('native frozen identity mismatch')
    for path,digest in frozen.get('code_sha256',{}).items():
        if sha(path)!=digest:raise ValueError('native run source changed')
    required=[HERE/x for x in ('native_worker.py','native_launch.py','native_gate.py','native_contract.py','native_capture.py','native_verify.py')]
    required += [E4/x for x in ('budget_runner.py','budget_adapter.py','decode_observer.py','off_adapter.py','identity_records.py')]
    if any(frozen.get('code_sha256',{}).get(str(p))!=sha(p) for p in required):raise ValueError('native source binding missing')
    records=read_jsonl(run/'records.jsonl');generated=read_jsonl(run/'generated.jsonl');comparisons=read_jsonl(run/'comparisons.jsonl')
    if len(records)!=12 or len(generated)!=12 or len(comparisons)!=6:raise ValueError('native inventory incomplete')
    from transformers import AutoTokenizer
    import numpy as np
    tokenizer=AutoTokenizer.from_pretrained(json.loads(PLAN.read_text())['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
    fixed=tokenizer.encode(' the',add_special_tokens=False)
    if not fixed:raise ValueError('forced token unavailable')
    arrays_hash={};index=0
    for anchor in anchors:
        for forced in (False,True):
            pair=records[index:index+2];events=generated[index:index+2];index+=2
            for r,event,path in zip(pair,events,('stock_native','budget1024')):
                if any(str(r.get(k))!=str(anchor[k]) for k in fields) or r.get('forced') is not forced:raise ValueError('native anchor mismatch')
                event_fields=('task','qid','group_id','forced','output_ids','eos_ids','generation_scalars')
                if any(event.get(k)!=r.get(k) for k in event_fields) or event.get('path')!=path:raise ValueError('generated journal mismatch')
                ids=r['output_ids'];eos=r['eos_ids']
                if forced:
                    if ids!=[fixed[0]]*33 or eos!=[] or r['eos'] or r['truncated']:raise ValueError('forced identity mismatch')
                elif (not ids or len(ids)>2048 or r['eos']!=bool(ids[-1] in eos) or
                      r['truncated']!=bool(len(ids)==2048 and ids[-1] not in eos) or not (r['eos'] or r['truncated'])):
                    raise ValueError('natural output endpoint invalid')
                if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['answer']:
                    raise ValueError('independent output decoding mismatch')
                full_len=len(r['prefill_position_ids'][0][0]);delta=(r['rope_delta'] if path=='stock_native' else r['adjusted_delta'])[0][0]
                validate_trace(r['decode_trace'],ids,full_len,delta)
                if r['decode_trace'][0]['position_ids']!=r['prefill_position_ids']:raise ValueError('prefill position trace mismatch')
                fp=r['cache_fingerprints']
                if len(fp)!=(33 if forced else 1) or any(len(step)!=36 or any(set(layer)!={'key','value'} for layer in step) for step in fp):
                    raise ValueError('KV fingerprint inventory incomplete')
                archive=Path(r['arrays']['path']).resolve()
                name=hashlib.sha256(json.dumps([anchor['task'],str(anchor['qid']),path,forced],separators=(',',':')).encode()).hexdigest()+'.npz'
                if r['arrays'].get('kind')!='full_arrays' or archive.parent!=(run/'arrays').resolve() or archive.name!=name or sha(archive)!=r['arrays']['sha256']:
                    raise ValueError('native array archive mismatch')
                with np.load(archive,allow_pickle=False) as payload:verify_array(r,path,payload)
                arrays_hash[str(archive)]=r['arrays']['sha256']
            compare_pair(*pair)
            with np.load(pair[0]['arrays']['path'],allow_pickle=False) as left,np.load(pair[1]['arrays']['path'],allow_pickle=False) as right:
                if set(left)!=set(right) or any(not np.array_equal(left[k],right[k]) for k in left):raise ValueError('actual native/budget tensors differ')
            marker=comparisons[(index//2)-1]
            if marker!={'task':anchor['task'],'qid':str(anchor['qid']),'group_id':anchor['group_id'],'forced':forced,'status':'passed'}:
                raise ValueError('native comparison marker mismatch')
    if len(arrays_hash)!=12:raise ValueError('native archive inventory mismatch')
    files=[a.e1_acceptance,a.e1_started,a.e1_terminal,a.smoke_proof,a.smoke_started,a.smoke_terminal,a.run_started,a.run_terminal,
           run/'launch.json',run/'exit.json',run/'frozen.json',run/'worker_complete.json',run/'records.jsonl',run/'generated.jsonl',run/'comparisons.jsonl']
    result={'status':'passed','stage':'stock_native_k1024','tasks':3,'pairs':6,'natural_pairs':3,'forced_pairs':3,
            'forced_decode_steps':32,'kv_fingerprints_compared':True,'GT_opened':False,'scoring_performed':False,
            'files_sha256':{str(p.resolve()):sha(p) for p in files},'array_sha256':arrays_hash,'verifier_sha256':sha(__file__)}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
