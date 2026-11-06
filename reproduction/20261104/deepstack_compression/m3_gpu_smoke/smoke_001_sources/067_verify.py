"""Independent CPU acceptance for one completed no-GT gradient diagnostic."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from local_imports import local
C=local('contract');G=local('gate');HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;E4=ROOT/'e4'
sys.path.insert(1,str(E4))
from identity_worker import WEIGHTS
from completion import alive

def terminal(run,started,ended,launch,exited):
    if (ended.get('reason')!='completed' or ended.get('worker_returncode')!=0 or ended.get('remaining_process_group')!=[] or
        ended.get('worker_pid')!=started.get('worker_pid')):raise ValueError('smoke supervisor did not exit cleanly')
    controller=launch.get('controller');container=exited.get('container_identity')
    if (not isinstance(controller,dict) or controller!=exited.get('controller') or controller.get('pid')!=ended.get('worker_pid') or
        not isinstance(container,dict) or not container.get('container_id') or not isinstance(container.get('host_init'),dict)):
        raise ValueError('process identity mismatch')
    for ident in (controller,container['host_init']):
        if (not isinstance(ident.get('pid'),int) or ident['pid']<=0 or not isinstance(ident.get('start_ticks'),int) or
            ident['start_ticks']<=0 or not ident.get('boot_id') or alive(ident)):raise ValueError('process identity invalid or alive')
    if (exited.get('returncode')!=0 or exited.get('container_exit_code')!=0 or exited.get('timed_out') is not False or
        exited.get('interrupted') is not None or exited.get('container_stopped_verified') is not True or
        exited.get('client_exited_while_container_running') is not False or exited.get('gpu_pids_after')!=[] or exited.get('errors')!=[]):
        raise ValueError('container exit not clean')
    cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    if [(cwd/x).resolve() for x in cmd if Path(x).name=='launcher.py']!=[(HERE/'launcher.py').resolve()] or cmd.count('--run-id')!=1:
        raise ValueError('wrong smoke launcher')
    index=cmd.index('--run-id')
    if index+1>=len(cmd) or run.name!='smoke_'+cmd[index+1]:raise ValueError('wrong smoke run directory')
    plan=json.loads((HERE/'plan.json').read_text());initial=ROOT/'initial_plan.json'
    if (C.sha(initial)!=plan['budget_source_sha256'] or json.loads(initial.read_text()).get('budget_end_unix')!=plan['budget_end_unix'] or
        started.get('timeout_seconds')!=plan['outer_timeout_seconds'] or started.get('stale_seconds')!=plan['progress_stale_seconds'] or
        started.get('progress_paths')!=[str(run/'events.jsonl')] or started.get('snapshot_seconds')!=plan['snapshot_seconds'] or
        started.get('poll_seconds')!=plan['poll_seconds'] or started.get('grace_seconds')!=plan['grace_seconds'] or
        launch.get('timeout_seconds')!=plan['inner_timeout_seconds'] or launch['started_unix']+plan['outer_timeout_seconds']>plan['budget_end_unix']):
        raise ValueError('resource supervision differs from frozen plan')
    dock=launch.get('command',[])
    if (dock.count(str(HERE/'worker.py'))!=1 or dock.count('--out')!=1 or Path(dock[dock.index('--out')+1]).resolve()!=run or
        launch.get('worker_sha256')!=C.sha(HERE/'worker.py') or launch.get('launcher_sha256')!=C.sha(HERE/'launcher.py') or
        launch.get('approval_bundle')!=G.check_approval()):raise ValueError('worker, launcher or approval changed')
    receipt=run/'host_upstream.json'
    if launch.get('host_upstream')!={'path':str(receipt),'sha256':C.sha(receipt)} or json.loads(receipt.read_text())!=G.upstream(host=True):
        raise ValueError('upstream native receipt differs')
    p=subprocess.run(['docker','inspect',container['container_id']],capture_output=True,text=True,timeout=10,check=True)
    state=json.loads(p.stdout)[0]
    if state['Id']!=container['container_id'] or state['State']['Running'] or state['State']['ExitCode']!=0:
        raise ValueError('independent container inspection failed')

def verify_arrays(path,result):
    import torch
    expected={**{'feature'+str(i):m for i,m in enumerate(result['features'])},
              **{'gradient'+str(i):m for i,m in enumerate(result['gradients'])},
              **{name:result[name] for name in ('native_actual_logits','native_reference_logits','teacher_first_logits')}}
    sizes={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}
    with np.load(path,allow_pickle=False) as arrays:
        if set(arrays.files)!=set(expected):raise ValueError('diagnostic array inventory mismatch')
        tensors={}
        for name,meta in expected.items():
            raw=arrays[name]
            if raw.dtype!=np.uint8 or meta.get('dtype') not in sizes:raise ValueError('bad array dtype')
            value=torch.from_numpy(raw.copy()).view(sizes[meta['dtype']]).reshape(meta['shape'])
            if C.tensor_meta(value)!=meta or not torch.isfinite(value).all():raise ValueError('array mismatch: '+name)
            tensors[name]=value
        for i in range(4):
            f=tensors['feature'+str(i)];g=tensors['gradient'+str(i)]
            if tuple(f.shape)!=(1024,4096) or f.shape!=g.shape or int(torch.count_nonzero(g))!=result['gradient_nonzero_counts'][i] or not torch.count_nonzero(g):
                raise ValueError('gradient evidence invalid')
            u=(f.float()*g.float()).abs().mean(dim=-1)
            if (result['utility'][i]['length']!=1024 or
                not math.isclose(float(u.min()),result['utility'][i]['min'],rel_tol=1e-5,abs_tol=1e-12) or
                not math.isclose(float(u.max()),result['utility'][i]['max'],rel_tol=1e-5,abs_tol=1e-12)):
                raise ValueError('utility summary mismatch')
        comparison=C.compare_logits(tensors['native_actual_logits'],tensors['native_reference_logits'],
                                    json.loads((HERE/'plan.json').read_text())['native_first_logits_max_abs_tolerance'])
        if comparison!=result['native_first_logits_comparison']:raise ValueError('native logits comparison differs')
        teacher_comparison=C.compare_logits(tensors['teacher_first_logits'],tensors['native_reference_logits'],
                                            json.loads((HERE/'plan.json').read_text())['native_first_logits_max_abs_tolerance'])
        if teacher_comparison!=result['teacher_first_logits_comparison']:raise ValueError('teacher logits comparison differs')
        diff=float((tensors['teacher_first_logits'].float()-tensors['native_reference_logits'].float()).abs().max())
        if diff!=result['teacher_first_vs_native_max_abs']:raise ValueError('teacher first logits diagnostic differs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--run-started',type=Path,required=True)
    p.add_argument('--run-terminal',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
    launch=json.loads((run/'launch.json').read_text());exited=json.loads((run/'exit.json').read_text())
    terminal(run,json.loads(a.run_started.read_text()),json.loads(a.run_terminal.read_text()),launch,exited)
    frozen=json.loads((run/'frozen.json').read_text());result=json.loads((run/'result.json').read_text());plan=json.loads((HERE/'plan.json').read_text())
    if ((run/'failure.json').exists() or frozen.get('stage')!='synthetic_target_gradient_smoke' or frozen.get('plan_sha256')!=C.sha(HERE/'plan.json') or
        frozen.get('approval')!=launch['approval_bundle'] or frozen.get('target_ids')!=[100,200,300] or result.get('target_ids')!=[100,200,300] or
        any(frozen.get(k) is not False or result.get(k) is not False for k in ('gt_read','scoring','generation')) or
        result.get('model_fully_frozen') is not True or result.get('model_eval') is not True or result.get('gradient_autograd_enabled') is not True or
        not math.isfinite(result['synthetic_loss']) or result['synthetic_loss']<0):raise ValueError('invalid no-GT experiment record')
    if any(result.get(k)!=plan[k] for k in ('task','qid','group_id')):raise ValueError('sample identity changed')
    row,ref=C.anchor(ROOT/'m3_oracle/sample_001.json',ROOT/'e4_native/native_001/records.jsonl',plan['task'],plan['qid'],plan['group_id'])
    if result['processor']!=ref['processor'] or result['features']!=[ref['main_visual'],*ref['full_deepstack']] or result['native_reference_logits']!=ref['prefill_logits']:
        raise ValueError('actual input/feature/logit differs from native anchor')
    if result['native_rope_delta']!=int(ref['rope_delta'][0][0]) or result['prompt_length']!=ref['processor']['input_ids']['shape'][1]:
        raise ValueError('native P3 provenance differs')
    for i in range(2):
        x=result['prompt_length']+result['native_rope_delta']+i
        if [v[0][i] for v in result['teacher_prefix_positions']]!=[x,x,x]:raise ValueError('wrong synthetic prefix positions')
    archive=result['arrays'];path=Path(archive['path']).resolve()
    if path!=run/'arrays.npz' or C.sha(path)!=archive['sha256']:raise ValueError('array archive changed')
    verify_arrays(path,result)
    if (not 0<result['model_load_seconds']<=result['total_seconds'] or result['peak_allocated_bytes']<=0 or result['peak_reserved_bytes']<=0):
        raise ValueError('resource measurements missing')
    identity=json.loads(WEIGHTS.read_text());stats=frozen['weight_stats_before']
    if frozen.get('weights_identity_sha256')!=C.sha(WEIGHTS) or len(identity['files'])!=4 or set(stats)!={x['path'] for x in identity['files']}:
        raise ValueError('weight identity mismatch')
    for item in identity['files']:
        s=Path(item['path']).stat()
        if s.st_size!=item['bytes'] or [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=stats[item['path']]:
            raise ValueError('model weight file changed')
    files=[a.run_started,a.run_terminal]+[run/x for x in ('launch.json','exit.json','frozen.json','result.json','events.jsonl','arrays.npz')]
    report={'status':'passed','stage':'synthetic_target_gradient_smoke','task':plan['task'],'qid':plan['qid'],'gradients':4,
        'target_ids':plan['target_token_ids'],'GT_opened':False,'scoring_performed':False,'generation_performed':False,
        'peak_allocated_bytes':result['peak_allocated_bytes'],'peak_reserved_bytes':result['peak_reserved_bytes'],
        'model_load_seconds':result['model_load_seconds'],'total_seconds':result['total_seconds'],
        'native_first_logits_comparison':result['native_first_logits_comparison'],
        'teacher_first_logits_comparison':result['teacher_first_logits_comparison'],
        'scope':'single DEV image, arbitrary synthetic targets, no quality or oracle claim',
        'files_sha256':{str(p.resolve()):C.sha(p) for p in files},'verifier_sha256':C.sha(__file__)}
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')

if __name__=='__main__':main()
