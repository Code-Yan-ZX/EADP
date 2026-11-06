"""Independent prelaunch review of the three-anchor four-budget TTFT pilot."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_budget_ttft_pilot_v3';I=B/'e1_independent_review';P=B/'e4_c_budget_pilot_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervised(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script:raise ValueError('supervisor failure')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=I/'e4c_budget_ttft_v3_prelaunch_review.json'
 if a.out.resolve()!=out:raise ValueError('output')
 runtime_path=D/'runtime_plan001.json';runtime=read(runtime_path);sources=runtime['source_sha256'];generation=runtime['generation_source_sha256']
 if sha(runtime_path)!='26762b5b30ed28509309e31cf47ef1bbcddef5f8cacf8fddf85c74bb1e32766a' or len(sources)!=185 or len(generation)!=7 or any(sha(k)!=v for k,v in sources.items()) or any(sources.get(k)!=v or sha(k)!=v for k,v in generation.items()):raise ValueError('source inventory')
 if runtime['jobs']!=48 or runtime['GT_opened'] is not False or runtime['scoring_performed'] is not False or (runtime['inner_timeout_seconds'],runtime['outer_timeout_seconds'],runtime['stale_seconds'],runtime['grace_seconds'])!=(3600,3720,1200,90):raise ValueError('runtime contract')
 cs=D/'monitor_cpu002/started.json';ct=D/'monitor_cpu002/terminal.json';c=read(cs);t=read(ct)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or c.get('worker_pid')!=t.get('worker_pid') or sha(cs)!=runtime['cpu_started_sha256'] or sha(ct)!=runtime['cpu_terminal_sha256'] or 'Ran 21 tests' not in (D/'monitor_cpu002/stderr.log').read_text():raise ValueError('CPU tests incomplete')
 spec=importlib.util.spec_from_file_location('_budget_ttft_v3_independent_identity',D/'identity_gate.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity);accepted=identity.admit_identity()
 plan=read(D/'pilot_plan001.json');anchors=plan['anchors'];jobs=plan['jobs']
 if sha(D/'pilot_plan001.json')!=runtime['pilot_plan_sha256'] or len(anchors)!=3 or len({x['task'] for x in anchors})!=3 or len({x['group_id'] for x in anchors})!=3 or any(x['split']!='DEV' for x in anchors):raise ValueError('anchor plan')
 seed='eadp-budget-ttft-20261105-001'
 def rank(*parts):return hashlib.sha256('|'.join(map(str,(seed,*parts))).encode()).hexdigest()
 expected=[]
 for phase,reps in (('warmup',1),('measure',3)):
  for rep in range(reps):
   for anchor in sorted(anchors,key=lambda x:rank('anchor',phase,rep,x['task'],x['group_id'])):
    for k in sorted((128,256,512,1024),key=lambda x:rank('budget',phase,rep,anchor['group_id'],x)):
     expected.append({'task':anchor['task'],'qid':str(anchor['qid']),'group_id':anchor['group_id'],'phase':phase,'replicate':rep,'budget':k,'order':len(expected)})
 if jobs!=expected or len(jobs)!=48 or sum(x['phase']=='warmup' for x in jobs)!=12:raise ValueError('independent schedule mismatch')
 ps=D/'monitor_preflight001/started.json';pt=D/'monitor_preflight001/terminal.json';start=supervised(D/'monitor_preflight001',D/'launch.py')
 if start['command'][-3:]!=['--run-id','001','--preflight-only'] or start['timeout_seconds']!=900 or start['grace_seconds']!=90:raise ValueError('preflight command')
 rp=D/'run_preflight001';launch=read(rp/'launch.json');ex=read(rp/'exit.json');w=read(rp/'worker_preflight.json');processor=read(rp/'processor_preflight.json');ident=read(rp/'container_identity.json')
 if launch['approval_bundle'] is not None or launch['runtime_plan_sha256']!=sha(runtime_path) or launch['worker_sha256']!=sha(D/'worker.py') or w.get('status')!='passed' or w.get('identity')!=accepted or w.get('jobs')!=48 or w.get('runtime_plan_sha256')!=sha(runtime_path) or w.get('processor_preflight_sha256')!=sha(rp/'processor_preflight.json') or set(w.get('generation_paths',[]))!=set(generation) or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or w.get('GT_opened') is not False or w.get('predictions_opened') is not True:raise ValueError('worker preflight')
 if processor.get('status')!='passed' or processor.get('same_arm_references')!=12 or processor.get('weight_payloads_hashed')!=4 or len(processor['anchors'])!=3 or not all(x['attention_mask_all_one'] for x in processor['anchors']):raise ValueError('processor preflight')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ex['container_identity']!=ident or ident['container_id'] not in ident['host_cgroup']:raise ValueError('container evidence')
 obj=json.loads(subprocess.check_output(['docker','inspect',ident['container_id']],text=True))[0];state=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];cmd=launch['command'];idx=cmd.index(ident['image'])
 if obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or state['Status']!='exited' or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] or cfg['Entrypoint']!=[cmd[idx-1]] or cfg['Cmd']!=cmd[idx+1:] or cfg['User']!='1000:1000' or host['NetworkMode']!='none' or host['Devices'] or host['DeviceRequests']:raise ValueError('actual container identity')
 env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
 if env.get('CUDA_VISIBLE_DEVICES')!='' or env.get('HF_HUB_OFFLINE')!='1':raise ValueError('container GPU/offline environment')
 mounts=[]
 for i,x in enumerate(cmd):
  if x=='--mount':
   parts=cmd[i+1].split(',');f=dict(v.split('=',1) for v in parts if '=' in v);mounts.append((f['src'],f['dst'],'readonly' not in parts))
 if sorted(mounts)!=sorted((x['Source'],x['Destination'],x['RW']) for x in obj['Mounts']):raise ValueError('container mounts')
 health=[json.loads(x) for x in (D/'monitor_preflight001/health.jsonl').read_text().splitlines() if x.strip()]
 if not health or any(x.get('time_namespace')!=launch['time_namespace'] or len(x.get('health_snapshot_window_monotonic_ns',[]))!=2 or x['health_snapshot_window_monotonic_ns'][0]>x['health_snapshot_window_monotonic_ns'][1] for x in health):raise ValueError('supervisor health clock')
 proof={'status':'passed','independent_review':True,'scope':'budget_ttft_v3_gpu_prelaunch','GT_opened':False,'scoring_performed':False,'predictions_opened':True,'source_sha256':sources,'errors':[],'jobs':48,'warmups':12,'measured':36,'limitations':['three DEV anchors with three repetitions per budget; exploratory, not population TTFT','first-token boundary includes image read/processor and synchronization; raw-output fsync occurs after first-token and affects completion boundary','health windows are timed; other unmeasured host interference may remain','the container identity file is visible before its write completes: an incomplete JSON read may cause an auditable pre-model failure; it cannot pass admission and must not be silently retried','existing E1/E4 DEV mixed records opened for input and same-arm identity; no labels opened']}
 for key,path in (('plan',runtime_path),('identity_review',I/'e4c_budget_identity001_review.json'),('worker_preflight',rp/'worker_preflight.json'),('processor_preflight',rp/'processor_preflight.json'),('preflight_launch',rp/'launch.json'),('preflight_exit',rp/'exit.json'),('preflight_started',ps),('preflight_terminal',pt),('reviewer',Path(__file__))):proof[key+'_sha256']=sha(path)
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
