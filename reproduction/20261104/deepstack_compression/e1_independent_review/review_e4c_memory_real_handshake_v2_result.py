"""Independent post-run reconstruction of the bounded real NVML handshake."""
import argparse,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_real_handshake_v2';R=D/'run_001';I=B/'e1_independent_review'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(m,script):
 s=load(m/'started.json');t=load(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];assert [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()];return s
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==I/'e4c_memory_real_handshake_v2_result_review.json'
plan=load(D/'plan001.json');assert len(plan['source_sha256'])==96 and all(sha(k)==v for k,v in plan['source_sha256'].items());assert plan['allocation_bytes']==16777216 and plan['hold_seconds']==8
s=term(D/'monitor_run001',D/'controller.py');assert s['command'][-2:]==['--run-id','001'] and s['timeout_seconds']==300 and s['grace_seconds']==90
ss=term(R/'sampler_monitor',D/'sampler.py');assert ss['timeout_seconds']==150 and ss['command'][-2:]==['--directory',str(R)]
pre=I/'e4c_memory_real_handshake_v2_prelaunch_review.json';pr=load(pre);term(I/'monitor_e4c_memory_real_handshake_v2_prelaunch',I/'review_e4c_memory_real_handshake_v2_prelaunch.py');assert pr['status']=='passed' and pr['independent_review'] is True and pr['plan_sha256']==sha(D/'plan001.json')
l=load(R/'launch.json');e=load(R/'exit.json');exp=load(R/'expected.json');ready=load(R/'worker_ready.json');sr=load(R/'sampler_ready.json');release=load(R/'release.json');alloc=load(R/'allocation.json');done=load(R/'worker_complete.json');st=load(R/'stopped.json');assert e['status']=='passed' and e['failure'] is None and l['approval']['review_sha256']==sha(pre) and l['approval']['plan_sha256']==sha(D/'plan001.json')
assert ready['cuda_initialized'] is False and ready['namespace_pid']==1 and all(ready[k]==exp[k] for k in ('container_id','nonce','mode')) and all(release[k]==exp[k] for k in ('container_id','nonce','mode'))
running=load(R/'running_inspect.json')[0];stopped=load(R/'stopped_inspect.json')[0];actual=json.loads(subprocess.run(['docker','inspect',exp['container_id']],text=True,capture_output=True,check=True,timeout=20).stdout)[0]
for x in (running,stopped,actual):
 assert x['Id']==exp['container_id'] and x['Name']=='/eadp-memory-realhandshake-'+exp['nonce'] and x['Config']['Labels']['eadp.memory.nonce']==exp['nonce'] and x['Image']==x['Config']['Image']=='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'
 cfg=x['Config'];h=x['HostConfig'];assert cfg['User']=='1000:1000' and h['NetworkMode']=='none' and cfg['Cmd']==[str(D/'worker.py'),'--out',str(R),'--nonce',exp['nonce']]
 dev=h['Devices'];assert {v['PathOnHost'] for v in dev}=={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'} and all(v['PathOnHost']==v['PathInContainer'] and v['CgroupPermissions']=='rwm' for v in dev)
assert running['State']['Running'] and running['State']['Pid']==exp['host_pid']
for x in (stopped,actual):assert x['State']['Running'] is False and x['State']['Pid']==0 and x['State']['ExitCode']==0 and x['State']['OOMKilled'] is False
assert st['inspect_sha256']==sha(R/'stopped_inspect.json') and all(st[k]==exp[k] for k in ('container_id','host_pid','start_ticks','nonce','boot_id'))
rows=[json.loads(x) for x in (R/'samples.jsonl').read_text().splitlines()];samples=[x for x in rows if x.get('type')=='sample'];md=[json.loads(x) for x in (R/'device_metadata.jsonl').read_text().splitlines()];assert len(samples)==len(md)==16 and len(rows)==18 and rows[-1]['status']=='completed' and rows[-1]['samples']==16 and load(R/'sampler_complete.json')['sample_count']==16
assert sr['first_sample']==samples[0] and samples[0]['used_bytes'] is None and release['first_sample_ns']==samples[0]['query_end_ns'] and samples[0]['query_end_ns']<=alloc['allocation_monotonic_ns']<done['free_start_monotonic_ns']<=done['release_monotonic_ns']<=st['monotonic_ns']
assert alloc['allocation_bytes']==done['allocation_bytes']==16777216 and done['status']=='completed' and alloc['gpu_model']=='NVIDIA A40' and alloc['model_loaded'] is False and done['GT_opened'] is False
inside=[];used=[]
for i,(x,m) in enumerate(zip(samples,md)):
 assert all(x[k]==exp[k] for k in ('gpu_uuid','host_pid','start_ticks','boot_id')) and x['query_start_ns']<=m['query_start_ns']<=m['query_end_ns']<=x['query_end_ns']==x['monotonic_ns'] and x['query_end_ns']-x['query_start_ns']<=plan['max_sample_gap_ns'] and m['gpu_uuid']==exp['gpu_uuid'] and m['status']=='completed'
 if i:assert x['query_start_ns']>=samples[i-1]['query_end_ns'] and x['query_end_ns']-samples[i-1]['query_start_ns']<=plan['max_sample_gap_ns']
 target=[q for q in m['processes'] if q['pid']==exp['host_pid'] and q['type']=='C'];assert len(target)<=1 and x['used_bytes']==(target[0]['used_bytes'] if target else None);assert x['other_gpu_processes']==[]
 if target:used.append(x['used_bytes'])
 if target and x['query_start_ns']>=alloc['allocation_monotonic_ns'] and x['query_end_ns']<=done['free_start_monotonic_ns']:inside.append(x)
assert samples[-1]['query_start_ns']>=st['monotonic_ns'] and len(inside)==13 and len(used)==13 and max(used)==e['lifecycle']['audit']['nvml_process_sampled_peak_bytes']==339738624
for q in ('before_gpu_query.json','after_gpu_query.json'):
 v=load(R/q);assert v['query_error'] is None and v['compute_processes']=='' and exp['gpu_uuid'] in v['inventory']
assert load(R/'after_gpu_query.json')['query_start_ns']>=st['monotonic_ns']
files=[x for x in R.rglob('*') if x.is_file()];proof={'status':'passed','independent_review':True,'scope':'no-model real CUDA allocation and NVML sampling','GT_opened':False,'model_loaded':False,'scoring_performed':False,'gpu_uuid':exp['gpu_uuid'],'host_pid':exp['host_pid'],'container_id':exp['container_id'],'allocation_bytes':16777216,'sample_count':len(samples),'held_window_positive_samples':len(inside),'sampled_process_max_bytes':max(used),'plan_sha256':sha(D/'plan001.json'),'prelaunch_review_sha256':sha(pre),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'source_count':len(plan['source_sha256']),'evidence_sha256':{str(x):sha(x) for x in files},'limitations':['16 MiB is the requested tensor, sampled whole-process memory includes CUDA context and allocator','MiB-rounded periodic samples can miss transient peaks','graphics clients were present; device usage is not target-process usage','does not validate model-workload or latency measurements'],'errors':[]};a.out.write_text(json.dumps(proof,indent=2)+'\n');print('passed',len(samples),len(inside),max(used))
