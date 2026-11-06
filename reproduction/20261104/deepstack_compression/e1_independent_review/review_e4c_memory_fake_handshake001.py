"""Independent check of fabricated sampler and real no-GPU container evidence."""
import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e4_c_memory_handshake';D=H/'run_001';I=R/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
s=mon(H/'monitor_run001');sm=mon(D/'sampler_monitor');plan=read(H/'launch_plan.json');pre=read(I/'e4c_memory_fake_handshake_prelaunch_review.json');launch=read(D/'launch.json');end=read(D/'exit.json');e=read(D/'expected.json');ready=read(D/'worker_ready.json');release=read(D/'release.json');stopped=read(D/'stopped.json');stored=read(D/'stopped_inspect.json');events=[json.loads(x) for x in (D/'samples.jsonl').read_text().splitlines() if x]
assert s['cwd']==str(R) and s['command']==plan['command'][plan['command'].index('--')+1:] and s['timeout_seconds']==360 and s['grace_seconds']==90
assert pre['status']=='passed' and pre['scope']=='no-GPU fake handshake only' and pre['gpu_authorized'] is False and all(sha(k)==v for k,v in pre['source_sha256'].items())
assert launch['approval']['review_sha256']==sha(I/'e4c_memory_fake_handshake_prelaunch_review.json') and launch['scope']=='synthetic no-GPU handshake only' and launch['gpu_authorized'] is False and end['status']=='passed' and end['gpu_authorized'] is False
cmd=launch['docker_create'];assert cmd[:2]==['docker','create'] and '--device' not in cmd and '--gpus' not in cmd;cid=e['container_id'];c=json.loads(subprocess.run(['docker','inspect','--type','container',cid],capture_output=True,text=True,check=True,timeout=30).stdout)[0];host=c['HostConfig'];cfg=c['Config'];state=c['State']
assert c['Name']=='/'+launch['container_name'] and c['Id']==cid and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False and host['NetworkMode']=='none' and host.get('Devices') in (None,[]) and host.get('DeviceRequests') in (None,[])
assert cfg['User']==cmd[cmd.index('--user')+1] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(c['Image'])+1:] and 'CUDA_VISIBLE_DEVICES=' in cfg['Env']
expected=[cmd[i+1] for i,v in enumerate(cmd) if v=='--mount'];actual={x['Destination']:x for x in c['Mounts']};assert len(expected)==len(actual)==3
for text in expected:
 parts=text.split(',');d=dict(x.split('=',1) for x in parts if '=' in x);m=actual[d['dst']];assert m['Type']=='bind' and os.path.realpath(m['Source'])==os.path.realpath(d['src']) and m['RW']==('readonly' not in parts)
assert ready['namespace_pid']==1 and ready['cuda_initialized'] is False and release['status']=='release' and all(ready[k]==release[k]==e[k] for k in ('container_id','nonce','mode'))
assert stopped['status']=='stopped' and all(stopped[k]==e[k] for k in ('container_id','nonce','host_pid','start_ticks','boot_id')) and stopped['inspect_sha256']==sha(D/'stopped_inspect.json') and len(stored)==1 and stored[0]['Id']==cid and stored[0]['State']['Running'] is False and stored[0]['State']['Pid']==0 and stored[0]['State']['ExitCode']==0
assert sm['command']==launch['fake_sampler_worker'] and events[0]['status']=='sampling' and events[-1]['type']=='summary';samples=events[1:-1];assert events[-1]['samples']==len(samples)>1 and events[-1]['target_observed'] is True
assert release['first_sample_ns']==samples[0]['query_end_ns'] and samples[0]['used_bytes'] is None and samples[-1]['query_start_ns']>=stopped['monotonic_ns'] and samples[-1]['used_bytes'] is None
for i,x in enumerate(samples):
 assert x['type']=='sample' and x['other_gpu_processes']==[] and all(x[k]==e[k] for k in ('host_pid','start_ticks','boot_id','gpu_uuid')) and x['query_start_ns']<=x['query_end_ns']==x['monotonic_ns'] and x['query_end_ns']-x['query_start_ns']<=e['max_sample_gap_ns']
 if i:assert x['query_start_ns']>=samples[i-1]['query_end_ns'] and x['query_end_ns']-samples[i-1]['query_start_ns']<=e['max_sample_gap_ns']
 assert x['used_bytes'] in (None,1048576)
assert any(x['used_bytes']==1048576 for x in samples) and end['lifecycle']['audit']['fabricated_sample_bytes']==1048576 and end['lifecycle']['audit']['samples']==len(samples)
files=['launch.json','exit.json','expected.json','worker_ready.json','release.json','stopped.json','stopped_inspect.json','samples.jsonl'];proof={'status':'passed','independent_review':True,'scope':'fake sampler, no GPU measurement','container_id':cid,'samples_verified':len(samples),'fabricated_sample_bytes':1048576,'files_sha256':{x:sha(D/x) for x in files},'started_sha256':sha(H/'monitor_run001/started.json'),'terminal_sha256':sha(H/'monitor_run001/terminal.json'),'sampler_started_sha256':sha(D/'sampler_monitor/started.json'),'sampler_terminal_sha256':sha(D/'sampler_monitor/terminal.json'),'reviewer_sha256':sha(__file__),'gpu_authorized':False,'GT_opened':False,'errors':[],'limitations':['The 1 MiB value is fabricated, not NVML or actual GPU memory.','This rehearsal does not authorize GPU sampling.']}
with (I/'e4c_memory_fake_handshake001_review.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
