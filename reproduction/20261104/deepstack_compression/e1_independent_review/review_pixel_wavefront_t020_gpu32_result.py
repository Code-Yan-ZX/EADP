"""Independent reviewer of 32-image GPU records; original GPU4 worker was co-authored by reviewer."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import torch
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_wavefront_t020_gpu32';RUN=D/'run_smoke001';I=B/'e1_independent_review';sys.path.insert(0,str(D))
from worker import check_plan,imports
from gate import require_final
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def term(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cwd=Path(s['cwd']);c=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
 return s
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==I/'pixel_wavefront_t020_gpu32_result_review.json';torch.set_num_threads(1);assert not torch.cuda.is_initialized()
plan=check_plan();pre=require_final();term(I/'monitor_pixel_wavefront_t020_gpu32_prelaunch',I/'review_pixel_wavefront_t020_gpu32_prelaunch.py',D/'independent_prelaunch_review.json');s=term(D/'monitor_smoke001',D/'launch.py');assert s['timeout_seconds']==1920 and s['grace_seconds']==90 and s['stale_seconds']==600
l=read(RUN/'launch.json');e=read(RUN/'exit.json');assert l['plan_sha256']==sha(D/'plan001.json') and l['controller']==e['controller'] and l['controller']['pid']==s['worker_pid'] and e['returncode']==e['container_exit_code']==0 and e['container_stopped_verified'] is True and e['client_exited_while_container_running'] is False and not e['timed_out'] and e['interrupted'] is None and e['failure_type'] is None and e['gpu_pids_after']==[]
cid=e['container_identity']['container_id'];assert cid in e['container_identity']['host_cgroup'];obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0];cfg=obj['Config'];hc=obj['HostConfig'];st=obj['State'];cmd=l['command'];image=obj['Image'];assert obj['Id']==cid and obj['Name']=='/'+l['name'] and cfg['Image']==image==e['container_identity']['image'] and st['Running'] is False and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(image)+1:] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and hc['NetworkMode']==cmd[cmd.index('--network')+1]
assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in hc['Devices']}=={(cmd[j+1],cmd[j+1],'rwm') for j,v in enumerate(cmd[:-1]) if v=='--device'} and not hc.get('DeviceRequests')
for j,v in enumerate(cmd[:-1]):
 if v=='--env':
  k=cmd[j+1].split('=',1)[0];assert [z for z in cfg['Env'] if z.startswith(k+'=')]==[cmd[j+1]]
mount=[]
for j,v in enumerate(cmd[:-1]):
 if v=='--mount':
  parts=cmd[j+1].split(',');attrs=dict(z.split('=',1) for z in parts if '=' in z);mount.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
actual=[(str(Path(z['Source']).resolve()),z['Destination'],z['RW']) for z in obj['Mounts'] if z['Type']=='bind'];assert len(actual)==len(obj['Mounts'])==len(mount) and sorted(actual)==sorted(mount)
data=read(RUN/'observations/complete.json');ev=[json.loads(x) for x in (RUN/'observations/progress.jsonl').read_text().splitlines()];assert data['status']=='attempts_completed' and len(data['cases'])==len(plan['anchors'])==32 and len(ev)==192 and data['GT_opened'] is False and data['model_loaded'] is False
_,author,normalize,candidate=imports();out=[]
for i,(case,anchor) in enumerate(zip(data['cases'],plan['anchors'])):
 key=anchor['task']+'/'+anchor['qid'];events=ev[i*6:(i+1)*6];assert case['key']==key and case['status']=='completed' and case['ordered_equal'] is True and len(case['arms'])==2 and [x['event'] for x in events]==['case_started','arm_started','arm_completed','arm_started','arm_completed','case_recorded'] and all(x['key']==key for x in events)
 cpu=torch.load(anchor['capture_path'],map_location='cpu',weights_only=True)['actual']['processor'];tokens=normalize(cpu['pixel_values']).reshape(1024,-1);expected=candidate(tokens,32,32,.2).tolist();assert expected==anchor['expected_indices'];times={}
 for j,arm in enumerate(case['arms']):
  assert arm['arm']==(['author','wavefront'] if i%2==0 else ['wavefront','author'])[j] and arm['indices']==expected and arm['matches_cpu'] is True and type(arm['wall_ns_synchronized']) is int and arm['wall_ns_synchronized']>0 and type(arm['start_monotonic_ns']) is int and type(arm['end_monotonic_ns']) is int and arm['end_monotonic_ns']-arm['start_monotonic_ns']==arm['wall_ns_synchronized']
  assert events[1+2*j]['arm']==arm['arm'] and all(events[2+2*j][k]==arm[k] for k in ('arm','indices','matches_cpu','wall_ns_synchronized','start_monotonic_ns','end_monotonic_ns'));times[arm['arm']]=arm['wall_ns_synchronized']
 out.append({'task':anchor['task'],'qid':anchor['qid'],'group_id':anchor['group_id'],'K':len(expected),'author_wall_ns':times['author'],'wavefront_wall_ns':times['wavefront']})
assert len({x['group_id'] for x in out})==32 and not torch.cuda.is_initialized();files=[D/'host_before001.txt',D/'host_after001.txt',D/'plan001.json',D/'independent_prelaunch_review.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',RUN/'launch.json',RUN/'exit.json',RUN/'observations/complete.json',RUN/'observations/progress.jsonl',Path(__file__)]
value={'status':'passed','independent_review':True,'images':32,'ordered_equal':32,'GT_opened':False,'model_loaded':False,'scoring_performed':False,'reviewer_sha256':sha(__file__),'cases':out,'evidence_sha256':{str(x):sha(x) for x in files},'limitations':['Candidate and current worker were written by another agent; this reviewer previously contributed to the earlier four-image worker from which the execution code descends.','All 32 images had been examined on CPU; 32 CPU captures had been studied post hoc, and four appeared in earlier .45 GPU pilot; these are not independent new images.','GPU input intermediate tensors are not archived separately; full selected index sequences and original captured CPU inputs are retained.','One synchronized host-wall selector sample per arm and image with recorded start/end; not a full model TTFT.','Only before/after host snapshots; timing-window background and its effect are unknown.'],'errors':[]};a.out.write_text(json.dumps(value,indent=2)+'\n');print('passed',len(out),flush=True)
