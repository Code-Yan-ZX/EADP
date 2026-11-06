"""Independent post-hoc review of three-anchor scatter Graph TTFT evidence."""
import argparse,hashlib,json,math,os,random,statistics,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e5/scatter_graph_ttft_v2';O=D/'run_smoke001/observations'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def term(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==R/'scatter_graph_ttft_v2_result_review.json'
formal=D/'smoke001_verification.json';f=read(formal);fs=term(D/'monitor_verify001',D/'verify.py',formal);rs=term(D/'monitor_smoke001',D/'launch.py');pre=read(D/'independent_prelaunch_review.json');term(Path(pre['review_terminal_path']).parent,Path(pre['reviewer_path']),D/'independent_prelaunch_review.json');plan=read(D/'plan001.json');vp=read(D/'verification_plan001.json');launch=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json')
assert sha(D/'verification_plan001.json')=='3d052f2723a5da7f398517d78458a3bb82336c4c84118668159240d3482b1c04' and vp['plan_sha256']==sha(D/'plan001.json') and vp['verifier_sha256']==f['verifier_sha256']==sha(D/'verify.py') and f['status']=='passed' and f['errors']==[] and f['GT_opened'] is False and f['scoring_performed'] is False
assert all(sha(x)==h for x,h in plan['sha256'].items()) and all(sha(x)==h for x,h in f['files_sha256'].items());assert pre['status']=='passed' and pre['independent_review'] and pre['plan_sha256']==sha(D/'plan001.json') and pre['reviewer_sha256']==sha(pre['reviewer_path']);assert rs['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['gpu_pids_after']==[]
obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];host=obj['HostConfig'];cfg=obj['Config'];cmd=launch['command'];assert obj['Id']==ex['container_identity']['container_id'] and not st['Running'] and st['Pid']==st['ExitCode']==0 and not st['OOMKilled'] and obj['Image']==ex['container_identity']['image'] and cfg['Cmd']==[str(D/'worker.py'),'--out',str(O)] and host['NetworkMode']=='none' and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]]
assert {x['PathOnHost'] for x in host['Devices']}=={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'}
mount=[]
for i,x in enumerate(cmd[:-1]):
 if x=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if x=='--env':assert cmd[i+1] in cfg['Env']
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind')
rng=random.Random(20261105);jobs=[]
for phase,n in [('warmup',1),('measured',5)]:
 for repeat in range(n):
  anchors=list(plan['anchors']);rng.shuffle(anchors)
  for anchor in anchors:
   methods=['original','graph'];rng.shuffle(methods)
   for method in methods:jobs.append({'phase':phase,'repeat':repeat,'task':anchor['task'],'qid':anchor['qid'],'method':method,'protocol':'fixed32'})
assert len(jobs)==36 and read(D/'preflight001.json')['schedule']==jobs
records=lines(O/'records.jsonl');raw=lines(O/'generated_raw.jsonl');gens=lines(O/'generated.jsonl');comparisons=lines(O/'comparisons.jsonl');progress=lines(O/'progress.jsonl');complete=read(O/'worker_complete.json');assert len(records)==len(raw)==len(gens)==36 and len(comparisons)==18 and len(progress)==73 and complete['jobs']==36 and complete['comparisons']==18
expected=[{'event':'worker_started'}]
for j in jobs:expected.extend([{'event':'job_started','job':j},{'event':'job_completed','job':j}])
assert progress==expected
pairs={};recomputed=[];times={};same_count=0
for job,r,rawrow,g in zip(jobs,records,raw,gens):
 assert r['job']==rawrow['job']==g['job']==job and r['status']=='completed' and r['result']==g['result'];z=r['result'];idx=r['ordered_indices'];assert len(idx)==256 and len(set(idx))==256 and all(type(v) is int and 0<=v<1024 for v in idx) and r['quota']==[256] and rawrow['output_ids']==[z['output_ids']] and len(z['output_ids'])==z['output_tokens']==32 and z['mode']=='C-ON' and z['protocol']=='fixed32' and z['min_new_tokens']==32
 b=z['boundaries'];assert b['clock_implementation']=='clock_gettime(CLOCK_MONOTONIC)' and b['first_criteria_length']==1 and b['criteria_calls']==32 and b['explicit_sync_boundaries']==['before_t0','first_token','generation_end'] and b['t0_before_image_read_ns']<b['first_token_selected_ns']<b['generation_complete_ns'];elapsed=(b['first_token_selected_ns']-b['t0_before_image_read_ns'])/1e6;assert math.isfinite(elapsed) and elapsed>0
 if job['phase']=='measured':times.setdefault((job['task'],job['qid'],job['method']),[]).append(elapsed)
 key=(job['phase'],job['repeat'],job['task'],job['qid'],job['protocol'])
 if key not in pairs:pairs[key]=r
 else:
  prev=pairs.pop(key);assert {prev['job']['method'],job['method']}=={'original','graph'};so=prev['ordered_indices']==idx;sq=prev['quota']==r['quota'];st=prev['result']['output_ids']==z['output_ids'];recomputed.append({'job':{k:job[k] for k in ('phase','repeat','task','qid','protocol')},'ordered_indices_equal':so,'quota_equal':sq,'output_ids_equal':st,'eligible_for_equal_output_timing':so and sq and st})
assert not pairs and recomputed==comparisons;unequal=sum(not x['eligible_for_equal_output_timing'] for x in comparisons);assert unequal==f['unequal_comparisons']==complete['unequal_comparisons'];assert f['anchors']==3 and f['jobs']==36 and f['comparisons']==18
for anchor in plan['anchors']:
 label=anchor['task']+':'+anchor['qid']
 for method in ('original','graph'):
  xs=times[(anchor['task'],anchor['qid'],method)];v=f['per_anchor'][label][method];assert len(xs)==5 and v=={'n':5,'median_ms':statistics.median(xs),'min_ms':min(xs),'max_ms':max(xs),'raw_ms':xs}
 if unequal:assert not f['aggregate_only_if_all_pairs_equal']
setup=read(O/'graph_setup.json');assert setup['setup_excluded_from_timing'] is True and setup['transform']['replacement_count']==1 and setup['transform']['source_written'] is False and setup['capture_wall_ms']>0
out={'status':'passed','independent_review':True,'anchors':3,'jobs':36,'comparisons':18,'unequal_comparisons':unequal,'formal_sha256':sha(formal),'formal_started_sha256':sha(D/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(D/'monitor_smoke001/terminal.json'),'plan_sha256':sha(D/'plan001.json'),'verification_plan_sha256':sha(D/'verification_plan001.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['formal verifier prepared after GPU completion and before inspecting values','three anchors and 5 warm fixed32 repeats, exploratory, not M1','background load not attributable to individual samples','full DeepStack/prefill arrays and post-job hook inventory not archived','graph setup excluded; generation end time includes fsync, not suitable for full-generation timing'],'errors':[]}
with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
print('passed')
