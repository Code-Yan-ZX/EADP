"""Independently re-read all warm192 raw evidence; no GT and no timing inference."""
import argparse,hashlib,importlib.util,json,os,struct,subprocess
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e6/pixel_wavefront_t020_warm192';E=B/'e6/pixel_wavefront_t020_warm192_extra_checks';I=B/'e1_independent_review';ROOT=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def done(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor '+str(mon))
 return s
def overlap(start,end,ns,windows):
 if windows is None:return {'status':'unknown','overlap_indices':[],'reason':'query coverage absent'}
 if not windows:return {'status':'unknown','overlap_indices':[],'reason':'no recorded queries'}
 if any(x['time_namespace']!=ns for x in windows):return {'status':'unknown','overlap_indices':[],'reason':'time namespace not proven identical'}
 if any(type(x['start_ns']) is not int or type(x['end_ns']) is not int or not 0<x['start_ns']<=x['end_ns'] for x in windows):raise ValueError('invalid observation window')
 return {'status':'observed','overlap_indices':[i for i,x in enumerate(windows) if x['start_ns']<=end and x['end_ns']>=start], 'reason':None,'background_activity_outside_queries':'unknown'}
def observation_windows():
 health=[]
 for p in (D/'monitor_run001/health.jsonl',D/'monitor_observer001/health.jsonl'):
  for x in lines(p):
   w=x['health_snapshot_window_monotonic_ns'];health.append({'start_ns':w[0],'end_ns':w[1],'time_namespace':x['time_namespace']})
 obs=lines(D/'observations001.jsonl');snap=[x for x in obs if x['event']=='snapshot'];complete=[x for x in obs if x['event']=='logging_complete'];end=[x for x in obs if x['event']=='observer_end']
 if len(end)!=1 or end[0]['reason']!='stop' or end[0]['samples']!=len(snap) or len(complete)!=len(snap):raise ValueError('observer termination')
 windows=[]
 for i,(x,y) in enumerate(zip(snap,complete)):
  if x['index']!=i or y['index']!=i or x['time_namespace']!=y['time_namespace']:raise ValueError('observer order')
  ns=x['time_namespace']
  for q in (x['ps'],x['docker']):windows.append({'start_ns':q['query_start_ns'],'end_ns':q['query_end_ns'],'time_namespace':ns})
  windows.append({'start_ns':x['docker']['query_end_ns'],'end_ns':y['first_fsync_end_ns'],'time_namespace':ns})
 manual=lines(D/'manual_health_windows001.jsonl');return health,windows,manual if manual else None
def docker_check(obj,exit,launch,expected):
 if expected.count('--rm')!=1:raise ValueError('fixed container command')
 expected.remove('--rm')
 if expected!=launch['command'] or exit['controller']!=launch['controller'] or exit['returncode']!=0 or exit['container_exit_code']!=0 or exit['container_stopped_verified'] is not True or exit['timed_out'] is not False or exit.get('failure_type') is not None or exit['gpu_pids_after']!=[]:raise ValueError('container receipt')
 state=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];image=exit['container_identity']['image'];cid=exit['container_identity']['container_id']
 if (obj['Id']!=cid or obj['Name']!='/'+launch['container_name'] or obj['Image']!=image or cfg['Image']!=image or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False or cfg['Entrypoint']!=[expected[expected.index('--entrypoint')+1]] or cfg['User']!=expected[expected.index('--user')+1] or cfg['WorkingDir']!=expected[expected.index('--workdir')+1] or cfg['Cmd']!=expected[expected.index(image)+1:] or host['NetworkMode']!=expected[expected.index('--network')+1] or host.get('PidMode') not in ('',None)):raise ValueError('actual Docker process')
 for env in (expected[i+1] for i,x in enumerate(expected[:-1]) if x=='--env'):
  if [x for x in cfg['Env'] if x.startswith(env.split('=',1)[0]+'=')]!=[env]:raise ValueError('Docker env')
 devices=[expected[i+1] for i,x in enumerate(expected[:-1]) if x=='--device'];actual=host.get('Devices') or []
 if host.get('DeviceRequests') not in (None,[]) or {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in actual}!={(x,x,'rwm') for x in devices} or len(actual)!=len(devices):raise ValueError('Docker device')
 mounts=[]
 for i,x in enumerate(expected[:-1]):
  if x=='--mount':
   parts=expected[i+1].split(',');v=dict(z.split('=',1) for z in parts if '=' in z);mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in parts))
 actual=[(str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind']
 if len(actual)!=len(obj['Mounts']) or sorted(actual)!=sorted(mounts):raise ValueError('Docker mounts')
 return cid
def request_check(identity,job,index,raw,record):
 if (record.get('status')!='completed' or any(str(record.get(k))!=str(identity.get(k)) for k in ('task','qid','group_id')) or record.get('arm')!=job['arm'] or record.get('processor')!=identity['processor'] or record.get('rope_state_reset_before_request') is not True or record.get('hook_cleanup_verified') is not True):raise ValueError('request identity')
 if record.get('recipe')!={'do_sample':False,'max_new_tokens':1,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}:raise ValueError('generation recipe')
 cache=record['cache_check']
 if cache.get('initial_kv_lengths')!=[0]*36 or cache.get('initial_cache_position')!=0 or cache.get('cache_position_check_synchronizes') is not True:raise ValueError('fresh KV')
 ids=record['selected_indices']
 if not isinstance(ids,list) or not ids or ids!=sorted(set(ids)) or len(ids)!=record.get('selected_count') or any(type(x) is not int or not 0<=x<1024 for x in ids):raise ValueError('selection')
 full=raw['full_sequence_ids'];n=record['input_length'];info=identity['processor']['input_ids'];event=raw['request']
 if not isinstance(full,list) or len(full)!=1 or len(full[0])!=n+1 or any(type(x) is not int for x in full[0]) or hashlib.sha256(struct.pack('<%dq'%n,*full[0][:n])).hexdigest()!=info['sha256'] or info['shape']!=[1,n] or info['dtype']!='torch.int64' or full[0][-1]!=record['first_token_id']:raise ValueError('prompt/token')
 phase='warmup' if index==0 else 'measure'
 if event.get('phase')!=phase or event.get('index')!=index or event.get('block')!=job['block'] or event.get('arm')!=job['arm'] or any(str(event.get(k))!=str(identity.get(k)) for k in ('task','qid','group_id')):raise ValueError('raw event')
 eos=record['eos_ids']
 if not eos or any(type(x) is not int for x in eos) or record['first_token_is_eos']!=(full[0][-1] in eos):raise ValueError('EOS')
 boundary=record['boundaries'];start=boundary['t0_before_image_read_ns'];first=boundary['first_token_selected_ns'];end=boundary['generation_complete_ns']
 if any(type(x) is not int for x in (start,first,end)) or not 0<start<first<=end or boundary['first_criteria_length']!=n+1 or boundary['explicit_sync_boundaries']!=['before_t0','first_token','generation_end']:raise ValueError('boundaries')
 return start,first,end,boundary['time_namespace']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 if a.out.resolve()!=(I/'pixel_t020_warm192_result_review.json').resolve():raise ValueError('output')
 formalpath=D/'verification001.json';done(D/'monitor_verify001',E/'formal_v3.py',formalpath);done(D/'monitor_snapshot001',D/'snapshot.py',D/'run_001_sources');started=done(D/'monitor_run001',E/'gpu_launch_v3.py');done(D/'monitor_observer001',D/'observer.py')
 f=read(formalpath);gpu=read(D/'gpu_plan001.json');source=gpu['source_sha256'];snapshot=read(D/'run_001_sources/manifest.json')
 if f.get('status')!='passed' or f.get('processes')!=8 or f.get('warmups')!=8 or f.get('measured')!=384 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(E/'formal_v3.py') or f.get('gpu_plan_sha256')!=sha(D/'gpu_plan001.json') or f.get('formal_plan_sha256')!=sha(D/'formal_plan001.json') or f.get('run_started_sha256')!=sha(D/'monitor_run001/started.json') or f.get('run_terminal_sha256')!=sha(D/'monitor_run001/terminal.json') or f.get('snapshot_manifest_sha256')!=sha(D/'run_001_sources/manifest.json'):raise ValueError('formal identity')
 if any(sha(k)!=v for k,v in source.items()) or set(snapshot['sources'])!=set(source) or any(v['sha256']!=source[k] or sha(v['copy'])!=v['sha256'] for k,v in snapshot['sources'].items()) or any(sha(k)!=v for k,v in f['files_sha256'].items()):raise ValueError('source/files')
 guard=read(D/'observer_guard001.json');needed=gpu['outer_timeout_seconds']+gpu['grace_seconds']
 if guard.get('GPU_queried') is not False or guard.get('required_remaining_seconds')!=needed or min(guard['deadline_lower_monotonic_ns'],guard['supervisor_deadline_lower_monotonic_ns'])<guard['checked_monotonic_ns']+needed*10**9:raise ValueError('observer lifecycle gate')
 health,observer,manual=observation_windows();author_path=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author=read(author_path)
 loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';sp=importlib.util.spec_from_file_location('_warm_result_author_loader',loaderpath);loader=importlib.util.module_from_spec(sp);sp.loader.exec_module(loader);runtime=loader.load_author(source)
 jobs=gpu['jobs'];formal_records=f['records'];expected_count=0;ids=set();counts={'warmup':0,'measure':0};records=[]
 for job in jobs:
  d=D/'run_001'/('job_%03d'%job['order']);launch=read(d/'launch.json');exit=read(d/'exit.json');cid=exit['container_identity']['container_id']
  cmd=runtime.container_command(author,[str(D/'worker.py'),'--job-index',str(job['order'])],True,launch['container_name'])
  obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0]
  if docker_check(obj,exit,launch,cmd) in ids:raise ValueError('reused container')
  ids.add(cid);events=lines(d/'journal.jsonl');complete=read(d/'worker_complete.json');model=read(d/'model_load.json');release=read(d/'release.json')
  if len(events)!=98 or complete.get('measured')!=48 or complete.get('warmup')!=1 or complete.get('post_source_and_weight_check') is not True or model.get('job')!=job or model.get('container_id')!=cid or release.get('host_init')!=exit['container_identity']['host_init'] or obj['Config']['Hostname']!=cid[:12]:raise ValueError('process completion')
  for index,identity in enumerate([job['warmup']]+job['measured']):
   first,last=events[index*2:index*2+2];phase='warmup' if index==0 else 'measure';path=d/('raw_%03d.json'%index)
   if first.get('event')!='started' or last.get('event')!='completed' or first.get('index')!=index or last.get('index')!=index or last.get('raw_path')!=str(path) or last.get('raw_sha256')!=sha(path):raise ValueError('request journal')
   record=last['record'];start,first_ns,end,ns=request_check(identity,job,index,read(path),record)
   if not model['load_end_monotonic_ns']<first['monotonic_ns']<=start<first_ns<=end<=last['monotonic_ns']:raise ValueError('time ordering')
   expected={'process_order':job['order'],'block':job['block'],'phase':phase,'record':record,'monitoring':{'supervisor_health':overlap(start,first_ns,ns,health),'observer':overlap(start,first_ns,ns,observer),'manual':overlap(start,first_ns,ns,manual)}}
   if formal_records[expected_count]!=expected:raise ValueError('formal record or window mismatch')
   records.append((record['task'],str(record['qid']),record['arm'],phase));counts[phase]+=1;expected_count+=1
 if expected_count!=392 or counts!={'warmup':8,'measure':384} or len(ids)!=8 or len(formal_records)!=392:raise ValueError('complete denominator')
 measured=[x for x in records if x[3]=='measure'];pairs={(t,q) for t,q,arm,phase in measured}
 if len(pairs)!=192 or any(sum(x[0]==t and x[1]==q and x[2]==arm for x in measured)!=1 for t,q in pairs for arm in ('author','wavefront')):raise ValueError('paired inventory')
 fields={'formal_sha256':formalpath,'formal_started_sha256':D/'monitor_verify001/started.json','formal_terminal_sha256':D/'monitor_verify001/terminal.json','run_started_sha256':D/'monitor_run001/started.json','run_terminal_sha256':D/'monitor_run001/terminal.json','snapshot_manifest_sha256':D/'run_001_sources/manifest.json','reviewer_sha256':Path(__file__)}
 out={'status':'passed','independent_review':True,'processes':8,'warmups':8,'measured':384,'pairs':192,'GT_opened':False,'scoring_performed':False,'errors':[],**{k:sha(v) for k,v in fields.items()},'files_sha256':f['files_sha256'],'limitations':['post-hoc 192 DEV one-token timing, one GPU','runtime selector/KV observations include synchronization','full visual/DS/KV tensors not saved; raw output/prompt and live checks reviewed','background outside recorded windows and second fsync unknown']}
 with a.out.open('x') as dest:json.dump(out,dest,indent=2);dest.write('\n');dest.flush();os.fsync(dest.fileno())
if __name__=='__main__':main()
