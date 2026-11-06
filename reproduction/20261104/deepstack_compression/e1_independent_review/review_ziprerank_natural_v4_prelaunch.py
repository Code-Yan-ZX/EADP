"""Independent CPU-only review of three-anchor natural ZipRerank prelaunch."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/ziprerank_natural_v4';I=B/'e1_independent_review'
M=I/'monitor_ziprerank_natural_v4_prelaunch';TEST=I/'monitor_ziprerank_natural_v4_cpu_review'
ROOT=B.parents[2];AUTHOR_PLAN=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
EXPECTED_PLAN='b87c850ce9636e3e63d5e34c9a8c188f9031aa76929e1a69bfca290d09e80758'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json')
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]
     or t.get('worker_pid')!=s.get('worker_pid') or s.get('process_group')!=s.get('worker_pid')):raise ValueError('supervisor failed: '+str(d))
 return s
def wait_started():
 until=time.monotonic()+5
 while True:
  try:return get(M/'started.json')
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=until:raise ValueError('review supervisor start absent')
   time.sleep(.05)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=D/'independent_prelaunch_review.json':raise ValueError('unexpected proof path')
 plan=get(D/'plan001.json')
 if sha(D/'plan001.json')!=EXPECTED_PLAN or len(plan['sha256'])!=71:raise ValueError('plan identity')
 for path,digest in plan['sha256'].items():
  if sha(path)!=digest:raise ValueError('source drift: '+path)
 expected_generation={'do_sample':False,'use_cache':True,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0,'return_dict_in_generate':True}
 if plan['generation']!=expected_generation or plan['arms']!=['disabled','keep_all','author_prefix_256','user_text_256']:raise ValueError('natural recipe/arms')
 rows=[json.loads(line) for line in (B/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
 old={json.loads(line)['group_id'] for line in (B/'data_manifest/anchor_exclusion/anchor_groups.jsonl').read_text().splitlines()}
 if len(rows)!=384 or len(old)!=16 or len(plan['anchors'])!=3 or len({x['group_id'] for x in plan['anchors']})!=3:raise ValueError('DEV anchor inventory')
 for anchor in plan['anchors']:
  matches=[x for x in rows if x['task']==anchor['task'] and str(x['qid'])==anchor['qid']]
  if (len(matches)!=1 or matches[0]['group_id']!=anchor['group_id'] or anchor['group_id'] in old
      or matches[0]['message_sha256']!=anchor['message_sha256'] or sha(matches[0]['image_path'])!=anchor['image_sha256']):raise ValueError('anchor identity')
 test=term(TEST);original_test=term(D/'monitor_test001')
 for source in (TEST/'stderr.log',D/'monitor_test001/stderr.log'):
  if 'Ran 7 tests' not in source.read_text() or '\nOK' not in source.read_text():raise ValueError('CPU tests incomplete')
 for started in (test,original_test):
  if (Path(started['cwd'])/started['command'][-1]).resolve()!=D/'test_worker.py':raise ValueError('test worker identity')
 s=term(D/'monitor_preflight001');pre=get(D/'preflight001.json');launch=get(D/'run_preflight001/launch.json');ex=get(D/'run_preflight001/exit.json')
 if (pre.get('status')!='passed' or pre.get('source_plan_sha256')!=sha(D/'plan001.json') or pre.get('cuda_initialized') is not False
     or pre.get('model_loaded') is not False or pre.get('GT_opened') is not False or pre.get('weight_sha256_verified')!=4
     or pre.get('transformers')!='4.57.1' or len(pre.get('weight_stats',[]))!=4):raise ValueError('worker preflight')
 for item in pre['weight_stats']:
  st=Path(item['path']).stat()
  if (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino,st.st_dev)!=(item['size'],item['mtime_ns'],item['ctime_ns'],item['inode'],item['device']):raise ValueError('weight stat changed')
 if ((Path(s['cwd'])/s['command'][-3]).resolve()!=D/'launch.py' or s['command'][-2:]!=['--stage','preflight']
     or launch.get('plan_sha256')!=sha(D/'plan001.json') or launch.get('stage')!='preflight'
     or s.get('worker_pid')!=launch.get('controller',{}).get('pid') or ex.get('controller')!=launch.get('controller')
     or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True
     or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure_type') is not None
     or ex.get('client_exited_while_container_running') is not False):raise ValueError('preflight launch/exit')
 identity=ex['container_identity'];raw=subprocess.run(['docker','inspect','--type','container',identity['container_id']],capture_output=True,text=True,timeout=15,check=True)
 values=json.loads(raw.stdout)
 if len(values)!=1:raise ValueError('container inspect ambiguous')
 obj=values[0];state=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];cmd=launch['command'];image=get(AUTHOR_PLAN)['container_image']
 if (obj['Id']!=identity['container_id'] or obj['Id']!=ex['last_inspect']['id'] or obj['Image']!=image or identity['image']!=image
     or obj['Name']!='/'+launch['container_name'] or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False
     or host['NetworkMode']!='none' or host['PidMode'] not in ('',None) or host['Devices'] not in ([],None) or host['DeviceRequests'] not in ([],None)):
  raise ValueError('container image/state/devices')
 if (cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['User']!=cmd[cmd.index('--user')+1]
     or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1]
     or cfg['Cmd']!=[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')]
     or '--device' in cmd or '--gpus' in cmd or '--rm' in cmd):raise ValueError('container command')
 for env in [cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env']:
  key=env.split('=',1)[0]+'='
  if [x for x in cfg['Env'] if x.startswith(key)]!=[env]:raise ValueError('container env')
 if [x for x in cfg['Env'] if x.startswith('CUDA_VISIBLE_DEVICES=')]!=['CUDA_VISIBLE_DEVICES=']:raise ValueError('CUDA visibility')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x!='--mount':continue
  parts=cmd[i+1].split(',');v=dict(q.split('=',1) for q in parts if '=' in q)
  if v.get('type')!='bind':raise ValueError('unexpected mount type')
  mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in parts))
 actual=[(str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind']
 if len(actual)!=len(obj['Mounts']) or sorted(actual)!=sorted(mounts):raise ValueError('container mount profile')
 hostpid=identity['host_init'];proc=Path('/proc')/str(hostpid['pid'])/'stat'
 if Path('/proc/sys/kernel/random/boot_id').read_text().strip()==hostpid['boot_id'] and proc.exists():
  if int(proc.read_text().rsplit(')',1)[1].split()[19])==hostpid['start_ticks']:raise ValueError('original container process remains alive')
 started=wait_started();result={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),
  'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),
  'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),
  'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(M/'started.json'),
  'review_terminal_path':str(M/'terminal.json'),'GT_opened':False,'scoring_performed':False,'gpu_authorized':'subject to root scheduling',
  'cpu_tests':7,'source_files':71,'scope':'three DEV anchors, natural generation diagnostic; no model/GPU execution yet',
  'limitations':['enabled arms use local RoPE-state repair; user-text selection includes instructions/options',
    'per-step cache count is global or first layer; final cache records per-layer shapes without contents',
    'no quality, accuracy, throughput or equal-compute comparison','preflight rehashed four weights; this review verifies recorded stat, not payload again'],
  'errors':[]}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
