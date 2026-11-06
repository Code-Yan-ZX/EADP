"""Independent review of the completed nine-image no-GPU container preflight."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_multistream_expansion352';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()],str(mon)
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();rp=D/'preflight001';mon=D/'monitor_preflight001';s,t=terminal(mon,D/'preflight_launch.py');terminal(D/'monitor_cpu_check001',D/'cpu_check.py');pre=I/'e4_multistream_expansion352_preflight_review.json';pres=I/'review_e4_multistream_expansion352_preflight_review.py';terminal(I/'monitor_e4_multistream_expansion352_preflight_review',pres,pre)
 cpu=read(D/'cpu_check001.json');q=read(pre);w=read(rp/'worker_preflight.json');launch=read(rp/'launch.json');ex=read(rp/'exit.json');ident=read(rp/'container_identity.json');selection=read(D/'selection001.json');source=cpu['source_sha256_after']
 assert cpu['status']=='passed' and cpu['source_sha256_before']==source and len(source)==116 and all(sha(k)==v for k,v in source.items()) and q['status']=='passed' and q['independent_review'] is True and q['preflight_only'] is True and q['source_sha256']==source and q['reviewer_sha256']==sha(pres) and q['cpu_sha256']==sha(D/'cpu_check001.json') and q['cpu_started_sha256']==sha(D/'monitor_cpu_check001/started.json') and q['cpu_terminal_sha256']==sha(D/'monitor_cpu_check001/terminal.json') and q['selection_sha256']==sha(D/'selection001.json') and q['errors']==[]
 assert w['status']=='passed' and (w['pilot_images'],w['full_inputs'],w['e4_reference_records'])==(9,384,9) and w['model_loaded'] is False and w['cuda_initialized'] is False and w['GT_opened'] is False and w['predictions_opened'] is True and w['scoring_performed'] is False and w['source_sha256']==source and w['selection_sha256']==sha(D/'selection001.json')
 assert launch['source_sha256']==source and launch['worker_sha256']==sha(D/'pilot_worker.py') and launch['launcher_sha256']==sha(D/'preflight_launch.py') and launch['gpu_requested'] is False and launch['controller']==ex['controller'] and ex['controller']['pid']==s['worker_pid'] and ident==ex['container_identity']
 assert ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['interrupted'] is None and ex['client_exited_while_container_running'] is False and ex['container_stopped_verified'] is True and ex['failure_type'] is None
 cid=ident['container_id'];basecid=read(B/'e4_c_multibudget_worker192_v6/run_preflight001/exit.json')['container_identity']['container_id'];doc,base=[json.loads(subprocess.check_output(['docker','inspect',x]))[0] for x in (cid,basecid)]
 assert doc['Id']==cid and not doc['State']['Running'] and doc['State']['Pid']==doc['State']['ExitCode']==0 and doc['Image']==base['Image']==ident['image'] and doc['HostConfig']['NetworkMode']=='none' and not doc['HostConfig']['Devices'] and not doc['HostConfig']['DeviceRequests']
 for k in ('Entrypoint','User','WorkingDir'):assert doc['Config'][k]==base['Config'][k]
 assert doc['Config']['Cmd']==[str(D/'pilot_worker.py'),'--out',str(rp),'--preflight-only'] and sorted(doc['Config']['Env'])==sorted(base['Config']['Env']) and len({x.split('=',1)[0] for x in doc['Config']['Env']})==len(doc['Config']['Env']) and sorted(json.dumps(x,sort_keys=True) for x in doc['Mounts'])==sorted(json.dumps(x,sort_keys=True) for x in base['Mounts']) and len(doc['Mounts'])==12 and cid in ident['host_cgroup'] and ident['host_argv'][-4:]==[str(D/'pilot_worker.py'),'--out',str(rp),'--preflight-only']
 weights=w['weight_stats_before'];assert len(weights)==4
 reference=read(B/'e4_c_multibudget_worker192_v6/run_preflight001/processor_preflight.json')['weight_stats'];assert weights==reference
 for path,v in weights.items():
  st=Path(path).stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==v
 pilot=selection['pilot'];keys={(x['task'],str(x['qid'])):x for x in pilot};assert len(pilot)==len(keys)==9
 e1=B/'e1/input_preparation_002/inputs.jsonl';completion=read(B/'e1/e1full001_completion.json');assert completion['status']=='passed' and completion['files']['inputs']['path']==str(e1) and completion['files']['inputs']['sha256']==sha(e1)
 matched={}
 for x in lines(e1):
  key=(x['task'],str(x['qid']))
  if key in keys:assert key not in matched and all(str(x[k])==str(keys[key][k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) and sha(x['image_path'])==x['image_sha256'];matched[key]=True
 assert len(matched)==9
 panel=B/'e4_panel_v2/panel_003/records.jsonl';formal=read(B/'e4_panel_v2/panel_003_verification.json');assert formal['status']=='passed' and formal['files_sha256'][str(panel)]==sha(panel);matched={}
 for x in lines(panel):
  key=(x['task'],str(x['qid']))
  if key not in keys or x.get('budget')!=256 or x.get('ds_on') is not True:continue
  assert key not in matched and all(str(x[k])==str(keys[key][k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) and x['selection_mode']=='author_greedy' and x['actual_injection_verified'] is True;matched[key]=True
 assert len(matched)==9
 out={'status':'passed','independent_review':True,'pilot_images':9,'full_inputs':384,'e4_reference_records':9,'GT_opened':False,'predictions_opened':True,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'cpu_sha256':sha(D/'cpu_check001.json'),'source_sha256':source,'preflight_review_sha256':sha(pre),'preflight_review_started_sha256':sha(I/'monitor_e4_multistream_expansion352_preflight_review/started.json'),'preflight_review_terminal_sha256':sha(I/'monitor_e4_multistream_expansion352_preflight_review/terminal.json'),'worker_preflight_sha256':sha(rp/'worker_preflight.json'),'launch_sha256':sha(rp/'launch.json'),'identity_sha256':sha(rp/'container_identity.json'),'exit_sha256':sha(rp/'exit.json'),'started_sha256':sha(mon/'started.json'),'terminal_sha256':sha(mon/'terminal.json'),'reviewer_sha256':sha(__file__),'scope':'actual no-GPU container and accepted nine-input identity only','limitations':['Existing accepted E1/E4 DEV mixed records parsed only to match identity; no GT, score, feature or new prediction.','Four weights rehashed by preflight; this review compares retained stats and does not rehash payloads.','Actual processor execution on nine inputs and model/GPU inference were not performed.','This proof does not authorize any future GPU run or altered source version.'],'errors':[]}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
