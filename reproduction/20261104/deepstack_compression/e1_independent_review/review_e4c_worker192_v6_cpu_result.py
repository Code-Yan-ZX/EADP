import argparse,hashlib,json,os,subprocess
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_c_multibudget_worker192_v6';I=B/'e1_independent_review';REF=B/'e4_c_multibudget_references192'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a=a.parse_args();p=read(D/'cpu_plan001.json');cpu=read(D/'cpu_proof001.json');review=I/'e4c_worker192_v6_cpu_preflight_review.json';rs=I/'review_e4c_worker192_v6_cpu_preflight.py';rm=I/'monitor_e4c_worker192_v6_cpu_preflight';q=read(review)
 assert p['status']=='prepared_cpu_preflight' and p['inputs']==222 and p['jobs']==7920 and p['GT_opened'] is False and p['target_predictions_opened'] is False and p['GPU_authorized'] is False
 src=p['source_sha256'];assert len(src)==254 and all(sha(k)==v for k,v in src.items())
 terminal(D/'monitor_cpu001',D/'freeze_cpu.py');terminal(rm,rs,review)
 assert cpu['status']=='passed' and cpu['source_sha256']==src and cpu['plan_sha256']==sha(D/'cpu_plan001.json') and cpu['test_returncode']==0
 assert q['status']=='passed' and q['independent_review'] is True and q['scope']=='worker192_cpu_preflight_only' and q['source_sha256']==src and q['reviewer_sha256']==sha(rs) and q['plan_sha256']==sha(D/'cpu_plan001.json') and q['cpu_proof_sha256']==sha(D/'cpu_proof001.json') and q['cpu_started_sha256']==sha(D/'monitor_cpu001/started.json') and q['cpu_terminal_sha256']==sha(D/'monitor_cpu001/terminal.json') and q['GPU_authorized'] is False and q['errors']==[]
 rp=D/'run_preflight001';w=read(rp/'worker_preflight.json');pr=read(rp/'processor_preflight.json');old=read(REF/'run_preflight001/processor_preflight.json');launch=read(rp/'launch.json');ex=read(rp/'exit.json');s,t=terminal(D/'monitor_preflight001',D/'launch.py')
 assert w['status']=='passed' and (w['inputs'],w['jobs'],w['existing_natural_references'])==(222,7920,888) and w['fixed32_new_references_opened'] is False and w['GT_opened'] is False and w['scoring_performed'] is False and w['cuda_initialized'] is False and w['model_loaded'] is False and w['GPU_authorized'] is False and w['predictions_opened'] is True
 assert w['plan_sha256']==sha(D/'cpu_plan001.json') and w['processor_preflight_sha256']==sha(rp/'processor_preflight.json') and pr['status']=='passed' and (pr['inputs_checked'],pr['references_checked'],pr['weight_payloads_hashed'])==(222,888,4) and pr['processor_inputs']==old['processor_inputs'] and pr['weight_stats']==old['weight_stats'] and pr['proposal_sha256']==sha(B/'e4_c_multibudget_expansion192/proposal002.json')
 for path,values in pr['weight_stats'].items():
  st=Path(path).stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==values
 assert ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['interrupted'] is None and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and ex['controller']['pid']==s['worker_pid']
 cid=ex['container_identity']['container_id'];rid=read(REF/'run_preflight001/exit.json')['container_identity']['container_id'];d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];r=json.loads(subprocess.check_output(['docker','inspect',rid]))[0]
 assert d['Id']==cid and d['State']['Running'] is False and d['State']['Pid']==0 and d['State']['ExitCode']==0 and d['Image']==r['Image']==ex['container_identity']['image']
 for k in ('Entrypoint','User','WorkingDir'):assert d['Config'][k]==r['Config'][k]
 assert d['Config']['Cmd']==[str(D/'worker.py'),'--out',str(rp),'--preflight-only'] and sorted(d['Config']['Env'])==sorted(r['Config']['Env']) and len({x.split('=',1)[0] for x in d['Config']['Env']})==len(d['Config']['Env'])
 for k in ('NetworkMode','PidMode','Devices','DeviceRequests','Binds','CgroupnsMode'):assert d['HostConfig'][k]==r['HostConfig'][k]
 assert d['HostConfig']['NetworkMode']=='none' and not d['HostConfig']['Devices'] and not d['HostConfig']['DeviceRequests'] and sorted(json.dumps(x,sort_keys=True) for x in d['Mounts'])==sorted(json.dumps(x,sort_keys=True) for x in r['Mounts'])
 assert ex['container_identity']['host_argv'][-4:]==[str(D/'worker.py'),'--out',str(rp),'--preflight-only'] and cid in ex['container_identity']['host_cgroup'] and launch['cpu_plan_sha256']==sha(D/'cpu_plan001.json')
 out={'status':'passed','independent_review':True,'scope':'worker192_cpu_preflight_result','GT_opened':False,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'existing_DEV_predictions_opened':True,'worker_preflight_sha256':sha(rp/'worker_preflight.json'),'processor_preflight_sha256':sha(rp/'processor_preflight.json'),'launch_sha256':sha(rp/'launch.json'),'exit_sha256':sha(rp/'exit.json'),'started_sha256':sha(D/'monitor_preflight001/started.json'),'terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'reviewer_sha256':sha(__file__),'cpu_plan_sha256':sha(D/'cpu_plan001.json'),'cpu_prelaunch_review_sha256':sha(review),'source_sha256':src,'inputs_verified':222,'existing_natural_references_checked':888,'fixed32_new_references_opened':False,'container_id':cid,'limitations':['Only no-GPU processor and old natural-reference identity validated; 888 fixed32 proof remains a separate GPU admission gate.','Payloads were hashed in actual preflight; this review checks their recorded stats and does not rehash payloads.','Existing accepted DEV prediction records parsed only for identity; no GT or new timings.'],'errors':[]}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
