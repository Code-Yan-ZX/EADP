"""Independent review of the four-budget CPU/no-GPU preflight only."""
import argparse, hashlib, json, subprocess
from pathlib import Path

B=Path(__file__).resolve().parent.parent
D=B/'e4_c_budget_pilot_v2'
EXPECTED_PLAN='8815b32b0dab83922a64f8fcb67ba357dddffc341cb77424441cef3537205632'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
 return h.hexdigest()
def read(p): return json.loads(p.read_text())
def terminal(name):
 s=read(D/name/'started.json');t=read(D/name/'terminal.json')
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
 assert Path(s['cwd']).resolve()==B
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.out.resolve()==B/'e1_independent_review/e4c_budget_pilot_v2_preflight_review.json'
 plan=D/'preflight_plan001.json'; assert sha(plan)==EXPECTED_PLAN
 spec=read(plan); sources=spec['source_sha256']; assert len(sources)==120
 for name,digest in sources.items(): assert sha(name)==digest, name
 assert spec['gpu_approval'] is False and spec['budgets']==[128,256,512,1024] and spec['GT_opened'] is False
 cpu=terminal('monitor_cpu006'); assert cpu['command'][1:]==['-m','unittest','discover','-s','e4_c_budget_pilot_v2','-p','test_*.py','-v']
 assert 'Ran 18 tests' in (D/'monitor_cpu006/stderr.log').read_text()
 assert spec['cpu_started_sha256']==sha(D/'monitor_cpu006/started.json') and spec['cpu_terminal_sha256']==sha(D/'monitor_cpu006/terminal.json')
 start=terminal('monitor_preflight001'); assert start['command'][1:] == ['e4_c_budget_pilot_v2/preflight_launch.py'] and start['timeout_seconds']==900 and start['grace_seconds']==90
 result=read(D/'preflight001.json'); assert result['status']=='passed' and result['plan_sha256']==EXPECTED_PLAN and result['sources_checked']==120
 assert result['cuda_initialized'] is False and result['model_loaded'] is False and result['GT_opened'] is False and result['scoring_performed'] is False
 assert result['predictions_opened'] is True and result['same_arm_references']==12 and result['weight_payloads_hashed']==4 and result['force_processor_code_identities_resolved']==2
 assert len(result['anchors'])==3 and all(x['attention_mask_all_one'] is True for x in result['anchors'])
 launch=read(D/'run_preflight001/launch.json'); exit=read(D/'run_preflight001/exit.json'); identity=read(D/'run_preflight001/container_identity.json')
 assert launch['plan_sha256']==EXPECTED_PLAN and exit['returncode']==0 and exit['container_stopped_verified'] is True and exit['container_exit_code']==0 and exit['timed_out'] is False and exit['failure_type'] is None
 assert identity==exit['container_identity'] and identity['host_init']['pid']>0 and identity['container_id'] in identity['host_cgroup']
 docker=json.loads(subprocess.check_output(['docker','inspect',identity['container_id']],text=True))[0]
 assert docker['Id']==identity['container_id'] and docker['Image']==identity['image']==launch['command'][-4]
 state=docker['State'];assert state['Status']=='exited' and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled']
 host=docker['HostConfig'];cfg=docker['Config']; assert host['NetworkMode']=='none' and not host['Devices'] and not host['DeviceRequests'] and host['PidMode']==''
 assert cfg['User']=='1000:1000' and cfg['Entrypoint']==[launch['command'][-5]] and cfg['Cmd']==launch['command'][-3:]
 env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
 assert env['CUDA_VISIBLE_DEVICES']=='' and env['HF_HUB_OFFLINE']=='1' and env['HF_DATASETS_OFFLINE']=='1'
 command=launch['command']; mounts=[]
 for i,x in enumerate(command):
  if x=='--mount':
   parts=command[i+1].split(',');data=dict(v.split('=',1) for v in parts if '=' in v);mounts.append((data['src'],data['dst'],'readonly' not in parts))
 actual=[(x['Source'],x['Destination'],x['RW']) for x in docker['Mounts']]
 assert sorted(mounts)==sorted(actual)
 assert any('/models--Qwen--Qwen3-VL-8B-Instruct' in dst and not rw for _,dst,rw in actual)
 fields={'status':'passed','independent_review':True,'scope':'CPU and real no-GPU preflight only; GPU worker and launcher not admitted','plan_sha256':sha(plan),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'cpu_started_sha256':sha(D/'monitor_cpu006/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu006/terminal.json'),'reviewer_sha256':sha(__file__),'source_sha256':sources,'container_id':identity['container_id'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'prediction_scope':result['prediction_scope'],'gpu_authorized':False,'errors':[]}
 with a.out.open('x') as f: json.dump(fields,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__': main()
