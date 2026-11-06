"""Independent final review of one-image memory pilot's real no-GPU preflight."""
import argparse,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_model_pilot';R=B/'e1_independent_review';sys.path.insert(0,str(D))
import gate
if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('foreign gate')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=D/'independent_final_review.json':raise ValueError('wrong output')
 plan=gate.verify_plan();sources=gate.source_hashes();cpu=D/'cpu_check009/verification.json';cs=D/'monitor_cpu009/started.json';ct=D/'monitor_cpu009/terminal.json'
 gate.supervised(cs,ct,D/'cpu_validation.py',cpu.parent);c=gate.read(cpu)
 if c.get('status')!='passed' or c.get('source_sha256_before')!=sources or c.get('source_sha256_after')!=sources or c.get('GT_opened') is not False or c.get('GPU_used') is not False or c.get('model_loaded') is not False:raise ValueError('CPU evidence changed')
 prior=gate.prior();approval=gate.review(preflight_only=True,host=True)
 if c.get('prior_identity')!=prior or approval['prior_identity']!=prior:raise ValueError('upstream identity changed')
 out=D/'run_preflight001';ps=D/'monitor_preflight001/started.json';pt=D/'monitor_preflight001/terminal.json';gate.supervised(ps,pt,D/'preflight_launch.py',out)
 start=gate.read(ps);launch=gate.read(out/'launch.json');ex=gate.read(out/'exit.json');result=gate.read(out/'worker_preflight.json')
 if start.get('timeout_seconds')!=1500 or start.get('grace_seconds')!=90 or launch.get('approval')!=approval:raise ValueError('supervision or approval mismatch')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False:raise ValueError('unclean preflight')
 if result.get('status')!='passed' or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('GT_opened') is not False or result.get('predictions_opened') is not True or result.get('weights_verified')!=4 or result.get('source_sha256')!=sources:raise ValueError('worker preflight incomplete')
 anchor=result.get('anchor',{});
 if (anchor.get('task'),anchor.get('qid'),anchor.get('group_id'))!=('DocVQA_VAL','63709','dd3c0ab5edc067ad04920cd66ea639ec85529b0ad6b6d25be7d423591bd59c80'):raise ValueError('wrong anchor')
 ident=gate.read(out/'container_identity.json');
 if ident!=ex['container_identity'] or ident['container_id'] not in ident['host_cgroup'] or ident['host_init']['pid']<=0:raise ValueError('container identity mismatch')
 obj=json.loads(subprocess.check_output(['docker','inspect',ident['container_id']],text=True))[0];state=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];cmd=launch['command']
 if obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or obj['Image']!=cmd[-5] or state['Status']!='exited' or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled']:raise ValueError('actual image or container state')
 if host['NetworkMode']!='none' or host['Devices'] or host['DeviceRequests'] or host['PidMode']!='' or cfg['User']!='1000:1000' or cfg['Entrypoint']!=[cmd[-6]] or cfg['Cmd']!=cmd[-4:]:raise ValueError('actual container configuration')
 env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
 if env.get('CUDA_VISIBLE_DEVICES')!='' or env.get('HF_HUB_OFFLINE')!='1':raise ValueError('GPU/offline environment')
 mounts=[]
 for i,x in enumerate(cmd):
  if x=='--mount':
   parts=cmd[i+1].split(',');fields=dict(v.split('=',1) for v in parts if '=' in v);mounts.append((fields['src'],fields['dst'],'readonly' not in parts))
 if sorted(mounts)!=sorted((x['Source'],x['Destination'],x['RW']) for x in obj['Mounts']):raise ValueError('actual mount mismatch')
 fields={'status':'passed','independent_review':True,'preflight_only':False,'reviewer_sha256':gate.sha(__file__),'source_sha256':sources,'cpu_sha256':gate.sha(cpu),'cpu_started_sha256':gate.sha(cs),'cpu_terminal_sha256':gate.sha(ct),'prior_identity':prior,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'prediction_scope':result['prediction_scope'],'errors':[], 'container_id':ident['container_id'],'limitations':['single-image pilot; sampled process/device memory and allocator peaks are distinct','0.5-second periodic NVML values can miss transients and have MiB granularity','real prior 16 MiB handshake is not a model memory measurement','1560 seconds is the sum of segment limits, not a separate watchdog; outer limit is 1680 seconds']}
 for key,path in (('preflight_launch',out/'launch.json'),('preflight_exit',out/'exit.json'),('preflight_result',out/'worker_preflight.json'),('preflight_started',ps),('preflight_terminal',pt)):fields[key+'_sha256']=gate.sha(path)
 with a.out.open('x') as f:json.dump(fields,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
