"""Independent review of frozen four-budget diagnostic GPU prelaunch."""
import argparse,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_budget_execution';R=B/'e1_independent_review';sys.path.insert(0,str(D))
import gate
if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('foreign gate')
EXPECTED='e4e46e54cf176764c485742b53b878d429d18bb7236cd86c85955691db539df7'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();proof=R/'e4c_budget_execution_prelaunch_review.json'
 if a.out.resolve()!=proof:raise ValueError('wrong output')
 plan=gate.plan();planfile=D/'execution_plan001.json';sources=plan['source_sha256']
 if gate.sha(planfile)!=EXPECTED or len(sources)!=142 or plan['GT_opened'] is not False or plan['scoring_performed'] is not False or plan['timing_claims'] is not False:raise ValueError('plan identity/scope')
 if (plan['natural_pairs'],plan['forced_pairs'],plan['forced_tokens'])!=(12,4,33) or (plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['stale_seconds'],plan['grace_seconds'])!=(3600,3720,1200,90):raise ValueError('plan denominator or bounds')
 cs=D/'monitor_cpu002/started.json';ct=D/'monitor_cpu002/terminal.json';cpu=gate.terminal(cs,ct);cmd=gate.resolved_command(cpu)
 if str(D/'cpu_check.py') not in cmd or gate.sha(cs)!=plan['cpu_started_sha256'] or gate.sha(ct)!=plan['cpu_terminal_sha256']:raise ValueError('CPU supervision')
 log=(D/'monitor_cpu002/stderr.log').read_text()
 if 'Ran 18 tests' not in log or 'Ran 6 tests' not in log:raise ValueError('synthetic test suites incomplete')
 prior=gate.prior();rp=D/'run_preflight001';ps=D/'monitor_preflight001/started.json';pt=D/'monitor_preflight001/terminal.json';start=gate.terminal(ps,pt);cmd=gate.resolved_command(start)
 if str(D/'launch.py') not in cmd or '--preflight-only' not in cmd or start['timeout_seconds']!=900 or start['grace_seconds']!=90:raise ValueError('wrong preflight command')
 launch=gate.read(rp/'launch.json');ex=gate.read(rp/'exit.json');w=gate.read(rp/'worker_preflight.json');processor=gate.read(rp/'processor_preflight.json');identity=gate.read(rp/'container_identity.json')
 if launch.get('approval_bundle') is not None or launch.get('plan_sha256')!=EXPECTED or launch.get('worker_sha256')!=gate.sha(D/'worker.py'):raise ValueError('preflight launch binding')
 if w.get('status')!='passed' or w.get('plan_sha256')!=EXPECTED or w.get('prior_review_sha256')!=prior['prior_proof_sha256'] or w.get('processor_preflight_sha256')!=gate.sha(rp/'processor_preflight.json') or w.get('GT_opened') is not False or w.get('scoring_performed') is not False or w.get('predictions_opened') is not True or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False:raise ValueError('worker preflight')
 if processor.get('status')!='passed' or processor.get('same_arm_references')!=12 or processor.get('weight_payloads_hashed')!=4 or processor.get('force_processor_code_identities_resolved')!=2 or len(processor['anchors'])!=3 or not all(x['attention_mask_all_one'] for x in processor['anchors']):raise ValueError('processor/forced control preflight')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or identity!=ex.get('container_identity') or identity['container_id'] not in identity['host_cgroup']:raise ValueError('container evidence')
 stages=[json.loads(x)['event'] for x in (rp/'stages.jsonl').read_text().splitlines()]
 if stages!=['cpu_preflight_started','cpu_preflight_completed']:raise ValueError('preflight stages')
 obj=json.loads(subprocess.check_output(['docker','inspect',identity['container_id']],text=True))[0];state=obj['State'];host=obj['HostConfig'];cfg=obj['Config'];argv=launch['command'];idx=argv.index(identity['image'])
 if obj['Id']!=identity['container_id'] or obj['Image']!=identity['image'] or state['Status']!='exited' or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled']:raise ValueError('actual container image/exit')
 if host['NetworkMode']!='none' or host['Devices'] or host['DeviceRequests'] or host['PidMode']!='' or cfg['User']!='1000:1000' or cfg['Entrypoint']!=[argv[idx-1]] or cfg['Cmd']!=argv[idx+1:]:raise ValueError('actual container command/devices')
 env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
 if env.get('CUDA_VISIBLE_DEVICES')!='' or env.get('HF_HUB_OFFLINE')!='1':raise ValueError('actual environment')
 mounts=[]
 for i,x in enumerate(argv):
  if x=='--mount':
   parts=argv[i+1].split(',');fields=dict(v.split('=',1) for v in parts if '=' in v);mounts.append((fields['src'],fields['dst'],'readonly' not in parts))
 if sorted(mounts)!=sorted((x['Source'],x['Destination'],x['RW']) for x in obj['Mounts']):raise ValueError('actual mounts')
 result={'status':'passed','independent_review':True,'scope':'gpu_diagnostic_prelaunch','GT_opened':False,'scoring_performed':False,'predictions_opened':True,'prediction_scope':'accepted E1 and E4 DEV mixed records for input and same-arm identity','source_sha256':sources,'errors':[],'container_id':identity['container_id'],'limitations':['diagnostic identity only, no latency or quality conclusion','32 generations comprise 12 natural and 4 forced pairs; forced sequences require 33 tokens/32 subsequent decode steps','3600 second cap is an operational bound, not a worst-case 2048-token guarantee','GPU run requires root scheduling and subsequent formal and independent result acceptance']}
 for key,path in (('plan',planfile),('prior_review',R/'e4c_budget_pilot_v2_preflight_review.json'),('worker_preflight',rp/'worker_preflight.json'),('processor_preflight',rp/'processor_preflight.json'),('preflight_launch',rp/'launch.json'),('preflight_exit',rp/'exit.json'),('preflight_started',ps),('preflight_terminal',pt),('reviewer',Path(__file__))):result[key+'_sha256']=gate.sha(path)
 with proof.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
