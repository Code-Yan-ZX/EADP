"""Independent final PriorTR DEV32 prelaunch review, no GPU or labels."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(HERE))
import gate,source_inventory
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.out.resolve()==HERE/'independent_review_final.json'
 cpu=HERE/'cpu_check_006/verification.json';cs=HERE/'monitor_cpu006/started.json';ct=HERE/'monitor_cpu006/terminal.json'
 pre=HERE/'independent_review_preflight.json';prs=HERE/'monitor_review_preflight/started.json';prt=HERE/'monitor_review_preflight/terminal.json'
 gate.terminal(cs,ct,HERE/'cpu_checks.py',cpu.parent);gate.terminal(prs,prt,HERE/'review_preflight.py',pre)
 c=read(cpu);r=read(pre);observed={p:sha(p) for p in source_inventory.sources()}
 assert len(observed)==107 and c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after']==observed
 assert r['status']=='passed' and r['independent_review'] is True and r['GT_opened'] is False and r['scoring_performed'] is False
 assert r['source_sha256']==observed and r['reviewer_sha256']==sha(HERE/'review_preflight.py') and r['cpu_proof_sha256']==sha(cpu) and r['cpu_started_sha256']==sha(cs) and r['cpu_terminal_sha256']==sha(ct)
 prior=gate.execution_gate(preflight=True);assert prior['cpu_sha256']==sha(cpu) and prior['review_sha256']==sha(pre)
 pf=HERE/'container_preflight_003.json';ps=HERE/'monitor_preflight003/started.json';pt=HERE/'monitor_preflight003/terminal.json';out=HERE/'preflight_003'
 gate.terminal(ps,pt,HERE/'preflight.py');started,terminal=read(ps),read(pt);assert started['worker_pid']==terminal['worker_pid']==started['process_group']
 f=read(pf);gate.check_preflight(f,out);launch=read(out/'launch.json');stdout=json.loads((out/'stdout.log').read_text().strip().splitlines()[-1]);result=f['result']
 # Bind the actual launcher receipt and worker stdout to this review and same upstream.
 assert launch['approval']==stdout['approval']==result['approval']==prior
 assert stdout==result and result['upstream']==prior['upstream']==c['upstream']==gate.upstream()
 assert prior['cpu_sha256']==sha(cpu) and prior['review_sha256']==sha(pre) and result['samples']==32
 assert result['cuda_initialized'] is False and result['model_loaded'] is False and result['predictions_parsed'] is False and result['labels_opened'] is False
 cmd=launch['command'];ident=f['container_id'];obj=json.loads(subprocess.check_output(['docker','inspect',ident],text=True))[0];state=obj['State']
 plan=read(BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
 assert obj['Id']==ident and obj['Image']==plan['container_image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False
 assert not obj['HostConfig']['Devices'] and not obj['HostConfig']['DeviceRequests'] and obj['HostConfig']['NetworkMode']=='none'
 assert obj['Config']['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and obj['Config']['Cmd']==cmd[cmd.index(plan['container_image'])+1:]
 assert obj['Config']['User']==cmd[cmd.index('--user')+1] and obj['Config']['WorkingDir']==cmd[cmd.index('--workdir')+1]
 envs=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'];assert all(x in obj['Config']['Env'] for x in envs) and 'CUDA_VISIBLE_DEVICES=' in envs
 q={'status':'passed','independent_review':True,'scope':'gpu_diagnostic_prelaunch','GT_opened':False,'scoring_performed':False,'gpu_used':False,
    'reviewer_sha256':sha(__file__),'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),'source_sha256':observed,
    'preflight_sha256':sha(pf),'preflight_started_sha256':sha(ps),'preflight_terminal_sha256':sha(pt),
    'preflight_launch_sha256':sha(out/'launch.json'),'preflight_stdout_sha256':sha(out/'stdout.log'),'provisional_review_sha256':sha(pre),
    'limits':['No actual GPU or labels used by this review/preflight.','Cross-backend first logits unavailable for these 32 samples; compare only completed output IDs.',
              'Different SDPA/FA2 environments prevent a clean speed comparison; diagnostic generation must be scheduled by the coordinator.']}
 with a.out.open('x') as f:json.dump(q,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','scope':q['scope'],'sources':len(observed)}))
if __name__=='__main__':main()
