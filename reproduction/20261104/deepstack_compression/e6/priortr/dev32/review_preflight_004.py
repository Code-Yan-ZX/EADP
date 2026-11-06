"""Independent PriorTR DEV32 source review, limited to CPU preflight."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(HERE))
import gate,source_inventory
from launch import configure

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.out.resolve()==HERE/'independent_review_preflight.json'
 cpu=HERE/'cpu_check_004/verification.json';started=HERE/'monitor_cpu004/started.json';terminal=HERE/'monitor_cpu004/terminal.json'
 s,t=read(started),read(terminal);cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']==s['process_group']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_checks.py']==[(HERE/'cpu_checks.py').resolve()]
 assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==cpu.parent
 c=read(cpu);assert c['status']=='passed' and c['returncode']==0 and c['timed_out'] is False and c['GT_opened'] is False and c['scoring_performed'] is False and c['gpu_used'] is False
 observed={p:sha(p) for p in source_inventory.sources()};assert len(observed)==107 and c['source_sha256_before']==c['source_sha256_after']==observed
 for stream in ('stdout','stderr'):assert c[stream+'_sha256']==sha(cpu.parent/(stream+'.log'))
 tests=(cpu.parent/'stderr.log').read_text();assert 'Ran 13 tests' in tests and tests.rstrip().endswith('OK')
 assert gate.upstream()==c['upstream']
 sample=read(HERE/'sample_001.json');assert len(sample['sample'])==32 and sample['physical_generations']==96
 assert len({(r['task'],str(r['qid'])) for r in sample['sample']})==32 and len({r['group_id'] for r in sample['sample']})==32
 amendment=read(HERE/'comparison_amendment_001.json');assert amendment['sample_unchanged_sha256']==sha(HERE/'sample_001.json') and not amendment['native_anchor_overlap'] and not amendment['GT_opened'] and not amendment['new_generations_performed']
 budget=read(HERE/'resource_plan.json');assert budget['budget_source_sha256']==sha(BASE/'initial_plan.json') and budget['inner_timeout_seconds']<budget['outer_timeout_seconds'] and budget['snapshot_seconds']==3600 and budget['gpu_exclusive'] is True
 # Compare the exact prospective Docker command assembled by the launcher to
 # the independent verifier's reconstruction from the earlier accepted run.
 planpath=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';plan=read(planpath)
 sys.path.insert(0,str(planpath.parent));spec=importlib.util.spec_from_file_location('_dev32_review_run_panel',planpath.parent/'run_panel.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 run=HERE/'run_REVIEW';name='eadp-priortr32-REVIEW-999';worker=HERE/'panel_worker.py';cache=run/'runtime_cache'
 actual=configure(module.container_command(plan,[str(worker),'--execute','--out',str(run)],True,name),cache);assert actual.count('--rm')==1;actual.remove('--rm')
 prep=HERE.parent/'preparation';old=read(prep/'smoke_001/launch.json')['command'];assert old.count('--name')==1;old[old.index('--name')+1]=name
 i=old.index(str(prep/'smoke_worker.py'));expected=configure(old[:i]+[str(worker),'--execute','--out',str(run)],cache);assert actual==expected
 result={'status':'passed','independent_review':True,'scope':'preflight_only','GT_opened':False,'scoring_performed':False,'gpu_used':False,
         'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(started),'cpu_terminal_sha256':sha(terminal),'reviewer_sha256':sha(__file__),
         'source_sha256':observed,'upstream':c['upstream'],'limits':['No full worker preflight or GPU run has occurred under this proof.',
         'PriorTR is pinned to SDPA and a different software environment than the E4 FA2 baseline.',
         'Cross-backend first logits are unavailable for these 32 samples; only completed output IDs can be compared.']}
 with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','scope':'preflight_only','sources':len(observed)}))
if __name__=='__main__':main()
