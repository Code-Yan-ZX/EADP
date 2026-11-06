"""Independent source review allowing only v2 no-GPU worker preflight."""
import argparse,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_model_pilot_v2';sys.path.insert(0,str(D))
import gate
if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('foreign gate')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=D/'independent_preflight_review.json':raise ValueError('wrong output')
 plan=gate.verify_plan();sources=gate.source_hashes();cpu=D/'cpu_check001/verification.json';s=D/'monitor_cpu001/started.json';t=D/'monitor_cpu001/terminal.json';gate.supervised(s,t,D/'cpu_validation.py',cpu.parent);c=gate.read(cpu);prior=gate.prior()
 if c.get('status')!='passed' or c.get('test_returncode')!=0 or c.get('plan_sha256')!=gate.sha(D/'plan001.json') or c.get('source_sha256_before')!=sources or c.get('source_sha256_after')!=sources or c.get('prior_identity')!=prior or c.get('GT_opened') is not False or c.get('GPU_used') is not False or c.get('model_loaded') is not False:raise ValueError('CPU proof mismatch')
 if 'Ran 13 tests' not in (cpu.parent/'stderr.log').read_text() or plan.get('GT_opened') is not False:raise ValueError('tests/scope incomplete')
 proof={'status':'passed','independent_review':True,'preflight_only':True,'reviewer_sha256':gate.sha(__file__),'source_sha256':sources,'cpu_sha256':gate.sha(cpu),'cpu_started_sha256':gate.sha(s),'cpu_terminal_sha256':gate.sha(t),'prior_identity':prior,'GT_opened':False,'scoring_performed':False,'gpu_authorized':False,'errors':[],'limitations':['v2 is a post-hoc repeat after examining v1','processor metadata and selection are persisted after request; timing of hashing is outside the request allocator window','P3/DS relies on same-source accepted C-ON identity; full high-dimensional values are not archived','only no-GPU preflight permitted by this proof']}
 with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
