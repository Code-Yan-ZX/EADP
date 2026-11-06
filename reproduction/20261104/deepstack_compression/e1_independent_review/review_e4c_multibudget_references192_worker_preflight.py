"""Independent CPU-only admission review; no target run, GT, model or GPU."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e4_c_multibudget_references192';A=B/'e4_c_multibudget_references192_accept'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('incomplete supervisor '+str(d))
 return s
def main(out):
 term(D/'monitor_execution_cpu001',D/'freeze_execution.py');term(A/'monitor_freeze001',A/'freeze.py')
 p=read(D/'execution_plan001.json');cpu=read(D/'execution_cpu001.json');ap=read(A/'verification_plan001.json');src=p['source_sha256'];asrc=ap['source_sha256']
 if p.get('stage')!='222-input four-budget fixed32 reference generation' or p.get('jobs')!=888 or p.get('GPU_authorized_by_plan') is not False or p.get('GT_opened') is not False or p.get('predictions_opened') is not True or cpu.get('status')!='passed' or cpu.get('test_returncode')!=0 or cpu.get('source_sha256')!=src or cpu.get('plan_sha256')!=sha(D/'execution_plan001.json') or cpu.get('GT_opened') is not False or cpu.get('predictions_opened') is not True or ap.get('status')!='prepared' or ap.get('GPU_authorized') is not False:raise ValueError('frozen CPU and acceptance plan')
 if len(src)!=235 or any(sha(k)!=v for k,v in src.items()) or any(sha(k)!=v for k,v in asrc.items()):raise ValueError('source hash or inventory')
 # Refuse accidental inclusion of current random full output or any label.
 if any('/random_full_exec' in x or '/score_panel' in x or x.endswith(('.tsv','.xlsx')) for x in src):raise ValueError('source scope')
 spec=importlib.util.spec_from_file_location('_independent_refs192_gate',D/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 if gate.plan()!=p:raise ValueError('machine plan mismatch')
 prior=gate.prior();fixed=gate.admit_fixed_identity();gate.execution_cpu(p)
 logs=[]
 for cwd,args in ((D,[sys.executable,'-m','unittest','-v','test_core','test_execution','test_proposal']),(A,[sys.executable,'-m','unittest','-v','test_verify'])):
  test=subprocess.run(args,cwd=cwd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180)
  if test.returncode!=0:raise ValueError('CPU synthetic tests failed '+test.stderr[-1500:])
  logs.append(test.stderr)
 if 'Ran 16 tests' not in logs[0] or 'Ran 2 tests' not in logs[1] or any(sha(k)!=v for k,v in src.items()) or any(sha(k)!=v for k,v in asrc.items()):raise ValueError('test counts or source changed')
 result={'status':'passed','independent_review':True,'scope':'references192_cpu_preflight_only','errors':[],'GT_opened':False,'predictions_opened':True,'prediction_scope':'previously accepted DEV prediction-bearing sources opened only as bytes for SHA; no target generation, no score or labels','scoring_performed':False,'GPU_authorized':False,'plan_sha256':sha(D/'execution_plan001.json'),'execution_cpu_sha256':sha(D/'execution_cpu001.json'),'execution_cpu_started_sha256':sha(D/'monitor_execution_cpu001/started.json'),'execution_cpu_terminal_sha256':sha(D/'monitor_execution_cpu001/terminal.json'),'source_sha256':src,'acceptance_plan_sha256':sha(A/'verification_plan001.json'),'acceptance_freeze_started_sha256':sha(A/'monitor_freeze001/started.json'),'acceptance_freeze_terminal_sha256':sha(A/'monitor_freeze001/terminal.json'),'reviewer_sha256':sha(__file__),'prior_identity':prior,'fixed_identity':fixed,'tests':16,'acceptance_tests':2,'limitations':['Only CPU synthetic/model-free and frozen sources reviewed; real same-image CPU worker preflight not yet performed.','No model execution or new generation; GPU requires subsequent independent review and scheduling.','Full-panel prefill/DS tensor arrays and decode KV content will not be archived; prefill KV fingerprints and per-step length/position only.']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args()
 if args.out.resolve()!=I/'e4c_multibudget_references192_worker_preflight_review.json' or args.out.exists():raise ValueError('fixed output required')
 main(args.out)
