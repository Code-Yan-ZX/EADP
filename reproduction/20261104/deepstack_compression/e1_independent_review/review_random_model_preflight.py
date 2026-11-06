"""Independent CPU review: authorize only the random model no-GPU preflight."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/random_smoke';I=B/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def term(path,script):
 s=read(path/'started.json');t=read(path/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'] and (cwd/c[1]).resolve()==script.resolve()
def main(out):
 plan=read(D/'execution_plan009.json');cpu=read(D/'model_cpu_proof009.json');term(D/'monitor_model_cpu009',D/'freeze_model_execution.py')
 assert plan['scope']=='random K256 three-task 18-arm diagnostic' and plan['GPU_authorized'] is False and plan['GT_opened'] is False and plan['predictions_opened'] is True and cpu['status']=='passed' and cpu['test_returncode']==0 and cpu['GT_opened'] is False and cpu['predictions_opened'] is True and cpu['model_loaded'] is False and cpu['GPU_authorized'] is False
 sources=plan['source_sha256'];before=cpu['source_sha256_before'];assert len(sources)==258 and cpu['source_sha256_after']==before and dict(sources,**{})==dict(before,**{str((D/'model_cpu_proof009.json').resolve()):sha(D/'model_cpu_proof009.json')})
 for path,digest in sources.items():assert sha(path)==digest
 prior=read(I/'random_smoke_preflight002_result_review.json');assert prior['status']=='passed' and prior['independent_review'] is True and prior['GT_opened'] is False;term(I/'monitor_random_smoke_preflight002_result',I/'review_random_smoke_preflight002_result.py')
 gate_hash=sha(D/'smoke_gate.py');assert gate_hash in (D/'model_worker.py').read_text() and gate_hash in (D/'model_launch.py').read_text()
 assert plan['preflight_inner_seconds']==1800 and plan['preflight_outer_seconds']==2100 and plan['grace_seconds']==90 and plan['preflight_outer_seconds']>plan['preflight_inner_seconds']+plan['grace_seconds'] and plan['budget_source_sha256']==sha(B/'initial_plan.json')
 names=['test_smoke_core','test_identity_contract','test_inventory','test_diagnostic_run','test_model_core','test_e1_refs','test_manifest_load','test_smoke_gate','test_model_worker_integration','test_actual_import_chain'];r=subprocess.run([sys.executable,'-m','unittest','-v',*names],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=120);assert r.returncode==0 and 'Ran 29 tests' in r.stderr
 assert all(sha(path)==digest for path,digest in sources.items())
 proof={'status':'passed','independent_review':True,'scope':'noGPU preflight only','GT_opened':False,'scoring_performed':False,'predictions_opened':True,'prediction_scope':'source files opened for SHA; synthetic tests only; later actual preflight parses accepted E1 DEV mixed journal for identity','plan_sha256':sha(D/'execution_plan009.json'),'cpu_proof_sha256':sha(D/'model_cpu_proof009.json'),'cpu_started_sha256':sha(D/'monitor_model_cpu009/started.json'),'cpu_terminal_sha256':sha(D/'monitor_model_cpu009/terminal.json'),'source_sha256':sources,'reviewer_sha256':sha(__file__),'errors':[],'limitations':['No model or GPU is authorized by this proof.','Prior EADP selection is computed and discarded; diagnostics are not timing-comparable.','Actual container and GPU continuation require separate reviews.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'random_model_preflight_review.json' and not a.out.exists();main(a.out)
