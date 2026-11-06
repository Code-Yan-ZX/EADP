"""Independent CPU-only review of the wavefront v3 no-GPU preflight scope."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_wavefront_t020_timing_final_v3';I=B/'e1_independent_review'
OUT=I/'pixel_wavefront_t020_timing_final_v3_preflight_review.json'
EXPECTED='f471215481d63322c3478cfaf1de4f32d1d99d1318b7edff67178c52d21a6274'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT.resolve() or OUT.exists():raise ValueError('unexpected or existing output')
 plan=D/'preflight_plan001.json'
 if sha(plan)!=EXPECTED:raise ValueError('unreviewed plan')
 frozen=read(plan);source=frozen['source_sha256']
 if len(source)!=429 or any(sha(path)!=value for path,value in source.items()):raise ValueError('source inventory or hash changed')
 gate_path=D/'preflight_gate.py';gate_sha=sha(gate_path)
 spec=importlib.util.spec_from_file_location('_independent_wavefront_runtime_gate',gate_path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 for name in ('preflight_entry.py','preflight_launch.py'):
  text=(D/name).read_text()
  if "GATE_SHA='"+gate_sha+"'" not in text or 'check_gate(D/' not in text:raise ValueError('gate import is not pinned')
 g.cpu();g.prior();g.terminal(D/'monitor_freeze001',D/'freeze_preflight.py')
 if g.sources()!=source:raise ValueError('transitive inventory mismatch')
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_locks','test_raw_prefix','test_preflight_gate','test_final_main'],cwd=D,capture_output=True,text=True,timeout=120)
 if run.returncode!=0 or 'Ran 15 tests' not in run.stderr:raise ValueError('independent CPU tests failed: '+run.stderr[-3000:])
 try:g.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU gate is not locked')
 if any(sha(path)!=value for path,value in source.items()):raise ValueError('source changed during review')
 proof={'status':'passed','independent_review':True,'preflight_only':True,'gpu_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'source_sha256':source,'plan_sha256':sha(plan),'cpu_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'prior_review_sha256':sha(I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json'),'reviewer_sha256':sha(__file__),'cpu_retest':run.stderr,'errors':[],'scope':'independent review allows only retained no-GPU import/preflight; no model, NVML sampling or GPU execution; full timing identity/admission remains unreviewed'}
 with OUT.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
