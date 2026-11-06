"""Independent preflight review; no feature values, labels or scores parsed."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_adaptive_geometry_v2';OUT=I/'e4_adaptive_geometry_v2_prelaunch_review.json'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.resolve()!=OUT.resolve() or OUT.exists():raise ValueError('unplanned output')
 freeze=read(D/'extraction_freeze001.json');cpu=read(D/'cpu_proof001.json');source=freeze['source_sha256']
 if freeze.get('status')!='frozen_before_feature_values' or freeze.get('GT_opened') is not False or freeze.get('feature_values_opened') is not False or freeze.get('score_values_opened') is not False or cpu.get('status')!='passed' or cpu.get('tests')!=6 or cpu.get('source_sha256_before')!=source or cpu.get('source_sha256_after')!=source or freeze.get('cpu_proof_sha256')!=sha(D/'cpu_proof001.json') or any(sha(p)!=h for p,h in source.items()):raise ValueError('CPU/freeze source identity')
 gate=D/'admission.py';expected='ec47e61d03abd9fb0a4f582df791e08b02bd90d96c902ac8640d8ff2c37bdee4'
 if sha(gate)!=expected or source.get(str(gate))!=expected:raise ValueError('admission source')
 spec=importlib.util.spec_from_file_location('_independent_geometry_admission',gate);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mod.terminal(D/'monitor_cpu001',D/'cpu_check.py');mod.terminal(D/'monitor_freeze001',D/'freeze.py');prior=mod.prior()
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_features','test_admission'],cwd=D,capture_output=True,text=True,timeout=120)
 if run.returncode!=0 or 'Ran 6 tests' not in run.stderr:raise ValueError('CPU synthetic tests: '+run.stderr[-3000:])
 if any(sha(p)!=h for p,h in source.items()):raise ValueError('source changed')
 proof={'status':'passed','independent_review':True,'GT_opened':False,'feature_values_opened':False,'score_values_opened':False,'predictions_opened':True,'prediction_scope':'accepted E4/E1 files hashed only; no prediction or geometric value parsed',
        'extraction_freeze_sha256':sha(D/'extraction_freeze001.json'),'cpu_proof_sha256':sha(D/'cpu_proof001.json'),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'reviewer_sha256':sha(__file__),'accepted_source_bindings':prior,'source_sha256':source,'cpu_retest':run.stderr,'errors':[],
        'limitations':['Geometry uses accepted E4 K128 selected positions; E4 verifier binds processor identity to accepted E1 indirectly.','E4 panel evidence is summary-level; no new tensor reconstruction.','Posthoc batch routing study, not measured end-to-end compute or speed.']}
 with OUT.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
