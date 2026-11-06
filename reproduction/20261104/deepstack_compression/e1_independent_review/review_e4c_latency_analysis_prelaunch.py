"""Independent synthetic-only prelaunch audit, no real timing values decoded."""
import argparse,hashlib,importlib.metadata,importlib.util,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e4_c_latency_analysis';I=R/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
a=argparse.ArgumentParser();a.add_argument('--out',required=True);out=Path(a.parse_args().out).resolve();assert out==(I/'e4c_latency_analysis_prelaunch_review.json').resolve()
cpu=read(H/'cpu_check004/verification.json');s=read(H/'monitor_cpu004/started.json');t=read(H/'monitor_cpu004/terminal.json')
assert cpu['status']=='passed' and cpu['tests']==13 and cpu['full_timing_values_opened'] is False and cpu['GT_opened'] is False
assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and s['cwd']==str(R)
assert s['command'][1:]==['e4_c_latency_analysis/cpu_check.py','--run-id','004'] and sha(H/'cpu_check004/stderr.log')==cpu['stderr_sha256']
spec=importlib.util.spec_from_file_location('audit_e4c_latency_analysis_gate',H/'analysis_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
src=g.source_hashes();assert len(src)==10 and src==cpu['source_sha256_before']==cpu['source_sha256_after'] and importlib.metadata.version('numpy')==cpu['numpy_version']
assert sha(H/'analysis_plan001.json')==g.PLAN_SHA and sha(H/'analysis_plan001_supplement.json')==g.SUP_SHA and read(H/'analysis_plan001_supplement.json')['base_plan_sha256']==g.PLAN_SHA
p=subprocess.run([sys.executable,'-m','unittest','-v','test_analysis_core','test_analysis_gate','test_manual_overlap'],cwd=H,capture_output=True,text=True,timeout=120);assert p.returncode==0 and 'Ran 13 tests' in p.stderr and 'OK' in p.stderr
u=g.upstream();assert u['formal_sha256']==sha(R/'e4_c_latency_full/full_001_verification.json') and src==g.source_hashes()
proof={'status':'passed','independent_review':True,'scope':'latency_analysis_prelaunch','full_timing_values_opened':False,'GT_opened':False,'errors':[],'source_sha256':src,'plan_sha256':g.PLAN_SHA,'supplement_sha256':g.SUP_SHA,'cpu_proof_sha256':sha(H/'cpu_check004/verification.json'),'cpu_started_sha256':sha(H/'monitor_cpu004/started.json'),'cpu_terminal_sha256':sha(H/'monitor_cpu004/terminal.json'),'reviewer_sha256':sha(__file__),'synthetic_tests':13,'upstream_metadata_checked':True}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
