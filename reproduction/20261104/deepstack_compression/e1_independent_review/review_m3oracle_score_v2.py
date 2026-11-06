"""Independent no-GT review of corrected M3 scoring admission and source chain."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;V2=ROOT/'m3_oracle_analysis_v2';EXEC=ROOT/'m3_oracle_exec';FORMAL=ROOT/'m3_oracle_verify_v2'
sys.path.insert(0,str(V2))
import prelaunch_gate,score_gate_v2
sha=prelaunch_gate.sha

def read(path):return json.loads(Path(path).read_text())
def require(v,msg):
 if not v:raise ValueError(msg)

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 require(a.out.resolve()==(V2/'independent_review.json').resolve(),'wrong proof output')
 cpu=V2/'cpu_check_002/verification.json';cs=V2/'monitor_cpu002/started.json';ct=V2/'monitor_cpu002/terminal.json'
 prelaunch_gate.runtime_gate.supervised(cs,ct,V2/'cpu_validation.py',cpu.parent)
 c=read(cpu);sources={str(p.resolve()):sha(p) for p in prelaunch_gate.sources()}
 require(len(sources)==14 and c.get('status')=='passed' and c.get('returncode')==0 and c.get('source_sha256_before')==c.get('source_sha256_after')==sources and c.get('GT_opened') is False and c.get('predictions_parsed') is False,'CPU proof/source mismatch')
 require(c.get('stdout_sha256')==sha(cpu.parent/'stdout.log') and c.get('stderr_sha256')==sha(cpu.parent/'stderr.log'),'CPU logs changed')
 logs=(cpu.parent/'stderr.log').read_text();require('Ran 6 tests' in logs and logs.rstrip().endswith('OK'),'gate tests did not pass')
 # Confirm the unchanged original scoring plan has its own separately accepted
 # source/statistics review. This gate only hashes proofs and source files.
 sys.path.insert(0,str(ROOT/'m3_oracle_analysis'));import analysis_gate
 require(Path(analysis_gate.__file__).resolve()==(ROOT/'m3_oracle_analysis/analysis_gate.py').resolve(),'wrong original analysis gate')
 old=analysis_gate.validate()
 binding=read(FORMAL/'acceptance_binding_002.json')
 require(binding['python_version']==sys.version and binding['executable_sha256']==sha(sys.executable),'review interpreter differs from runtime')
 args=SimpleNamespace(run=EXEC/'runtime_001',proof=FORMAL/'runtime_001_verification_author.json',verify_started=FORMAL/'monitor_verify002/started.json',verify_terminal=FORMAL/'monitor_verify002/terminal.json',run_started=EXEC/'monitor_runtime001/started.json',run_terminal=EXEC/'monitor_runtime001/terminal.json',review=HERE/'m3oracle_panel_review.json',review_started=HERE/'monitor_m3oracle_panel001/started.json',review_terminal=HERE/'monitor_m3oracle_panel001/terminal.json')
 accepted=score_gate_v2.gate(args)
 require(accepted['n_questions']==128 and accepted['new_generations']==768 and accepted['scoring_performed'] is False,'actual accepted inventory')
 # Independently reject a substituted review, without reading any prediction.
 wrong=SimpleNamespace(**vars(args));wrong.review=HERE/'wrong'
 try:score_gate_v2.gate(wrong)
 except ValueError:pass
 else:raise ValueError('wrong review path accepted')
 require(all(sha(p)==h for p,h in sources.items()),'source changed during review')
 result={'status':'passed','independent_review':True,'GT_opened':False,'predictions_parsed':False,'scoring_performed':False,'source_sha256':sources,'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),'reviewer_sha256':sha(__file__),'original_analysis_approval':old,'scope':'source and complete accepted-proof binding only; no score or reference/prediction parsing'}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();__import__('os').fsync(f.fileno())
 print(json.dumps({'status':'passed','sources':len(sources),'questions':128,'new_generations':768}))
if __name__=='__main__':main()
