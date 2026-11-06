"""Independent source review permitting only real no-GPU worker preflight."""
import argparse,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_model_pilot';sys.path.insert(0,str(D))
import gate
if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('foreign gate')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=D/'independent_preflight_review.json':raise ValueError('wrong review output')
 plan=gate.verify_plan();sources=gate.source_hashes();cpu=D/'cpu_check009/verification.json';started=D/'monitor_cpu009/started.json';terminal=D/'monitor_cpu009/terminal.json'
 gate.supervised(started,terminal,D/'cpu_validation.py',cpu.parent);c=gate.read(cpu)
 if c.get('status')!='passed' or c.get('test_returncode')!=0 or c.get('source_sha256_before')!=sources or c.get('source_sha256_after')!=sources or c.get('plan_sha256')!=gate.sha(D/'plan001.json') or c.get('GT_opened') is not False or c.get('GPU_used') is not False or c.get('model_loaded') is not False:raise ValueError('CPU evidence incomplete')
 if 'Ran 13 tests' not in (cpu.parent/'stderr.log').read_text():raise ValueError('CPU test count missing')
 if c.get('prior_identity')!=gate.prior():raise ValueError('prior proof changed')
 if len(sources)<149 or plan.get('GT_opened') is not False:raise ValueError('source inventory or scope incomplete')
 result={'status':'passed','independent_review':True,'preflight_only':True,'reviewer_sha256':gate.sha(__file__),'source_sha256':sources,'cpu_sha256':gate.sha(cpu),'cpu_started_sha256':gate.sha(started),'cpu_terminal_sha256':gate.sha(terminal),'prior_identity':gate.prior(),'GT_opened':False,'scoring_performed':False,'gpu_authorized':False,'errors':[], 'limitations':['only no-GPU worker preflight admitted; final GPU review and root scheduling remain required','preflight parses existing E1/E4 DEV prediction records for same-anchor input/identity','1560 seconds is a sum of stage limits, not an independent wall-clock watchdog; outer supervision is 1680 seconds','NVML samples are whole-process/device values, MiB-rounded, and can miss transient peaks']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('passed')
if __name__=='__main__':main()
