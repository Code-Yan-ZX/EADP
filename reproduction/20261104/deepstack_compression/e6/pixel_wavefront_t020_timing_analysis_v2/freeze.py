import json,os
from pathlib import Path
import gate
D=Path(__file__).resolve().parent
def main():
 if (gate.R/'run_001').exists() or (D/'analysis_plan001.json').exists():raise ValueError('must freeze before actual GPU run')
 cpu=gate.read(D/'cpu_proof001.json');start=gate.read(D/'monitor_cpu001/started.json');term=gate.read(D/'monitor_cpu001/terminal.json');actual=gate.sources()
 if cpu.get('status')!='passed' or cpu.get('timing_results_opened') is not False or cpu.get('GT_opened') is not False or cpu['source_sha256_before']!=actual or cpu['source_sha256_after']!=actual or term.get('reason')!='completed' or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[] or start.get('worker_pid')!=term.get('worker_pid') or (Path(start['cwd'])/start['command'][1]).resolve()!=D/'cpu_check.py':raise ValueError('CPU source proof')
 value={'scope':'three-anchor descriptive per-image paired same-repeat TTFT','source_sha256':actual,'gpu_plan_sha256':gate.sha(gate.R/'gpu_plan001.json'),'formal_plan_sha256':gate.sha(gate.A/'verification_plan001.json'),'warmups':6,'measured':18,'ratio':'wavefront TTFT divided by author TTFT per anchor and repeat','aggregation':'per-arm median of three measured independent processes per anchor; all raw rows retained','limits':['post-hoc 3 DEV anchors','separate process warmup does not warm later process','observed monitoring windows only; missing background is unknown','no population inference or confidence interval','t=.2 uses dynamic visual budget'],'GT_opened':False,'timing_results_opened':False}
 with (D/'analysis_plan001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
