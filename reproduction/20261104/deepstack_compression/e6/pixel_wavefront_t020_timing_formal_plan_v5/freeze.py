"""Freeze formal acceptance after GPU plan and before GPU execution."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_final_v9';A=B/'e6/pixel_wavefront_t020_timing_accept_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 out=A/'verification_plan001.json';gpu=R/'gpu_plan001.json'
 if out.exists() or (R/'run_001').exists() or not gpu.exists():raise ValueError('formal plan timing')
 plan=read(gpu);cpu=read(A/'cpu_proof001.json');s=read(A/'monitor_cpu001/started.json');t=read(A/'monitor_cpu001/terminal.json')
 if cpu.get('status')!='passed' or cpu.get('GT_opened') is not False or cpu.get('GPU_used') is not False or cpu.get('predictions_opened') is not False or cpu['source_sha256_before']!=cpu['source_sha256_after'] or any(sha(k)!=v for k,v in cpu['source_sha256_after'].items()) or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (Path(s['cwd'])/s['command'][1]).resolve()!=A/'cpu_check.py':raise ValueError('formal CPU proof')
 value={'scope':'24 retained container TTFT formal acceptance before interpretation','gpu_plan_sha256':sha(gpu),'source_sha256':cpu['source_sha256_after'],'cpu_proof_sha256':sha(A/'cpu_proof001.json'),'cpu_started_sha256':sha(A/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(A/'monitor_cpu001/terminal.json'),'run_supervisor_command':plan['run_supervisor_command'],'run_supervisor_cwd':plan['run_supervisor_cwd'],'prediction_scope':'existing accepted DEV mixed records read only for identities','GT_opened':False,'GPU_used_by_verifier':False,'limitations':['all 24 jobs retained; three anchors post hoc','separate-process warmup is not in-process steady-state','on-device selector clone is included in both timings','unrecorded background windows unknown','endpoint weight stats cannot exclude transient replacement']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
