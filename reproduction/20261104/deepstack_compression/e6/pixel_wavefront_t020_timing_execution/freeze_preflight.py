"""Freeze CPU-only source/preflight admission plan, never authorize GPU."""
import argparse,json,os
from pathlib import Path
import gate
D=Path(__file__).resolve().parent;B=D.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=D/'preflight_plan001.json':raise ValueError('unplanned plan')
 gate.terminal(D/'monitor_cpu001',D/'cpu_check.py',D/'cpu_check001');proof=gate.read(D/'cpu_check001/verification.json');sources=gate.sources()
 if proof.get('status')!='passed' or proof.get('source_sha256_before')!=sources or proof.get('source_sha256_after')!=sources or proof.get('gpu_used') is not False:raise ValueError('CPU evidence mismatch')
 plan={'stage':'t020 natural timing no-GPU import preflight only','source_sha256':sources,'cpu_sha256':gate.sha(D/'cpu_check001/verification.json'),'cpu_started_sha256':gate.sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':gate.sha(D/'monitor_cpu001/terminal.json'),'budget_source_sha256':gate.sha(B/'initial_plan.json'),'GT_opened':False,'existing_DEV_predictions_opened':True,'gpu_authorized':False,'limitations':['No timing or sampled memory GPU implementation authorized.','Candidate preflight opens prior six DEV prediction identity records.','Underlying model and tensor cloning affect future timings; no numerical claims.']}
 with a.out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
