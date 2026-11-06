"""Freeze an independent no-GPU execution-chain preflight, not a GPU run."""
import argparse,json,os
from pathlib import Path
import source_inventory as inv
import schedule_contract,execution_upstream
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/pixel_wavefront_model_t020'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.resolve()!=D/'preflight_plan002.json':raise ValueError('unexpected plan')
 c=D/'cpu_check003/verification.json';s=json.loads((D/'monitor_cpu003/started.json').read_text());t=json.loads((D/'monitor_cpu003/terminal.json').read_text());proof=json.loads(c.read_text());sources=inv.hashes()
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or proof.get('status')!='passed' or proof.get('source_sha256_before')!=sources or proof.get('source_sha256_after')!=sources or proof.get('existing_DEV_predictions_opened') is not True:raise ValueError('CPU evidence')
 upstream=execution_upstream.verify();base=json.loads((P/'preflight_plan002.json').read_text());jobs=schedule_contract.schedule(base['images'])
 plan={'stage':'six-arm identity execution no-GPU import and container preflight only','threshold':.2,'jobs':jobs,'source_sha256':sources,'cpu_sha256':inv.sha(c),'cpu_started_sha256':inv.sha(D/'monitor_cpu003/started.json'),'cpu_terminal_sha256':inv.sha(D/'monitor_cpu003/terminal.json'),'upstream_review_sha256':upstream['review_sha256'],'budget_source_sha256':inv.sha(B/'initial_plan.json'),'GT_opened':False,'gpu_authorized':False,'existing_DEV_predictions_opened':True,'limitations':['preflight only; no model or GPU identity executed','preflight reviewer is candidate author and does not independently audit candidate','real six-arm controller, GPU plan, termination and independent admission are not yet approved']}
 with a.out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
