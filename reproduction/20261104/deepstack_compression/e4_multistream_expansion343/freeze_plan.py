"""Freeze 343 extension protocol before new features or scores are read."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';P=B/'e4_multistream_expansion352';V=B/'e4_multistream_expansion352_launch_v4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 out=D/'plan_freeze001.json'
 if out.exists():raise ValueError('exists')
 proof=I/'e4_multistream_expansion352_pilot_v4_result_review.json';reviewer=I/'review_e4_multistream_expansion352_pilot_v4_result.py';s=read(I/'monitor_e4_multistream_expansion352_pilot_v4_result/started.json');t=read(I/'monitor_e4_multistream_expansion352_pilot_v4_result/terminal.json');q=read(proof);selection=read(P/'selection001.json')
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('images')!=9 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('errors')!=[] or q.get('reviewer_sha256')!=sha(reviewer) or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (Path(s['cwd'])/s['command'][1]).resolve()!=reviewer or len(selection.get('remaining',[]))!=352 or len(selection.get('pilot',[]))!=9:raise ValueError('pilot acceptance')
 paths=[D/'PLAN.md',D/'freeze_plan.py',P/'selection001.json',B/'e4_multistream_budget_diagnostic_v2/PLAN.md',proof,reviewer,I/'monitor_e4_multistream_expansion352_pilot_v4_result/started.json',I/'monitor_e4_multistream_expansion352_pilot_v4_result/terminal.json',V/'verification001.json']
 obj={'status':'frozen','source_sha256':{str(p.resolve()):sha(p) for p in paths},'scope':'posthoc 343 extraction protocol freeze before inspecting any new features or scores','GT_opened':False,'score_values_opened':False,'new_feature_values_opened':False,'GPU_used':False,'errors':[]}
 with out.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
