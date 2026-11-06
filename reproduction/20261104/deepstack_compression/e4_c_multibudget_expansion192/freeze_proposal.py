"""Bind completed E1 metadata and freeze no-GT proposal; no model/GPU entry."""
import hashlib,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review'
import contract
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);c=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor exit or command')
 return s,t
def main():
 dest=D/'proposal002.json'
 if dest.exists():raise ValueError('already exists')
 cpu=read(D/'cpu_proof002.json');terminal(D/'monitor_cpu002',D/'cpu_check.py')
 if cpu.get('status')!='passed' or cpu.get('GT_opened') is not False or cpu.get('predictions_opened') is not False or cpu.get('sample_identities_opened') is not False or cpu.get('GPU_authorized') is not False or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after') or any(sha(p)!=h for p,h in cpu['source_sha256_after'].items()):raise ValueError('CPU source proof')
 e1=B/'e1';proofpath=e1/'e1full001_completion.json';reviewpath=I/'e1full001_review.json';fm=e1/'monitorverifye1full001';rm=I/'monitor_e1full001'
 terminal(fm,B/'e1_analysis/completion.py',proofpath);terminal(rm,I/'review_full.py',reviewpath)
 p=read(proofpath);q=read(reviewpath)
 if p.get('status')!='passed' or p.get('natural_questions')!=384 or p.get('natural_records')!=1536 or p.get('GT_opened') is not False or p.get('scoring_performed') is not False or p.get('verifier_sha256')!=sha(B/'e1_analysis/completion.py') or q.get('status')!='passed' or q.get('n_verified')!=384 or q.get('errors')!=[] or q.get('completion_proof_sha256')!=sha(proofpath) or q.get('completion_started_sha256')!=sha(fm/'started.json') or q.get('completion_terminal_sha256')!=sha(fm/'terminal.json') or q.get('reviewer_sha256')!=sha(I/'review_full.py'):raise ValueError('E1 source acceptance')
 ip=B/'e1/input_preparation_002/inputs.jsonl';pp=B/'data_manifest/panel_manifest.jsonl'
 for key,path in (('inputs',ip),('panel',pp)):
  item=p['files'][key]
  if item['path']!=str(path) or item['sha256']!=sha(path):raise ValueError('accepted input or panel changed')
 anchorpath=B/'e6/pixelprune_t045_dev384/plan001.json';pilotpath=B/'e4_c_multibudget_expansion/pilot_plan001.json';budgetpath=B/'initial_plan.json'
 anchors=read(anchorpath)['anchors'];inputs=lines(ip);panel=[r for r in lines(pp) if r.get('split')=='DEV'];pilot=read(pilotpath)
 key=lambda x:(x['task'],str(x['qid']),x['group_id'])
 im={key(x):x for x in inputs};pm={key(x):x for x in panel};am={key(x):x for x in anchors}
 if len(im)!=384 or len(pm)!=384 or len(am)!=384 or set(im)!=set(pm) or set(im)!=set(am) or any(x.get('split')!='DEV' for x in pm.values()):raise ValueError('E1 DEV metadata inventory')
 rows=[]
 for k,a in am.items():
  x=im[k]
  if any(x.get(z)!=a.get(z) for z in ('task','group_id','image_sha256','message_sha256')) or str(x.get('qid'))!=str(a.get('qid')):raise ValueError('anchor/input identity')
  rows.append({z:x[z] for z in ('task','qid','group_id','image_path','image_sha256','message_sha256')}|{'split':'DEV','processor_reference':a['processor'],'image_start_reference':a['image_start']})
 selected=contract.build(rows,pilot);budget=read(budgetpath);pilotterm=read(B/'e4_c_multibudget_worker_v3/monitor_run001/terminal.json')
 if budget.get('budget_start_unix')!=1793806214 or budget.get('budget_end_unix')!=1794065414 or pilotterm.get('reason')!='completed' or pilotterm.get('worker_returncode')!=0 or pilotterm.get('remaining_process_group')!=[] or not isinstance(pilotterm.get('elapsed_seconds'),(int,float)):raise ValueError('budget/pilot status')
 # This naive 7920/1520 scale deliberately includes pilot startup each time;
 # it is descriptive, not a safe time bound or a GPU authorization.
 elapsed=pilotterm['elapsed_seconds'];estimate=elapsed*7920/1520
 sources={'selection_plan_sha256':sha(D/'SELECTION_PLAN.md'),'selection_freeze_sha256':sha(D/'selection_plan_freeze001.json'),'schedule_contract_sha256':sha(D/'SCHEDULE_CONTRACT.md'),'cpu_proof_sha256':sha(D/'cpu_proof002.json'),'cpu_started_sha256':sha(D/'monitor_cpu002/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu002/terminal.json'),'e1_proof_sha256':sha(proofpath),'e1_review_sha256':sha(reviewpath),'e1_formal_started_sha256':sha(fm/'started.json'),'e1_formal_terminal_sha256':sha(fm/'terminal.json'),'e1_review_started_sha256':sha(rm/'started.json'),'e1_review_terminal_sha256':sha(rm/'terminal.json'),'inputs_sha256':sha(ip),'panel_sha256':sha(pp),'anchors_sha256':sha(anchorpath),'pilot_sha256':sha(pilotpath),'pilot_run_terminal_sha256':sha(B/'e4_c_multibudget_worker_v3/monitor_run001/terminal.json'),'budget_sha256':sha(budgetpath)}
 selected.update(status='proposal_frozen',scope='posthoc exploratory proposal; no GPU/actual model or prediction results',created_unix=time.time(),sources=sources,source_sha256=cpu['source_sha256_after'],counts={'measurement_images':192,'warmup_images':30,'measurement_jobs':7680,'warmup_jobs':240,'jobs':7920,'fixed32_reference_arms':888},budget={'start_unix':budget['budget_start_unix'],'end_unix':budget['budget_end_unix'],'pilot_elapsed_seconds':elapsed,'pilot_jobs':1520,'naive_scaled_7920_seconds':estimate,'interpretation':'rough wall-clock scale only; excludes 888 fresh references and future preparation/verification, and is not an approved timeout or forecast'},limits=['posthoc DEV; not confirmatory','no CONFIRM/FINAL or labels','old 30 warmup only; all 62 prior pilot groups excluded from measured cohort','processor anchor metadata not independently reconstructed','requires fresh 888 fixed32 identity references and separately frozen analysis','proposal cannot launch GPU'])
 with dest.open('x') as f:json.dump(selected,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('frozen 192-image exploration proposal; no GPU authority')
if __name__=='__main__':main()
