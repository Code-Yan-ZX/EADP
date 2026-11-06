"""Independent graph-traversal audit of the post-hoc document split; no labels."""
import argparse, collections, hashlib, json, re
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R/'m3_train_docsplit'; I=R/'e1_independent_review'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def readlines(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip(): yield json.loads(line)
def terminal(p):
 s=load(p/'started.json');t=load(p/'terminal.json')
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[]
 assert s['worker_pid']==s['process_group']==t['worker_pid']
 return s,t
def run(out):
 plan=P/'plan_001.json'; result=P/'run_001/result.json'; assigned=P/'run_001/assignments.jsonl'
 rows=R/'m3_train_data/train_audit_001/rows.jsonl'; manifest=R/'data_manifest/full_manifest.jsonl'
 p=load(plan); res=load(result); s,t=terminal(P/'monitor_run001')
 assert s['command']==['/home/dell/miniconda3/bin/python',str(P/'run.py')] and s['cwd']==str(R)
 assert res['status']=='completed' and res['plan_sha256']==digest(plan)
 assert res['source_rows_sha256']==p['source_rows_sha256']==digest(rows)
 assert res['eval_manifest_sha256']==p['eval_full_manifest_sha256']==digest(manifest)
 assert res['assignments_sha256']==digest(assigned) and res['source_sha256']==digest(P/'run.py')
 assert res['review_sha256']==digest(P/'independent_review.json')
 assert all(digest(k)==v for k,v in p['evidence_sha256'].items())
 assert all(res.get(k) is False for k in ('labels_opened','predictions_opened','gpu_used','training_performed'))
 evdocs=set(); n=0
 for x in readlines(manifest):
  if x['task']=='DocVQA_VAL':
   n+=1; m=re.fullmatch(r'(.+)_p(\d+)',Path(x['source_image_path']).stem)
   assert m; evdocs.add(m[1])
 assert n==p['docvqa_eval_rows']
 source=list(readlines(rows)); assert len(source)==p['row_count']==39463
 graph=collections.defaultdict(set); qs=set(); prior={}; buckets=collections.defaultdict(list)
 for x in source:
  q=x['question_id'];g=x['group_id'];d=x['document_id'];sp=x['split']
  assert type(q) is int and q not in qs and isinstance(g,str) and g and isinstance(d,str) and d and d.strip()==d and sp in ('fit','validation','excluded_eval')
  qs.add(q); assert g not in prior or prior[g]==sp; prior[g]=sp
  a=('d',d);b=('g',g);graph[a].add(b);graph[b].add(a);buckets[g].append(x)
 visited=set(); comps={}
 for start in graph:
  if start in visited: continue
  todo=[start];visited.add(start);docs=set();groups=set()
  while todo:
   node=todo.pop();(docs if node[0]=='d' else groups).add(node[1])
   for other in graph[node]:
    if other not in visited:visited.add(other);todo.append(other)
  canonical=json.dumps({'docs':sorted(docs),'groups':sorted(groups)},sort_keys=True,separators=(',',':'),ensure_ascii=True)
  cid=hashlib.sha256(canonical.encode()).hexdigest(); assert cid not in comps
  rr=[x for g in groups for x in buckets[g]]
  comps[cid]={'docs':docs,'groups':groups,'rows':rr,'excluded':bool(docs&evdocs or any(prior[g]=='excluded_eval' for g in groups))}
 eligible=[c for c in comps if not comps[c]['excluded']]
 eligible.sort(key=lambda c:(hashlib.sha256(('20261105|TRAIN_DOC_COMPONENT|'+c).encode()).hexdigest(),c))
 cut=4*len(eligible)//5; mapping={c:('fit' if i<cut else 'validation') for i,c in enumerate(eligible)}
 mapping.update({c:'excluded_eval' for c in comps if comps[c]['excluded']})
 expected={}
 summary={'source_rows':len(source),'components':len(comps),'row_counts':{},'document_counts':{},'group_counts':{},'component_counts':{},'max_component_rows':{},'max_component_documents':{},'max_component_groups':{}}
 for cid,c in comps.items():
  for x in c['rows']:
   q=x['question_id'];assert q not in expected
   expected[q]={'question_id':q,'group_id':x['group_id'],'document_id':x['document_id'],'component_id':cid,'split':mapping[cid]}
 for sp in ('fit','validation','excluded_eval'):
  cs=[c for cid,c in comps.items() if mapping[cid]==sp]
  for metric,key in [('row_counts','rows'),('document_counts','docs'),('group_counts','groups')]:summary[metric][sp]=sum(len(c[key]) for c in cs)
  summary['component_counts'][sp]=len(cs)
  for metric,key in [('max_component_rows','rows'),('max_component_documents','docs'),('max_component_groups','groups')]:summary[metric][sp]=max((len(c[key]) for c in cs),default=0)
 den=summary['row_counts']['fit']+summary['row_counts']['validation']; summary['fit_fraction_eligible_rows']=summary['row_counts']['fit']/den if den else None
 den=summary['group_counts']['fit']+summary['group_counts']['validation']; summary['fit_fraction_eligible_groups']=summary['group_counts']['fit']/den if eligible else None
 assert summary==res['summary']
 actual=list(readlines(assigned)); assert len(actual)==len(source) and [x['question_id'] for x in actual]==sorted(qs)
 assert all(x==expected[x['question_id']] for x in actual)
 proof={'status':'passed','independent_review':True,'rows_verified':len(actual),'components_verified':len(comps),'eval_rows_verified':n,'component_algorithm':'independent bipartite graph traversal','run_result_sha256':digest(result),'run_started_sha256':digest(P/'monitor_run001/started.json'),'run_terminal_sha256':digest(P/'monitor_run001/terminal.json'),'plan_sha256':digest(plan),'assignments_sha256':digest(assigned),'reviewer_sha256':digest(__file__),'source_sha256':{'rows':digest(rows),'eval_manifest':digest(manifest),'run':digest(P/'run.py'),'core':digest(P/'core.py')},'GT_opened':False,'predictions_opened':False,'training_performed':False,'limitations':p['limitations'],'errors':[]}
 with Path(out).open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);run(a.parse_args().out)
