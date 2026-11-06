"""Post-hoc document×image-group transitive separation, no data access."""
import collections,hashlib,json
SPLITS=('fit','validation','excluded_eval')
class DSU:
 def __init__(self):self.p={}
 def find(self,x):
  self.p.setdefault(x,x)
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  x,y=self.find(a),self.find(b)
  if x!=y:self.p[y]=x
def compute(rows,eval_docs):
 rows=list(rows)
 if any(not isinstance(d,str) or not d or d.strip()!=d for d in eval_docs):raise ValueError('invalid eval document')
 graph=DSU();qids=set();group_prior={}
 for r in rows:
  d,g,s,q=r['document_id'],r['group_id'],r['split'],r['question_id']
  if not isinstance(d,str) or not d or d.strip()!=d or not isinstance(g,str) or not g or s not in SPLITS or type(q) is not int or q in qids:raise ValueError('invalid source metadata')
  qids.add(q)
  if g in group_prior and group_prior[g]!=s:raise ValueError('original image group crosses splits')
  group_prior[g]=s;graph.union(('doc',d),('group',g))
 comps=collections.defaultdict(lambda:{'docs':set(),'groups':set(),'rows':[]})
 for r in rows:
  c=comps[graph.find(('doc',r['document_id']))];c['docs'].add(r['document_id']);c['groups'].add(r['group_id']);c['rows'].append(r)
 data={}
 for c in comps.values():
  text=json.dumps({'docs':sorted(c['docs']),'groups':sorted(c['groups'])},sort_keys=True,separators=(',',':'),ensure_ascii=True)
  cid=hashlib.sha256(text.encode()).hexdigest()
  if cid in data:raise ValueError('component collision')
  c['excluded']=bool(c['docs']&eval_docs or any(group_prior[g]=='excluded_eval' for g in c['groups']));data[cid]=c
 eligible=sorted((cid for cid,c in data.items() if not c['excluded']),key=lambda cid:(hashlib.sha256(('20261105|TRAIN_DOC_COMPONENT|'+cid).encode()).hexdigest(),cid));cut=len(eligible)*4//5
 selected={cid:('fit' if i<cut else 'validation') for i,cid in enumerate(eligible)}
 for cid,c in data.items():
  if c['excluded']:selected[cid]='excluded_eval'
 out=[];doc_splits=collections.defaultdict(set);group_splits=collections.defaultdict(set)
 for cid,c in data.items():
  split=selected[cid]
  for d in c['docs']:doc_splits[d].add(split)
  for g in c['groups']:group_splits[g].add(split)
  for r in c['rows']:out.append({'question_id':r['question_id'],'group_id':r['group_id'],'document_id':r['document_id'],'component_id':cid,'split':split})
 if len(out)!=len(rows) or any(len(x)!=1 for x in doc_splits.values()) or any(len(x)!=1 for x in group_splits.values()) or any(selected[cid]!='excluded_eval' and c['docs']&eval_docs for cid,c in data.items()):raise ValueError('leakage or accounting')
 summary={'source_rows':len(rows),'components':len(data),'row_counts':{},'document_counts':{},'group_counts':{},'component_counts':{},'max_component_rows':{},'max_component_documents':{},'max_component_groups':{}}
 for split in SPLITS:
  cs=[c for cid,c in data.items() if selected[cid]==split]
  summary['row_counts'][split]=sum(len(c['rows']) for c in cs);summary['document_counts'][split]=sum(len(c['docs']) for c in cs);summary['group_counts'][split]=sum(len(c['groups']) for c in cs);summary['component_counts'][split]=len(cs)
  for metric,key in [('max_component_rows','rows'),('max_component_documents','docs'),('max_component_groups','groups')]:summary[metric][split]=max((len(c[key]) for c in cs),default=0)
 eligible_rows=summary['row_counts']['fit']+summary['row_counts']['validation'];summary['fit_fraction_eligible_rows']=summary['row_counts']['fit']/eligible_rows if eligible_rows else None
 summary['fit_fraction_eligible_groups']=summary['group_counts']['fit']/(summary['group_counts']['fit']+summary['group_counts']['validation']) if eligible else None
 return sorted(out,key=lambda x:x['question_id']),summary
