"""Load six previously independently accepted natural outputs as identities."""
import json,importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent
name='_pixel_t020_timing_admission';path=D/'admission.py';admission=sys.modules.get(name)
if admission is not None and (not getattr(admission,'__file__',None) or Path(admission.__file__).resolve()!=path):raise ValueError('foreign timing admission')
if admission is None:
 spec=importlib.util.spec_from_file_location(name,path);admission=importlib.util.module_from_spec(spec);sys.modules[name]=admission
 try:spec.loader.exec_module(admission)
 except BaseException:del sys.modules[name];raise
B=D.parents[1];RUN=B/'e6/pixel_wavefront_model_t020_run_v2';A=B/'e6/pixel_wavefront_t020_accept_v2';I=B/'e1_independent_review'
def load():
 approval=admission.require_identity();plan=json.loads((RUN/'execution_plan001.json').read_text());formal=json.loads((A/'verification001.json').read_text());review=json.loads((I/'pixel_wavefront_t020_run_v2_result_review.json').read_text());refs={}
 for job in plan['jobs']:
  p=(RUN/'run_001'/f"arm_{job['order']:03d}"/'observations'/'record.json').resolve();digest=admission.sha(p)
  if formal['files_sha256'].get(str(p))!=digest or review['files_sha256'].get(str(p))!=digest:raise ValueError('reference record evidence mismatch')
  r=json.loads(p.read_text());key=(job['task'],str(job['qid']),job['arm'])
  if key in refs or r.get('status')!='completed' or any(str(r.get(k))!=str(job[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or r.get('path')!=job['arm'] or r.get('GT_opened') is not False or r.get('scoring_performed') is not False:raise ValueError('reference identity')
  refs[key]={'processor':r['processor'],'selected_indices':r['selected_indices'],'tokens':r['output_ids'],'eos_ids':r['eos_ids'],'generation_kwargs':r['generation_scalars'],'selected_count':len(r['selected_indices']),'record_sha256':digest}
 if len(refs)!=6:raise ValueError('six identity records required')
 return refs,approval
