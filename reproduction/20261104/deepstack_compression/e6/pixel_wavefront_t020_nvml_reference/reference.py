"""Bind nine previously accepted same-anchor identities; no model or GPU access."""
import hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';DESIGN=B/'e6/pixel_wavefront_t020_nvml_nine/design001.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('reference supervision')
def load_file(name,p,expected):
 p=Path(p).resolve()
 if name in sys.modules or sha(p)!=expected:raise ValueError('reference source or module identity')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def validate_t0(record,raw,anchor):
 if record.get('status')!='completed' or any(str(record.get(k))!=str(anchor[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('t0 row identity')
 ids=record.get('selected_indices');count=record.get('selected_count');tokens=raw.get('generated_token_ids');full=raw.get('full_sequence_ids');n=raw.get('prompt_length');eos=record.get('eos_token_ids')
 if type(count) is not int or not 1<=count<=1024 or not isinstance(ids,list) or len(ids)!=count or ids!=sorted(set(ids)) or any(type(x) is not int or x<0 or x>=1024 for x in ids):raise ValueError('t0 ordered indices')
 if type(n) is not int or n<1 or not isinstance(full,list) or not isinstance(tokens,list) or not tokens or full[n:]!=tokens or len(tokens)!=record.get('generated_count') or not eos or bool(tokens[-1] in eos)!=record.get('ends_eos') or not (record.get('ends_eos') is True or len(tokens)==2048):raise ValueError('t0 generated identity')
 recipe={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
 if record.get('generation_kwargs')!=recipe:raise ValueError('t0 generation recipe')
 checks=record.get('diagnostic_checks');expected={'unique_selection','vision_and_three_ds','same_main_text_positions','natural_complete','calls_match_tokens','hooks_removed','positions_and_kv'}
 if not isinstance(checks,dict) or set(checks)!=expected or any(x is not True for x in checks.values()):raise ValueError('t0 diagnostics')
 return {'processor':record['processor_identity']['processor'],'selected_indices':ids,'selected_count':count,'tokens':tokens,'eos_ids':eos,'generation_kwargs':record['generation_kwargs']}
def load_refs():
 design=read(DESIGN);jobs=design['jobs']
 if design.get('gpu_authorized') is not False or len(jobs)!=9 or [j['order'] for j in jobs]!=list(range(9)) or len({(x['task'],str(x['qid']),x['arm']) for x in jobs})!=9 or any(sha(p)!=v for p,v in design['prior_proof_sha256'].items()):raise ValueError('design drift')
 prior=B/'e6/pixel_wavefront_t020_model_timing_v2/reference.py';plan=read(B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_plan001.json');module=load_file('_pixel_memory_t020_ref',prior,plan['source_sha256'][str(prior.resolve())]);r020,approval=module.load()
 formal=B/'e6/pixelprune_dev384_accept/verification001.json';review=I/'pixel_dev384_result_review.json';reviewer=I/'review_pixel_dev384_result.py';f=read(formal);q=read(review)
 terminal(B/'e6/pixelprune_dev384_accept/monitor_verify001',B/'e6/pixelprune_dev384_accept/verify.py',formal);terminal(I/'monitor_pixel_dev384_result',reviewer,review)
 if f.get('status')!='passed' or q.get('status')!='passed' or q.get('independent_review') is not True or f.get('questions')!=384 or q.get('questions')!=384 or f.get('errors')!=[] or q.get('errors')!=[] or q.get('formal_sha256')!=sha(formal) or q.get('reviewer_sha256')!=sha(reviewer) or q.get('formal_started_sha256')!=sha(B/'e6/pixelprune_dev384_accept/monitor_verify001/started.json') or q.get('formal_terminal_sha256')!=sha(B/'e6/pixelprune_dev384_accept/monitor_verify001/terminal.json'):raise ValueError('t0 acceptance')
 refs={};anchors={}
 for job in jobs:
  key=(job['task'],str(job['qid']));anchors[key]=job
  if job['arm'].startswith('t020_'):
   arm=job['arm'][5:];ref=r020[(key[0],key[1],arm)];refs[(key[0],key[1],job['arm'])]=ref
 if len(anchors)!=3 or len(refs)!=6:raise ValueError('six t020 references')
 for key,anchor in anchors.items():
  stem=key[0]+'_'+key[1];base=B/'e6/pixelprune_dev384/run_smoke001/observations';p=(base/(stem+'.json')).resolve();g=(base/(stem+'_generated.json')).resolve()
  for x in (p,g):
   digest=sha(x)
   if f['files_sha256'].get(str(x))!=digest or q['per_question_files_sha256'].get(str(x))!=digest:raise ValueError('t0 evidence hash')
  value=validate_t0(read(p),read(g),anchor);value['record_sha256']=sha(p);value['generated_sha256']=sha(g);refs[(*key,'t0_author')]=value
  peers=[refs[(*key,arm)] for arm in ('t020_author','t020_wavefront','t0_author')]
  if any(x['processor']!=peers[0]['processor'] or x['eos_ids']!=peers[0]['eos_ids'] for x in peers):raise ValueError('processor/EOS cross-arm identity')
 if len(refs)!=9:raise ValueError('nine references')
 return refs,{'design_sha256':sha(DESIGN),'t020_approval':approval,'t0_formal_sha256':sha(formal),'t0_review_sha256':sha(review),'reference_scope':'archived input identities, generated tokens, and selected indices; intermediate high-dimensional tensors not fully archived'}
