"""Draft same-arm DEV reference binder. No GT, no model and no GPU admission.

Not part of plan001; must be included in a subsequent frozen/reviewed source
inventory and joined to the real controller before GPU access is possible.
"""
import importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
spec=importlib.util.spec_from_file_location('_pixel_bootstrap_reference_v2',HERE/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all();pre=boot.pin('preflight',HERE/'preflight.py')

def supervisor(directory,script,output):
 s=pre.read(directory/'started.json');t=pre.read(directory/'terminal.json');cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 resolved=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid')
     or resolved.count(str(script.resolve()))!=1 or cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd)
     or resolved[cmd.index('--out')+1]!=str(output.resolve())):raise ValueError('reference supervisor failed')
 return s,t

def pixel_refs(anchors):
 pre.validate_upstream();out={}
 for arm,sub,accept,review_name in [('pixel_t0','pixelprune_dev384','pixelprune_dev384_accept','pixel_dev384'),('pixel_t045','pixelprune_t045_dev384','pixelprune_t045_accept','pixel_t045_dev384')]:
  formal_path=BASE/'e6'/accept/'verification001.json';review_path=BASE/'e1_independent_review'/(review_name+'_result_review.json')
  formal=pre.read(formal_path);review=pre.read(review_path)
  reviewer=BASE/'e1_independent_review'/('review_'+review_name+'_result.py')
  supervisor(BASE/'e1_independent_review'/('monitor_'+review_name+'_result'),reviewer,review_path)
  if review.get('reviewer_sha256')!=pre.sha(reviewer):raise ValueError('pixel reviewer source mismatch')
  for a in anchors:
   key=(a['task'],str(a['qid']));d=BASE/'e6'/sub/'run_smoke001/observations';stem=key[0]+'_'+key[1]
   q=(d/(stem+'.json')).resolve();g=(d/(stem+'_generated.json')).resolve()
   for p in (q,g):
    h=pre.sha(p)
    if formal['files_sha256'].get(str(p))!=h or review['per_question_files_sha256'].get(str(p))!=h:raise ValueError('pixel reference evidence hash mismatch')
   record=pre.read(q);raw=pre.read(g)
   if record.get('task')!=key[0] or str(record.get('qid'))!=key[1] or any(record.get(k)!=a.get(k) for k in ('group_id','message_sha256','image_sha256')):raise ValueError('pixel reference identity mismatch')
   tokens=raw.get('generated_token_ids');k=record.get('selected_count');eos=record.get('eos_token_ids')
   if not isinstance(tokens,list) or not tokens or len(tokens)!=record.get('generated_count') or type(k) is not int or not 1<=k<=1024 or not eos:raise ValueError('pixel reference lacks output/K/EOS')
   if raw.get('full_sequence_ids',[])[raw.get('prompt_length',-1):]!=tokens:raise ValueError('pixel raw token mismatch')
   diagnostics=record.get('diagnostic_checks',{})
   expected_checks={'unique_selection','vision_and_three_ds','same_main_text_positions','natural_complete','calls_match_tokens','hooks_removed','positions_and_kv'}
   if record.get('status')!='completed' or set(diagnostics)!=expected_checks or any(v is not True for v in diagnostics.values()):raise ValueError('pixel reference diagnostic failed')
   if bool(tokens[-1] in eos)!=record.get('ends_eos') or not (record.get('ends_eos') is True or len(tokens)==2048):raise ValueError('pixel EOS mismatch')
   out[(arm,*key)]={'tokens':tokens,'eos_ids':eos,'selected_count':k,'processor':record['processor_identity']['processor'],'evidence_sha256':{str(q):pre.sha(q),str(g):pre.sha(g)}}
 return out

def eadp_refs(anchors):
 proof=BASE/'e1/e1full001_completion.json';start=BASE/'e1/monitorverifye1full001/started.json';terminal=BASE/'e1/monitorverifye1full001/terminal.json'
 review=BASE/'e1_independent_review/e1full001_review.json';r=pre.read(review);f=pre.read(proof)
 verifier=BASE/'e1_analysis/completion.py';reviewer=BASE/'e1_independent_review/review_full.py'
 # Completion verifier path is verified by its own proof, and the review's command and source are checked here.
 supervisor(BASE/'e1_independent_review/monitor_e1full001',reviewer,review)
 if r.get('reviewer_sha256')!=pre.sha(reviewer):raise ValueError('E1 reviewer source mismatch')
 s0=pre.read(start);t0=pre.read(terminal)
 if t0.get('reason')!='completed' or t0.get('worker_returncode')!=0 or t0.get('remaining_process_group')!=[] or s0.get('worker_pid')!=t0.get('worker_pid'):raise ValueError('E1 verifier supervisor failed')
 if f.get('status')!='passed' or f.get('natural_questions')!=384 or r.get('status')!='passed' or r.get('n_verified')!=384 or r.get('errors')!=[] or r.get('completion_proof_sha256')!=pre.sha(proof) or r.get('completion_started_sha256')!=pre.sha(start) or r.get('completion_terminal_sha256')!=pre.sha(terminal):raise ValueError('E1 independent reference gate failed')
 journal=Path(f['files']['journal']['path']);expected=f['files']['journal']['sha256']
 if pre.sha(journal)!=expected:raise ValueError('E1 journal mismatch')
 wanted={(a['task'],str(a['qid'])):a for a in anchors};out={}
 with journal.open() as src:
  for line in src:
   record=json.loads(line);key=(record['task'],str(record['qid']))
   if key not in wanted or record.get('arm')!='P3-D1' or record.get('forced') is not False:continue
   a=wanted[key]
   if key in out or any(record.get(k)!=a.get(k) for k in ('group_id','message_sha256')):raise ValueError('E1 reference identity/uniqueness')
   tokens=record.get('output_ids');eos=record.get('eos_ids')
   if not isinstance(tokens,list) or not tokens or not eos or len(record.get('ordered_ids',[]))!=256:raise ValueError('E1 reference output/K/EOS')
   out[key]={'tokens':tokens,'eos_ids':eos,'selected_count':256,'processor':record['processor'],'evidence_sha256':{str(journal):expected}}
 if len(out)!=3:raise ValueError('missing E1 references')
 return {('eadp_k256',*key):value for key,value in out.items()}

def load_refs(anchors):
 if len(anchors)!=3 or any(a.get('split')!='DEV' for a in anchors):raise ValueError('not three DEV anchors')
 refs=pixel_refs(anchors);refs.update(eadp_refs(anchors))
 if len(refs)!=9:raise ValueError('incomplete same-arm references')
 for anchor in anchors:
  key=(anchor['task'],str(anchor['qid']))
  values=[refs[(arm,*key)] for arm in ('pixel_t0','pixel_t045','eadp_k256')]
  if any(v['processor']!=values[0]['processor'] or v['eos_ids']!=values[0]['eos_ids'] for v in values[1:]):raise ValueError('cross-arm processor/EOS reference mismatch')
 return refs
