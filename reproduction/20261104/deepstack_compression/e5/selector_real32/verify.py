"""CPU evidence acceptance: failures count as observations, not omissions."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;sys.path[:0]=[str(BASE/'e4_c_microbench_pilot'),str(BASE/'e4_c_latency')]
from admission import admit,validate_container
from identity_gate import sha
from latency_admission import terminal
from input_preflight import load_accepted_e1_references,tensor_meta
from latency_worker import load_runtime
from real_worker import PLAN
from panel_reference import get_records
from pilot_worker import get_reference
from identity_contract import validate as validate_e1
from input_contract import compare_real
import inspect
if Path(inspect.getsourcefile(compare_real)).resolve()!=HERE/'input_contract.py':raise ValueError('real input module source mismatch')
from case_contract import partition,validate_candidate_ids
from gates import synthetic_eligibility,endpoint_eligible,reconstruction_claim

def accept(out,monitor):
 import torch
 out=Path(out).resolve();monitor=Path(monitor).resolve();s=terminal(monitor/'started.json',monitor/'terminal.json');launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());done=json.loads((out/'worker_complete.json').read_text());approval=admit(preflight_only=False,host=True)
 if launch.get('approval_bundle')!=approval or done.get('source_sha256')!=approval['source_sha256'] or done.get('status')!='completed' or done.get('images')!=32 or done.get('cases')!=96 or done.get('GT_opened') is not False or done.get('scoring_performed') is not False or (launch.get('controller') or {}).get('pid')!=s.get('worker_pid') or ex.get('controller')!=launch.get('controller'):raise ValueError('run/source completion mismatch')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure_type') is not None or ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[]:raise ValueError('container exit mismatch')
 obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];inspection=validate_container(launch,ex,obj,gpu=True);manifest=json.loads((HERE/'sample_manifest_001.json').read_text());pilot=manifest['pilot'];records=get_records(pilot,approval['prior_identity']['e4']);e1=BASE/'e1';rows,_=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');index={(x['task'],str(x['qid'])):x for x in rows};events=[json.loads(x) for x in (out/'journal.jsonl').read_text().splitlines() if x.strip()];chunks=partition(events,pilot)
 if events[0].get('cases')!=done.get('diagnostic_synthetic'):raise ValueError('synthetic records changed')
 try:allowed=synthetic_eligibility(events[0]['cases']);syn_error=None
 except ValueError as exc:allowed={'fused':False,'graph':False};syn_error=str(exc)
 if events[0].get('eligibility')!=allowed or events[0].get('validation_error')!=syn_error or done.get('synthetic_validation_error')!=syn_error:raise ValueError('synthetic eligibility mismatch')
 from transformers import AutoTokenizer
 from types import SimpleNamespace
 plan=json.loads(PLAN.read_text());_torch,_native,wrapper_class,_versions=load_runtime({'plan':plan});stub=wrapper_class.__new__(wrapper_class);stub.min_pixels=plan['recipe']['min_pixels'];stub.max_pixels=plan['recipe']['max_pixels'];stub.fps=None;stub.processor=SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False))
 arrays={};counts={'capture_failed':0,'e1_identity_failed':0,'instruction_reconstruction_failed':0,'author_e4_mismatch_or_unavailable':0,'cross_budget_inequality':0,'fused_mismatch_or_unavailable':0,'graph_mismatch_or_unavailable':0,'not_evaluated':0};setups={}
 for i,(sample,chunk) in enumerate(zip(pilot,chunks)):
  row=index[(sample['task'],str(sample['qid']))]
  if any(row.get(k)!=sample.get(k) for k in ('task','group_id','message_sha256','image_sha256')):raise ValueError('frozen input mismatch')
  for e in chunk:
   if e.get('status')=='graph_setup':
    k=e.get('budget')
    if k in setups:raise ValueError('duplicate graph capture')
    setups[k]=e['result']
  captures=[x for x in chunk if x.get('status')=='capture_archived'];failures=[x for x in chunk if x.get('status')=='capture_failed']
  if len(captures)+len(failures)!=1:raise ValueError('capture evidence count')
  exposed=[e for e in chunk if e.get('status')=='capture_generated'];clean=[e for e in chunk if e.get('status')=='capture_cleanup']
  if len(exposed)>1 or any(any(str(e.get(k))!=str(row.get(k)) for k in ('task','qid','group_id')) for e in exposed):raise ValueError('diagnostic exposure identity mismatch')
  if failures:
   counts['capture_failed']+=1;counts['not_evaluated']+=3;counts['author_e4_mismatch_or_unavailable']+=3;counts['cross_budget_inequality']+=3;counts['fused_mismatch_or_unavailable']+=3;counts['graph_mismatch_or_unavailable']+=3;continue
  if len(exposed)!=1 or len(clean)!=1 or clean[0].get('passed') is not True or clean[0].get('capture_completed') is not True:raise ValueError('capture cleanup/exposure mismatch')
  event=captures[0]
  if any(str(event.get(k))!=str(row.get(k)) for k in ('task','qid','group_id')):raise ValueError('capture evidence identity')
  path=out/('image_%03d'%i)/'capture.pt';digest=sha(path)
  if event.get('sha256')!=digest:raise ValueError('capture archive mismatch')
  arrays[str(path)]=digest;data=torch.load(path,map_location='cpu',weights_only=True);actual=data['actual'];args=data['selector_args'];ref=get_reference(row)
  if stub._tokenize_instruction(row['message'],dataset=row['task'])[0].tolist()!=data['instruction_ids']:raise ValueError('author instruction tokenizer mismatch')
  try:identity=validate_e1(actual,ref,tensor_meta)['status']=='passed'
  except ValueError:identity=False
  if identity!=(event.get('identity',{}).get('status')=='passed'):raise ValueError('E1 identity record inconsistent')
  counts['e1_identity_failed']+=not identity
  visual=bool(torch.equal(args[0],actual['visual'][0]))
  if visual!=event.get('visual_arg_equal'):raise ValueError('visual pruner argument mismatch')
  try:reconstruction=compare_real(torch,actual,args,data['instruction_ids'],ref['image_start'])
  except ValueError:reconstruction=None
  reported=event.get('reconstruction',{});good=reconstruction_claim(reconstruction,reported)
  counts['instruction_reconstruction_failed']+=not good
  metadata={'processor':{k:tensor_meta(actual['processor'][k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'main_visual':tensor_meta(actual['visual'][0]),'full_deepstack':[tensor_meta(x) for x in actual['visual'][1]]};numbers={}
  for k in (128,256,512):
   rec=records[(row['task'],str(row['qid']),k)];entries=[e for e in chunk if e.get('status')=='author_case_archived' and e.get('budget')==k];failed=[e for e in chunk if e.get('status')=='author_case_failed' and e.get('budget')==k]
   if len(entries)+len(failed)!=1:raise ValueError('author case inventory')
   if failed:numbers[k]=None;counts['author_e4_mismatch_or_unavailable']+=1;continue
   e=entries[0];path=out/('image_%03d'%i)/('case_%d.pt'%k);digest=sha(path)
   if e.get('sha256')!=digest:raise ValueError('case archive mismatch')
   arrays[str(path)]=digest;data_k=torch.load(path,map_location='cpu',weights_only=True);ids=data_k['ordered'][0].tolist();reported=e['result'];common=metadata['processor']==rec['processor'] and metadata['main_visual']==rec['main_visual'] and metadata['full_deepstack']==rec['full_deepstack'];flags={'e4_common_metadata_equal':common,'e4_order_equal':ids==rec['ordered_ids'],'e4_spatial_equal':sorted(ids)==rec['spatial_ids'],'e4_pruned_hash_equal':tensor_meta(data_k['pruned'][0])==rec['pruned_visual']}
   if reported.get('author_config')!={'visual_token_num':k,'alpha':.5,'beta':2.,'visual_dim':4096,'spatial_merge_size':2} or reported.get('budget')!=k or reported.get('ordered_ids')!=ids or reported.get('spatial_ids')!=sorted(ids) or reported.get('similarity_meta')!=tensor_meta(data_k['similarity']) or reported.get('importance_meta')!=tensor_meta(data_k['importance']) or any(reported.get(x)!=v for x,v in flags.items()):raise ValueError('author endpoint evidence mismatch')
   if k==256 and ids!=actual['ordered'].tolist():flags['capture_order_equal']=False
   counts['author_e4_mismatch_or_unavailable']+=not all(flags.values());numbers[k]=data_k
  comparisons=[x for x in chunk if x.get('status')=='cross_budget_comparison']
  if len(comparisons)!=1:raise ValueError('cross-budget comparison missing')
  for k in (128,256,512):
   if numbers[k] is None or numbers[256] is None:expected={'available':False}
   else:expected={'available':True,'importance_equal_256':bool(torch.equal(numbers[k]['importance'],numbers[256]['importance'])),'similarity_equal_256':bool(torch.equal(numbers[k]['similarity'],numbers[256]['similarity']))}
   if comparisons[0]['values'].get(str(k))!=expected:raise ValueError('cross-budget tensor claim mismatch')
   if not expected.get('available') or not expected['importance_equal_256'] or not expected['similarity_equal_256']:counts['cross_budget_inequality']+=1
  baseline=numbers.get(256);capture_order_equal=baseline is not None and bool(torch.equal(baseline['ordered'].reshape(-1),actual['ordered'].reshape(-1)))
  for case in [x for x in chunk if x.get('status')=='case_completed']:
   r=case['result']
   if isinstance(r,str) or r.get('status')!='measured':
    counts['not_evaluated']+=1;counts['fused_mismatch_or_unavailable']+=1;counts['graph_mismatch_or_unavailable']+=1;continue
   if numbers[case['budget']] is None:raise ValueError('measured case missing author archive')
   author=[x for x in chunk if x.get('status')=='author_case_archived' and x.get('budget')==case['budget']][0]['result'];cross=comparisons[0]['values'][str(case['budget'])]
   if not endpoint_eligible({'status':'passed' if identity else 'failed'},{'status':'passed' if good else 'failed'},visual,author,cross,capture_order_equal):raise ValueError('measurement proceeded before endpoint identity')
   validate_candidate_ids(r,numbers[case['budget']]['ordered'][0].tolist(),allowed)
   for method in ('fused','graph'):
    m=[x for x in r['measurements'] if x['implementation']==method];same=r['identity'].get(method+'_equals_author') is True and all(x.get('equals_author') is True for x in m)
    if r['candidate_matches_author'].get(method)!=same:raise ValueError('candidate status mismatch')
    counts[method+'_mismatch_or_unavailable']+=not same
 evidence=[out/x for x in ('launch.json','exit.json','journal.jsonl','worker_complete.json')]+[monitor/'started.json',monitor/'terminal.json']
 return {'status':'passed','images':32,'cases':96,'counts':counts,'synthetic':done['diagnostic_synthetic'],'synthetic_eligibility':allowed,'graph_setups':setups,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'files_sha256':{str(p):sha(p) for p in evidence},'array_sha256':arrays,'source_sha256':approval['source_sha256'],'verifier_sha256':sha(__file__),'inspection':inspection,'limitations':['passed means complete evidence, not all candidate or reconstruction cases passed','E4 tensors available only as saved hashes and metadata, not original arrays','GPU mean equality reported by runtime; CPU reconstruction may round differently','selector comparison is post-pilot exploratory and not end-to-end TTFT']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--monitor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:r=accept(a.run_dir,a.monitor)
 except Exception as e:r={'status':'failed','errors':[{'type':type(e).__name__,'message':str(e)}],'GT_opened':False,'scoring_performed':False}
 r['finished_unix']=time.time()
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if r['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
