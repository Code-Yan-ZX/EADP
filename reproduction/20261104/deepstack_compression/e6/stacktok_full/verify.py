"""Full-panel no-GT acceptance. Summaries are not offline tensor reconstructions."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path[:0]=[str(HERE),str(BASE/'e4_c_latency'),str(BASE/'e6/stacktok_audit/prototype'),str(BASE/'e4')]
from admission import admit,sources,validate_container
from latency_admission import terminal
from real_worker import sha,PLAN
from input_preflight import load_accepted_e1_references
from full_refs import e4_records,validate_reference
from full_arm import validate_recipe,validate_trace
from identity_records import tensor_meta
from local_adapter import user_question

def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def metadata(m):
 return isinstance(m,dict) and set(m)=={'dtype','shape','sha256'} and isinstance(m['dtype'],str) and isinstance(m['shape'],list) and all(type(x) is int and x>0 for x in m['shape']) and re.fullmatch('[0-9a-f]{64}',m['sha256']) is not None

def validate_record(r,event,row,ref,recipe,tokenizer):
 import torch
 if any(r.get(k)!=row.get(k) for k in ('task','qid','group_id','image_sha256','message_sha256')) or r.get('path')!='local_stacktok' or r.get('budget')!=256 or r.get('selector_calls')!=1 or r.get('forced') is not False:raise ValueError('record identity')
 if any(r.get(k)!=event.get(k) for k in ('task','qid','group_id','path','forced','output_ids','eos_ids','generation_scalars')):raise ValueError('generated output journal differs')
 validate_recipe(r,recipe,False,ref['eos_ids']);validate_reference(r,ref);ids=r['output_ids'];eos=r['eos_ids']
 if not ids or len(ids)>2048 or any(type(i) is not int for i in ids) or r['eos']!=bool(ids[-1] in eos) or r['truncated']!=bool(len(ids)==2048 and ids[-1] not in eos) or not(r['eos'] or r['truncated']) or tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['answer']:raise ValueError('output termination/decode')
 if r['question_text']!=user_question(row['message']):raise ValueError('selector question differs')
 toks=tokenizer(r['question_text'].split(),is_split_into_words=True,return_tensors='pt',padding=True,truncation=True)['input_ids']
 if tensor_meta(toks)!=r['question_ids']:raise ValueError('selector tokenization differs')
 if len(r['greedy_ids'])!=256 or sorted(r['greedy_ids'])!=r['spatial_ids']:raise ValueError('selector greedy sequence')
 start=ref['image_start'];length=ref['original_length'];keep=torch.tensor(list(range(start))+[start+i for i in r['spatial_ids']]+list(range(start+1024,length)))
 if r['retained_indices']!=tensor_meta(keep):raise ValueError('retained token indices')
 validate_trace(r['decode_trace'],ids,length,256,int(r['rope_delta'][0][0]))
 arrays=r['arrays'];tensor_names={'premerger','averaged_premerger','visual','prefill','prefill_logits','question_ids','question_embeddings','prefill_mask','prefill_positions','visual_mask','retained_indices','ds0','ds1','ds2','injected0','injected1','injected2','processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw'}
 if arrays.get('kind')!='summary_only' or set(arrays.get('tensors',{}))!=tensor_names or any(not metadata(v) for v in arrays['tensors'].values()):raise ValueError('tensor summaries incomplete')
 pairs={'visual':'main_visual','prefill':'prefill_embeds','prefill_logits':'prefill_logits','question_ids':'question_ids','question_embeddings':'question_embeddings','prefill_mask':'prefill_mask','prefill_positions':'prefill_positions','visual_mask':'visual_mask','retained_indices':'retained_indices','premerger':'premerger','averaged_premerger':'averaged_premerger'}
 if any(arrays['tensors'][k]!=r[v] for k,v in pairs.items()) or any(arrays['tensors']['ds'+str(i)]!=r['full_deepstack'][i] or arrays['tensors']['injected'+str(i)]!=r['injected_deepstack'][i] for i in range(3)):raise ValueError('tensor summaries inconsistent')
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run=HERE/'run_001';s=HERE/'monitor_run001/started.json';t=HERE/'monitor_run001/terminal.json';started=terminal(s,t);launch=json.loads((run/'launch.json').read_text());ex=json.loads((run/'exit.json').read_text())
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or launch.get('controller')!=ex.get('controller') or launch['controller']['pid']!=started['worker_pid'] or launch['approval_bundle']!=admit(preflight_only=False,host=True):raise ValueError('run exit or approval')
 obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];validate_container(launch,ex,obj,gpu=True)
 frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text());expected={str(p.resolve()):sha(p) for p in sources()}
 if frozen.get('source_sha256')!=expected or frozen.get('plan_sha256')!=sha(HERE/'plan.json') or done.get('status')!='completed' or done.get('questions')!=384 or done.get('generations')!=384 or frozen.get('GT_opened') is not False:raise ValueError('full run incomplete')
 e1=BASE/'e1';rows,_=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');refs=e4_records(rows);records=lines(run/'records.jsonl');events=lines(run/'generated.jsonl');returned=lines(run/'returned.jsonl');journal=lines(run/'journal.jsonl')
 if len(rows)!=384 or len(records)!=384 or len(events)!=384 or len(returned)!=384:raise ValueError('incomplete panel')
 expected_events=[(status,row['task'],str(row['qid'])) for row in rows for status in ('started','returned','generated','completed')];actual=[(e['event'],e.get('task'),str(e.get('qid'))) for e in journal if e['event']!='decode_progress']
 if actual!=expected_events:raise ValueError('journal sequence or failures')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(json.loads(PLAN.read_text())['recipe']['model_path'],local_files_only=True);recipe=json.loads(PLAN.read_text())['recipe']
 for row,r,event,ret in zip(rows,records,events,returned):
  validate_record(r,event,row,refs[(row['task'],str(row['qid']))],recipe,tokenizer)
  n=ret['input_length'];full=ret['full_ids']
  if (ret.get('task'),str(ret.get('qid')),ret.get('group_id'),ret.get('path'))!=(row['task'],str(row['qid']),row['group_id'],'local_stacktok') or not isinstance(n,int) or n<=0 or full[n:]!=r['output_ids'] or tensor_meta(__import__('torch').tensor([full[:n]],dtype=__import__('torch').long))!=r['processor']['input_ids']:raise ValueError('returned token sequence mismatch')
 files=[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl','returned.jsonl','journal.jsonl')]+[s,t]
 proof={'status':'passed','questions':384,'generations':384,'GT_opened':False,'scoring_performed':False,'prediction_scope':'accepted E1/E4 DEV mixed journals parsed for identity and this complete panel parsed for validation','scope':'local Qwen3 StackTok K256; tensor summaries, not offline reconstruction','files_sha256':{str(p.resolve()):sha(p) for p in files},'verifier_sha256':sha(__file__)}
 with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
