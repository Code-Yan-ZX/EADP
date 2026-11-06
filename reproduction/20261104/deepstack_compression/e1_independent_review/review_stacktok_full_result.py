"""Independent full StackTok no-GT record/identity review; summaries only."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
B=Path(__file__).resolve().parent.parent;D=B/'e6/stacktok_full';R=D/'run_001';I=B/'e1_independent_review';sys.path.insert(0,str(D))
from admission import admit,sources,validate_container
from latency_admission import terminal
from input_preflight import load_accepted_e1_references
from full_refs import e4_records
from real_worker import PLAN
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def term(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cwd=Path(s['cwd']);c=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
 return s
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def meta(t):
 x=t.detach().contiguous().cpu();return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
def valid(m):return isinstance(m,dict) and set(m)=={'dtype','shape','sha256'} and isinstance(m['dtype'],str) and isinstance(m['shape'],list) and all(type(x) is int and x>0 for x in m['shape']) and re.fullmatch('[0-9a-f]{64}',m['sha256']) is not None
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==I/'stacktok_full_run001_review.json';torch.set_num_threads(1);assert not torch.cuda.is_initialized()
f=D/'run_001_verification.json';formal=read(f);term(D/'monitor_verify001',D/'verify.py',f);started=term(D/'monitor_run001',D/'launch.py');assert formal['status']=='passed' and formal['questions']==formal['generations']==384 and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['verifier_sha256']==sha(D/'verify.py');files=formal['files_sha256'];assert len(files)==10 and all(sha(k)==v for k,v in files.items())
launch=read(R/'launch.json');ex=read(R/'exit.json');assert launch['approval_bundle']==admit(preflight_only=False,host=True) and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and not ex['timed_out'] and ex['interrupted'] is None and ex['failure_type'] is None and ex['gpu_pids_after']==[] and launch['controller']==ex['controller'] and launch['controller']['pid']==started['worker_pid'];obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];container=validate_container(launch,ex,obj,gpu=True)
source={str(x.resolve()):sha(x) for x in sources()};frozen=read(R/'frozen.json');done=read(R/'worker_complete.json');assert frozen['source_sha256']==source and frozen['GT_opened'] is False and done['status']=='completed' and done['questions']==done['generations']==384
rows,_=load_accepted_e1_references(B/'e1/e1full001_completion.json',B/'e1/monitorverifye1full001/started.json',B/'e1/monitorverifye1full001/terminal.json');refs=e4_records(rows);records=lines(R/'records.jsonl');generated=lines(R/'generated.jsonl');returned=lines(R/'returned.jsonl');journal=lines(R/'journal.jsonl');assert len(rows)==len(refs)==len(records)==len(generated)==len(returned)==384 and len({(r['task'],str(r['qid']),r['group_id']) for r in rows})==384
expected_events=[(event,r['task'],str(r['qid'])) for r in rows for event in ('started','returned','generated','completed')];actual=[(x['event'],x.get('task'),str(x.get('qid'))) for x in journal if x['event']!='decode_progress'];assert expected_events==actual
model=read(PLAN)['recipe'];tokenizer=AutoTokenizer.from_pretrained(model['model_path'],local_files_only=True);counters={};
for index,(row,r,g,ret) in enumerate(zip(rows,records,generated,returned)):
 key=(row['task'],str(row['qid']));ref=refs[key];assert all(r[k]==row[k] for k in ('task','qid','group_id','message_sha256','image_sha256')) and r['path']=='local_stacktok' and r['forced'] is False and r['budget']==256 and r['selector_calls']==1
 assert all(r[k]==g[k] for k in ('task','qid','group_id','path','forced','output_ids','eos_ids','generation_scalars')) and (ret['task'],str(ret['qid']),ret['group_id'],ret['path'])==(key[0],key[1],row['group_id'],'local_stacktok')
 full=ret['full_ids'];n=ret['input_length'];ids=r['output_ids'];assert type(n) is int and n>0 and full[n:]==ids and meta(torch.tensor([full[:n]],dtype=torch.int64))==r['processor']['input_ids'] and r['processor']==ref['processor']
 expected_recipe={k:model[k] for k in ('top_p','top_k','temperature','repetition_penalty')};expected_recipe.update(do_sample=False,max_new_tokens=2048);assert r['generation_scalars']==ref['generation_scalars']==expected_recipe and r['eos_ids']==ref['eos_ids']
 eos=r['eos_ids'];assert ids and len(ids)<=2048 and all(type(x) is int and x>=0 for x in ids) and r['eos']==(ids[-1] in eos) and r['truncated']==(ids[-1] not in eos and len(ids)==2048) and (r['eos'] or r['truncated']) and tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer']
 text=' '.join(x['value'] for x in row['message'] if x.get('type')=='text' and x.get('value')).strip().replace('<image>','').strip();text='Question: '+text;assert r['question_text']==text and meta(tokenizer(text.split(),is_split_into_words=True,return_tensors='pt',padding=True,truncation=True)['input_ids'])==r['question_ids']
 assert r['main_visual']==ref['main_visual'] and r['full_deepstack']==ref['full_deepstack'] and r['rope_delta']==ref['full_delta'];spatial=r['spatial_ids'];assert len(spatial)==256 and spatial==sorted(set(spatial)) and all(type(x) is int and 0<=x<1024 for x in spatial) and len(r['greedy_ids'])==256 and sorted(r['greedy_ids'])==spatial
 start=ref['image_start'];length=ref['original_length'];keep=list(range(start))+[start+x for x in spatial]+list(range(start+1024,length));assert r['retained_indices']==meta(torch.tensor(keep));expected_pos=[[[axis[0][i] for i in keep]] for axis in ref['full_position_ids']];assert r['prefill_position_ids']==expected_pos and r['injected_mask']==[[False]*start+[True]*256+[False]*(length-start-1024)] and len(r['injected_deepstack'])==3
 trace=r['decode_trace'];assert len(trace)==len(ids) and trace[0]['position_ids']==expected_pos;compact=length-768;delta=r['rope_delta'][0][0]
 for j,x in enumerate(trace):
  assert x['rope_delta']==[[delta]] and x['layer_cache_lengths_after']==[compact+j]*36
  if j:assert x['cache_position']==[compact+j-1] and x['position_ids']==[[[length+j-1+delta]]]*3 and x['layer_cache_lengths']==[compact+j-1]*36
  else:assert x['cache_position']==list(range(compact)) and x['layer_cache_lengths'] in ([],[0]*36)
 arrays=r['arrays'];names={'premerger','averaged_premerger','visual','prefill','prefill_logits','question_ids','question_embeddings','prefill_mask','prefill_positions','visual_mask','retained_indices','ds0','ds1','ds2','injected0','injected1','injected2','processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw'};assert arrays['kind']=='summary_only' and set(arrays['tensors'])==names and all(valid(m) for m in arrays['tensors'].values())
 mappings={'premerger':'premerger','averaged_premerger':'averaged_premerger','visual':'main_visual','prefill':'prefill_embeds','prefill_logits':'prefill_logits','question_ids':'question_ids','question_embeddings':'question_embeddings','prefill_mask':'prefill_mask','prefill_positions':'prefill_positions','visual_mask':'visual_mask','retained_indices':'retained_indices'};assert all(arrays['tensors'][k]==r[v] for k,v in mappings.items()) and all(arrays['tensors']['ds'+str(j)]==r['full_deepstack'][j] and arrays['tensors']['injected'+str(j)]==r['injected_deepstack'][j] for j in range(3))
 for name in ('input_ids','attention_mask','pixel_values','image_grid_thw'):assert arrays['tensors']['processor_'+name]==r['processor'][name]
 counts=counters.setdefault(row['task'],{'n':0,'eos':0,'truncated':0,'empty':0});counts['n']+=1;counts['eos']+=int(r['eos']);counts['truncated']+=int(r['truncated']);counts['empty']+=int(not r['answer'].strip())
assert all(v['n']==128 for v in counters.values()) and source=={str(x.resolve()):sha(x) for x in sources()} and all(sha(k)==v for k,v in files.items()) and not torch.cuda.is_initialized()
result={'status':'passed','independent_review':True,'questions':384,'generations':384,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(f),'formal_started_sha256':sha(D/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'files_sha256':files,'source_sha256':source,'container':container,'task_counts':counters,'limitations':['This is local Qwen3 adaptation of author selector, not the original Qwen3 author integration.','The full panel preserves tensor summaries and runtime checks rather than full arrays; high-dimensional tensors and actual selected DS bytes cannot be independently reconstructed offline.','KV records verify per-step lengths and positions, not KV byte contents.','This review opens completed DEV mixed prediction journals only after formal acceptance; no answers from truth files and no scoring.'],'errors':[]};a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed',counters,flush=True)
