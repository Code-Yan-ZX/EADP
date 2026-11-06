"""Independent complete M3 panel audit. Only run after formal no-GT acceptance."""
import argparse,collections,hashlib,json,math,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;EXEC=ROOT/'m3_oracle_exec';sys.path.insert(0,str(EXEC))
import runtime_gate,container_identity
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f]
def question_events(p):
 # The teacher journal contains encoded reference tokens. Recognize and omit
 # that payload before JSON parsing; its existence/order is sufficient here.
 events=[]
 with Path(p).open('rb') as f:
  for raw in f:
   if raw.startswith(b'{"event":"target_encoded",'):
    events.append({'event':'target_encoded'})
   else:
    event=json.loads(raw)
    if event.get('event')=='target_encoded':raise ValueError('unexpected target record encoding')
    events.append(event)
 return events
def seed(*items):return int.from_bytes(hashlib.sha256('|'.join(map(str,('20261104',*items))).encode()).digest()[:8],'big')
def proposals(qid,greedy,utilities):
 assert len(greedy)==256 and len(set(greedy))==256 and len(utilities)==4
 ranks=[]
 for stream in utilities:
  assert len(stream)==1024 and all(math.isfinite(float(v)) for v in stream)
  rank=[0.]*1024
  for i,k in enumerate(sorted(range(1024),key=lambda k:(float(stream[k]),-k))):rank[k]=i/1023
  ranks.append(rank)
 rest=[i for i in range(1024) if i not in set(greedy)]
 pools={'MAIN':sorted(rest,key=lambda i:(-ranks[0][i],i))[:64],
        'MULTI':sorted(rest,key=lambda i:(-sum(r[i] for r in ranks)/4,i))[:64],
        'RANDOM':random.Random(seed(qid,'RANDOM','POOL')).sample(rest,64)}
 result={'seed_encoding':'SHA256 UTF-8 pipe fields, first 8 bytes big-endian','pool_size':64,'greedy_order':greedy,'pools':pools,'by_r':{}}
 for r in (8,16):
  block={'removed':greedy[-r:],'sources':{}}
  for name in ('MAIN','MULTI','RANDOM'):
   pool=pools[name];sets=[sorted(pool[:r])]
   for j in range(1,8):
    rng=random.Random(seed(qid,name,r,j))
    for _ in range(128):
     trial=sorted(rng.sample(pool,r))
     if trial not in sets:sets.append(trial);break
    else:break
   block['sources'][name]=sets
  count=min(len(x) for x in block['sources'].values());block['n_proposals_per_source']=count
  for name in block['sources']:block['sources'][name]=block['sources'][name][:count]
  result['by_r'][str(r)]=block
 return result
def check_question(row,e1,raw,events,decode,formal):
 counts=collections.Counter(x['event'] for x in events)
 singles=('e1_baseline_first_logits','e1_baseline_verified','reference_access_started','target_encoded','full_gradient_completed','question_started','search_completed','question_completed')
 assert all(counts[x]==1 for x in singles) and all(counts[x]==6 for x in ('generation_started','generation_returned','generation_completed'))
 assert counts['evaluation_started']==counts['evaluation_completed'] and 1<=counts['evaluation_completed']<=49
 assert set(counts)==set(singles)|{'evaluation_started','evaluation_completed','generation_started','generation_returned','generation_completed'}
 def one(name):return next(x for x in events if x['event']==name)
 names=[x['event'] for x in events]
 assert [names.index(x) for x in singles[:-1]]==sorted(names.index(x) for x in singles[:-1]) and names[-1]=='question_completed'
 first=one('e1_baseline_first_logits');binding=one('e1_baseline_verified');grad=one('full_gradient_completed');order=e1['ordered_ids']
 assert first['identical'] is True and first['actual']==first['expected']==e1['logits'][0]
 assert binding['qid']==str(row['qid']) and binding['arm']=='P3-D1' and binding['verified'] is True and binding['record_sha256']==hashlib.sha256(raw).hexdigest()
 assert binding['record_sha256']==formal['baseline_record_sha256'] and one('question_started')['baseline']=={k:v for k,v in binding.items() if k!='event'} and one('question_started')['python_version']==sys.version
 utility=grad['utility'];evidence=grad['evidence'];assert grad['gradient_streams']==4 and math.isfinite(grad['mean_nll'])
 assert evidence['features']==[e1['visual_sha256']]+e1['deepstack_full_sha256'] and len(evidence['gradients'])==len(evidence['gradient_nonzero_counts'])==4
 assert all(x['shape']==[1024,4096] and len(x['sha256'])==64 for x in evidence['gradients']) and all(type(x) is int and x>0 for x in evidence['gradient_nonzero_counts'])
 prop=proposals(str(row['qid']),order,utility);assert one('question_started')['proposal']==prop
 losses={};pending=None
 for event in events:
  if event['event']=='evaluation_started':
   key=tuple(event['spatial_ids']);assert pending is None and key not in losses;pending=key
  elif event['event']=='evaluation_completed':
   key=tuple(event['spatial_ids']);v=event['mean_nll'];assert key==pending and isinstance(v,(int,float)) and math.isfinite(v);losses[key]=v;pending=None
 assert pending is None;base=tuple(sorted(order));assert base in losses;expected={base};winners=[]
 for r in (8,16):
  block=prop['by_r'][str(r)];n=block['n_proposals_per_source'];assert 1<=n<=8
  for source in ('MAIN','MULTI','RANDOM'):
   best=(losses[base],-1,list(base))
   for i,add in enumerate(block['sources'][source]):
    assert len(add)==r and len(set(add))==r and not set(add)&set(order)
    selected=sorted(order[:-r]+add);assert len(selected)==len(set(selected))==256
    key=tuple(selected);expected.add(key);candidate=(losses[key],i,selected)
    if candidate[:2]<best[:2]:best=candidate
   winners.append({'r':r,'source':source,'proposal_index':best[1],'spatial_ids':best[2],'mean_nll':best[0],'n_proposals':n})
 search=one('search_completed');assert set(losses)==expected and search['unique_evaluations']==len(losses) and search['winners']==winners and search['baseline_nll']==losses[base]
 gen=[x for x in events if x['event'].startswith('generation_')];assert [x['event'] for x in gen]==['generation_started','generation_returned','generation_completed']*6
 for i,w in enumerate(winners):
  begin,returned,complete=gen[i*3:i*3+3];key={k:w[k] for k in ('source','r','spatial_ids')};assert all(all(x.get(k)==v for k,v in key.items()) for x in (begin,returned,complete))
  rawout=returned['output'];out=complete['output'];ids=out['token_ids'];assert rawout['token_ids']==ids and 1<=len(ids)<=2048 and all(type(x) is int and x>=0 for x in ids)
  eos=ids[-1] in e1['eos_ids'];assert out['eos']==eos and out['truncated']==(not eos and len(ids)==2048) and (eos or out['truncated'])
  assert all(x['eos_ids']==e1['eos_ids'] and x['generation_scalars']==e1['generation_scalars'] for x in (rawout,out)) and out['answer']==decode(ids) and out['actual_selection_verified'] is True
  obs=out['evidence'];selected=w['spatial_ids'];start=e1['image_start'];length=e1['original_length'];retained=list(range(start))+[start+x for x in selected]+list(range(start+1024,length));delta=e1['full_rope_delta'][0][0]+768
  assert obs['processor']==e1['processor'] and obs['full_streams']==[e1['visual_sha256']]+e1['deepstack_full_sha256'] and obs['spatial_ids']==selected and obs['retained_ids']==retained
  assert obs['position_ids']==[[[axis[0][j] for j in retained]] for axis in e1['original_position_ids']] and obs['rope_delta']==[[delta]] and obs['ds_mask']==[[False]*start+[True]*256+[False]*(length-start-1024)] and len(obs['selected_ds'])==3
  trace=obs['decode_trace'];assert len(trace)==len(ids);first=trace[0];n=len(retained)
  assert first['cache_position']==list(range(n)) and first['rope_delta']==[[delta]] and first['layer_cache_lengths'] in ([],[0]*36) and first['layer_cache_lengths_after']==[n]*36
  for step,x in enumerate(trace[1:],1):
   pos=n+step-1;assert x['cache_position']==[pos] and x['position_ids']==[[[pos+delta]]]*3 and x['rope_delta']==[[delta]] and x['layer_cache_lengths']==[pos]*36 and x['layer_cache_lengths_after']==[pos+1]*36
 assert one('question_completed')['generation_count']==6

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/m3oracle_panel_review.json'
 run=EXEC/'runtime_001';v2=ROOT/'m3_oracle_verify_v2';proof=v2/'runtime_001_verification_author.json';vs=v2/'monitor_verify002/started.json';vt=vs.parent/'terminal.json';rs=EXEC/'monitor_runtime001/started.json';rt=rs.parent/'terminal.json'
 runtime_gate.supervised(vs,vt,v2/'verify.py',proof);formal=read(proof)
 assert formal['status']=='passed' and formal['stage']=='oracle_full' and formal['n_questions']==128 and formal['new_generations']==768 and formal['scoring_performed'] is False and formal['reference_sources_opened_by_verifier'] is False and formal['verifier_sha256']==sha(v2/'verify.py')
 envpath=v2/'environment_002_v2.json';env=read(envpath);verifier_start=read(vs)
 assert env['status']=='passed' and env['supervisor_started_sha256']==sha(vs) and env['verifier_pid']==verifier_start['worker_pid'] and env['questions_checked']==128 and env['verifier_sha256']==sha(v2/'verify.py')
 assert env['python_version']==sys.version and env['recorded_version_counts']=={sys.version:128} and Path(sys.executable).resolve()==Path(env['actual_proc_executable']).resolve() and sha(sys.executable)==env['executable_sha256']
 assert env['GT_opened'] is False and env['prediction_files_opened'] is True and env['prediction_events_parsed'] is False and env['reference_token_events_parsed'] is False
 bindingpath=v2/'acceptance_binding_002.json';binding=read(bindingpath);assert binding['status']=='passed' and binding['environment_sha256']==sha(envpath) and binding['formal_proof_sha256']==sha(proof) and binding['formal_started_sha256']==sha(vs) and binding['formal_terminal_sha256']==sha(vt)
 assert binding['verifier_sha256']==sha(v2/'verify.py') and binding['executable_sha256']==sha(sys.executable) and binding['python_version']==sys.version and binding['n_questions']==128 and binding['new_generations']==768 and binding['reference_sources_opened'] is False and binding['scoring_performed'] is False
 v2snap=v2/'verify_002_sources';v2manifest=read(v2snap/'manifest.json');assert v2manifest['cpu_proof_sha256']==sha(v2/'cpu_check_001/verification.json')
 for path,digest in v2manifest['source_sha256'].items():assert sha(path)==sha(v2snap/Path(path).name)==digest
 assert v2manifest['source_sha256'][str(v2/'verify.py')]==sha(v2/'verify.py')
 launch=read(run/'launch.json');exit=read(run/'exit.json');start=read(rs);end=read(rt);assert end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[] and start['worker_pid']==end['worker_pid']==start['process_group']
 assert exit['returncode']==exit['container_exit_code']==0 and exit['container_stopped_verified'] is True and exit['client_exited_while_container_running'] is False and exit['timed_out'] is False and exit['interrupted'] is None and exit['errors']==[] and exit['gpu_pids_after']==[]
 obj=json.loads(subprocess.check_output(['docker','inspect',exit['container_identity']['container_id']],text=True))[0];container_identity.validate(run,launch,exit,obj)
 for field,path in [('run_start_sha256',rs),('run_terminal_sha256',rt),('launch_sha256',run/'launch.json'),('exit_sha256',run/'exit.json'),('panel_events_sha256',run/'panel/panel_events.jsonl'),('worker_complete_sha256',run/'panel/worker_complete.json')]:assert formal[field]==sha(path)
 approval=runtime_gate.validate(host=True,require_future=False);frozen=read(run/'frozen.json');assert frozen['approval']==approval and read(run/'host_receipt.json')==approval and launch['approval']==approval and formal['approval_sha256']==approval['approval_sha256']
 cpu=read(EXEC/'cpu_check_009/verification.json');src=cpu['source_sha256_after'];assert cpu['source_sha256_before']==src and all(sha(k)==v for k,v in src.items())
 snap=EXEC/'runtime_001_sources';manifest=read(snap/'manifest.json');assert manifest['source_sha256']==src
 assert all(sha(snap/'files'/str(Path(k)).lstrip('/'))==v for k,v in src.items())
 sample=read(ROOT/'m3_oracle/sample_001.json')['examples'];assert len(sample)==128 and collections.Counter(x['task'] for x in sample)=={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32}
 eproof=read(ROOT/'e1/e1full001_completion.json');journal=Path(eproof['files']['journal']['path']);inputs=Path(eproof['files']['inputs']['path']);assert sha(journal)==eproof['files']['journal']['sha256'] and sha(inputs)==eproof['files']['inputs']['sha256']
 baseline={};rawlines=journal.read_bytes().splitlines();assert len(rawlines)==1536
 for raw in rawlines:
  r=json.loads(raw)
  if r.get('arm')=='P3-D1' and r.get('forced') is False:
   key=(r['task'],str(r['qid']),r['group_id']);assert key not in baseline;baseline[key]=(r,raw)
 inputrows=lines(inputs);imap={(x['task'],str(x['qid']),x['group_id']):x for x in inputrows};assert len(inputrows)==len(imap)==384
 from transformers import AutoTokenizer
 recipe=read(eproof['files']['plan']['path'])['recipe'];tokenizer=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
 def decode(ids):return tokenizer.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
 pe=lines(run/'panel/panel_events.jsonl');begins=[x for x in pe if x['event']=='question_started'];ends=[x for x in pe if x['event']=='question_completed'];assert len(begins)==len(ends)==128 and pe[-1]['event']=='panel_completed'
 files=formal['result_files_sha256'];assert len(files)==128 and len(formal['questions'])==128
 seen=set()
 for i,row in enumerate(sample):
  key=(row['task'],str(row['qid']),row['group_id']);b,z=begins[i],ends[i];assert (b['task'],b['qid'],b['group_id'])==key==(z['task'],z['qid'],z['group_id']) and b['index']==z['index']==i and b['journal']==z['journal'] and z['new_generations']==6
  inp=imap[key];assert all(inp.get(k)==row.get(k) for k in ('image_path','image_sha256','message_sha256'));path=run/'panel'/b['journal'];assert path.parent.resolve()==(run/'panel').resolve() and str(path) in files and sha(path)==files[str(path)] and str(path) not in seen;seen.add(str(path))
  record,raw=baseline[key];check_question(row,record,raw,question_events(path),decode,formal['questions'][i])
 assert set(files)==seen
 result={'status':'passed','independent_review':True,'n_questions':128,'new_generations':768,'errors':[],
  'scoring_performed':False,'reference_sources_opened':False,'formal_proof_sha256':sha(proof),'formal_started_sha256':sha(vs),'formal_terminal_sha256':sha(vt),
  'frozen_sha256':sha(run/'frozen.json'),'host_receipt_sha256':sha(run/'host_receipt.json'),'reviewer_sha256':sha(__file__),'verifier_environment_sha256':sha(envpath),'acceptance_binding_sha256':sha(bindingpath),'verifier_snapshot_manifest_sha256':sha(v2snap/'manifest.json'),
  'scope':'independent proposal/selection, output decoding, position and cache-length replay; no scoring or source answers; unarchived tensors not reconstructed'}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','questions':128,'new_generations':768}))
if __name__=='__main__':main()
