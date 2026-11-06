"""Independently reconstruct archived StackTok prefill/DS and audit run (no GT)."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
B=Path(__file__).resolve().parent.parent;D=B/'e6/stacktok_smoke';R=D/'run_001';I=B/'e1_independent_review';sys.path.insert(0,str(D))
from admission import admit,validate_container,sources
from real_worker import PLAN
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def term(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cwd=Path(s['cwd']);c=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
 return s
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
dtypes={str(x):x for x in (torch.bfloat16,torch.float16,torch.float32,torch.float64,torch.int64,torch.int32,torch.uint8,torch.bool)}
def tensor(z,name,meta):
 b=z[name];raw=b.tobytes();assert b.dtype==np.uint8 and hashlib.sha256(raw).hexdigest()==meta['sha256'];x=torch.frombuffer(bytearray(raw),dtype=dtypes[meta['dtype']]);assert x.numel()==int(np.prod(meta['shape']));return x.reshape(meta['shape'])
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'stacktok_smoke001_review.json';torch.set_num_threads(1);assert not torch.cuda.is_initialized()
f=D/'run_001_verification.json';formal=read(f);vs=term(D/'monitor_verify001',D/'verify.py',f);rs=term(D/'monitor_run001',D/'launch.py');assert formal['status']=='passed' and formal['images']==3 and formal['generations']==18 and formal['keepall_pairs']==6 and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['verifier_sha256']==sha(D/'verify.py') and formal['run_started_sha256']==sha(D/'monitor_run001/started.json') and formal['run_terminal_sha256']==sha(D/'monitor_run001/terminal.json') and formal['run_exit_sha256']==sha(R/'exit.json')
launch=read(R/'launch.json');ex=read(R/'exit.json');assert launch['approval_bundle']==admit(preflight_only=False,host=True) and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None and launch['controller']['pid']==rs['worker_pid']==ex['controller']['pid'];ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];container=validate_container(launch,ex,ins,gpu=True)
frozen=read(R/'frozen.json');src={str(x.resolve()):sha(x) for x in sources()};assert frozen['source_sha256']==src and read(R/'worker_complete.json')['generations']==18
records=lines(R/'records.jsonl');generated=lines(R/'generated.jsonl');journal=lines(R/'journal.jsonl');pairs=lines(R/'comparisons.jsonl');plan=read(D/'plan001.json');assert len(records)==len(generated)==18 and len(pairs)==6
started=[x for x in journal if x.get('event')=='started'];completed=[x for x in journal if x.get('event')=='recorded'];assert len(started)==len(completed)==18 and not any(x.get('event')=='failed' for x in journal)
tok=AutoTokenizer.from_pretrained(read(PLAN)['recipe']['model_path'],local_files_only=True);forced_id=tok.encode(' the',add_special_tokens=False)[0];array_sha={};summaries=[]
keys=('task','qid','group_id','image_sha256','message_sha256','forced','processor','main_visual','full_deepstack','prefill_embeds','prefill_mask','prefill_position_ids','rope_delta','injected_mask','injected_deepstack','prefill_logits','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','decode_trace','cache_fingerprints')
for z,anchor in enumerate(plan['anchors']):
 for fi,forced in enumerate((False,True)):
  b=(z*2+fi)*3;group=records[b:b+3];ev=generated[b:b+3]
  for j,(rec,event,path) in enumerate(zip(group,ev,('stock_native','local_keepall','local_stacktok'))):
   assert rec['path']==path and rec['forced']==forced and all(rec[k]==anchor[k] for k in ('task','qid','group_id','image_sha256','message_sha256')) and all(rec[k]==event[k] for k in ('task','qid','group_id','path','forced','output_ids','eos_ids','generation_scalars'))
   assert all(started[b+j][k]==rec[k] and completed[b+j][k]==rec[k] for k in ('task','qid','path','forced'))
   ids=rec['output_ids'];assert ids and tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==rec['answer'] and (ids==[forced_id]*33 if forced else (len(ids)<=2048 and (ids[-1] in rec['eos_ids'] or len(ids)==2048)))
   fpath=Path(rec['arrays']['path']).resolve();assert fpath.parent==(R/'arrays').resolve() and sha(fpath)==rec['arrays']['sha256']==formal['arrays_sha256'][str(fpath)];array_sha[str(fpath)]=sha(fpath)
   shape=rec['processor']['input_ids']['shape'];full=shape[1];n=256 if path=='local_stacktok' else 1024;physical=full-1024+n;trace=rec['decode_trace'];assert len(trace)==len(ids) and trace[0]['position_ids']==rec['prefill_position_ids']
   delta=rec['rope_delta'][0][0]
   for t,x in enumerate(trace):
    expected_before=physical+t-1;expected_after=physical+t
    assert x['rope_delta']==[[delta]] and x['layer_cache_lengths_after']==[expected_after]*36
    if t:assert x['cache_position']==[expected_before] and x['position_ids']==[[[full+t-1+delta]]]*3 and x['layer_cache_lengths']==[expected_before]*36
   fp=rec['cache_fingerprints'];assert len(fp)==(33 if forced else 1)
   for t,layer in enumerate(fp):
    assert len(layer)==36
    for kv in layer:
     for v in kv.values():assert v['dtype']=='torch.bfloat16' and len(v['shape'])==4 and v['shape'][-2]==physical+t and len(v['sha256'])==64
  assert all(group[0][k]==group[1][k] for k in keys) and group[1]['spatial_ids']==list(range(1024));assert pairs[z*2+fi]=={'task':anchor['task'],'qid':anchor['qid'],'forced':forced,'status':'passed'}
  with np.load(group[0]['arrays']['path'],allow_pickle=False) as n:
   native=tensor(n,'prefill',group[0]['prefill_embeds']);native_ids=tensor(n,'processor_input_ids',group[0]['processor']['input_ids']);native_mask=tensor(n,'processor_attention_mask',group[0]['processor']['attention_mask']);positions=torch.tensor(group[0]['prefill_position_ids']);visualmask=torch.tensor(group[0]['injected_mask']);where=torch.nonzero(visualmask[0]).flatten();assert len(where)==1024 and torch.equal(where,torch.arange(where[0],where[0]+1024));start=int(where[0]);full=native_ids.shape[1]
   for rec in group[1:]:
    idx=rec['spatial_ids'];k=rec['budget'];assert len(idx)==k and idx==sorted(set(idx)) and all(0<=i<1024 for i in idx)
    if k==256:assert len(rec['greedy_ids'])==256 and sorted(rec['greedy_ids'])==idx and rec['selector_repeat_exact'] is True
    retained=torch.tensor(list(range(start))+[start+i for i in idx]+list(range(start+1024,full)))
    with np.load(rec['arrays']['path'],allow_pickle=False) as v:
     checks={'retained_indices':rec['retained_indices'],'prefill':rec['prefill_embeds'],'prefill_mask':rec['prefill_mask'],'prefill_positions':rec['prefill_positions'],'visual_mask':rec['visual_mask']}
     got={k:tensor(v,k,m) for k,m in checks.items()};assert torch.equal(got['retained_indices'],retained) and torch.equal(got['prefill'],native.index_select(1,retained)) and torch.equal(got['prefill_mask'],native_mask.index_select(1,retained)) and torch.equal(got['prefill_positions'],positions.index_select(2,retained)) and torch.equal(got['visual_mask'],visualmask.index_select(1,retained))
     for j in range(3):
      orig=tensor(n,'ds'+str(j),group[0]['full_deepstack'][j]);whole=tensor(v,'ds'+str(j),rec['full_deepstack'][j]);injected=tensor(v,'injected'+str(j),rec['injected_deepstack'][j]);assert torch.equal(orig,whole) and torch.equal(injected,orig[idx])
     pre=tensor(v,'premerger',rec['premerger']);avg=tensor(v,'averaged_premerger',rec['averaged_premerger']);assert torch.equal(pre.reshape(-1,4,pre.shape[-1]).mean(1),avg)
  summaries.append({'task':anchor['task'],'qid':anchor['qid'],'forced':forced,'stock_tokens':len(group[0]['output_ids']),'stacktok_tokens':len(group[2]['output_ids']),'K':256})
assert len(array_sha)==18 and not torch.cuda.is_initialized()
value={'status':'passed','independent_review':True,'images':3,'generations':18,'keepall_pairs':6,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(f),'formal_started_sha256':sha(D/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'array_sha256':array_sha,'container':container,'summaries':summaries,'limitations':['local Qwen3 adapter around the author selector, not the author Qwen3 integration','selector repeat-exact is a runtime check; the GPU intermediate selector scores were not archived','36-layer KV is represented by runtime fingerprint hashes and shapes, not archived tensor bytes','3 DEV anchors provide diagnostic evidence only; no scoring or quality conclusion'],'errors':[]};a.out.write_text(json.dumps(value,indent=2)+'\n');print('passed',flush=True)
