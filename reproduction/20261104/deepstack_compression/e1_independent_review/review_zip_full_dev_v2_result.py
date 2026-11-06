"""Independent reconstruction of all locally adapted Zip DEV observations; no GT."""
import argparse,hashlib,json,struct,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/ziprerank_full_dev_v2';O=D/'run_smoke001/observations';A=B/'e6/ziprerank_full_dev_v2_accept';R=B/'e1_independent_review'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def term(d):
 s=read(d/'started.json');t=read(d/'terminal.json')
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group'] or s['worker_pid']!=t['worker_pid']:raise ValueError('unclean supervisor '+str(d))
 return s,t
def require(v,msg):
 if not v:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 formal=A/'verification001.json';f=read(formal);fs,ft=term(A/'monitor_verify001');run_s,run_t=term(D/'monitor_smoke001');term(R/'monitor_zip_full_v2_prelaunch002')
 require(f['status']=='passed' and f['questions']==384 and f['physical_generations']==768 and f['GT_opened'] is False and f['scoring_performed'] is False,'formal scope')
 require(f['verifier_sha256']==sha(A/'verify.py') and any((Path(fs['cwd'])/x).resolve()==(A/'verify.py').resolve() for x in fs['command'][1:]) and str(formal) in fs['command'],'formal verifier')
 plan=read(D/'plan001.json');launch=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json');done=read(O/'worker_complete.json')
 for path,h in plan['sha256'].items():require(sha(path)==h,'source identity '+path)
 for path,h in f['files_sha256'].items():require(sha(path)==h,'formal artifact '+path)
 require(len(plan['anchors'])==384 and len({(x['task'],str(x['qid'])) for x in plan['anchors']})==384 and plan['arms']==['author_prefix_256','user_text_256'],'panel plan')
 require(launch['plan_sha256']==sha(D/'plan001.json') and run_s['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and launch['controller']==ex['controller'] and ex['returncode']==0 and ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['timed_out'] is False and ex['gpu_pids_after']==[],'run chain')
 inspect=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],check=True,capture_output=True,text=True,timeout=15).stdout)[0];state=inspect['State']
 require(inspect['Id']==ex['last_inspect']['id'] and inspect['Image']==ex['container_identity']['image'] and not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'],'container state')
 require(inspect['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)] and inspect['HostConfig']['NetworkMode']=='none','container command')
 devices=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
 require({(z['PathOnHost'],z['PathInContainer'],z['CgroupPermissions']) for z in inspect['HostConfig']['Devices']}=={(z,z,'rwm') for z in devices},'container devices')
 mounts=[]
 for i,arg in enumerate(launch['command'][:-1]):
  if arg=='--mount':
   vals=dict(x.split('=',1) if '=' in x else (x,True) for x in launch['command'][i+1].split(','));mounts.append((str(Path(vals['src']).resolve()),vals['dst'],'readonly' not in vals))
 require(sorted(mounts)==sorted((str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in inspect['Mounts'] if m['Type']=='bind'),'container mounts')
 e1=read(B/'e1/e1full001_completion.json');journal=Path(e1['files']['journal']['path']);require(sha(journal)==e1['files']['journal']['sha256'],'E1 journal')
 refs={}
 with journal.open() as h:
  for line in h:
   x=json.loads(line)
   if x.get('arm')=='P3-D1' and x.get('forced') is False:
    key=(x['task'],str(x['qid']));require(key not in refs,'E1 duplicate');refs[key]=x
 require(len(refs)==384,'E1 count')
 events=[json.loads(s) for s in (O/'progress.jsonl').read_text().splitlines()];actual=[]
 for x in events:
  event=x['event']
  if event in ('question_started','question_completed'):actual.append((event,x['task'],str(x['qid'])))
  elif event in ('arm_started','arm_recorded'):actual.append((event,x['key'],None))
 expected=[];statuses={};summaries=[];files={}
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 for anchor in plan['anchors']:
  task,qid=anchor['task'],str(anchor['qid']);ref=refs[(task,qid)];require(all(ref[k]==anchor[k] for k in ('task','group_id','message_sha256')),'E1 identity')
  pp=O/(task+'_'+qid+'_processor.json');processor=read(pp);files[str(pp)]=sha(pp);require(processor['processor']==ref['processor'] and processor['image_start']==ref['image_start'] and processor['visual_tokens']==1024,'processor')
  expected.append(('question_started',task,qid));digests=[]
  for arm in plan['arms']:
   key=task+'_'+qid+'_'+arm;expected.extend([('arm_started',key,None),('arm_recorded',key,None)]);rp=O/(key+'.json');gp=O/(key+'_generated.json');row=read(rp);raw=read(gp);files[str(rp)]=sha(rp);files[str(gp)]=sha(gp)
   require(row['status']=='completed' and row['arm']==arm and all(str(row[k])==str(anchor[k]) for k in ('task','qid','group_id')),'arm identity')
   seq=raw['input_sequence_length'];full=raw['sequences'];require(type(seq) is int and len(full)==1 and 0<seq<=len(full[0]),'sequence')
   prompt=full[0][:seq];tok=full[0][seq:];digest=hashlib.sha256(struct.pack('<'+'q'*seq,*prompt)).hexdigest();digests.append(digest)
   require(row['prompt_sha256']==digest and ref['processor']['input_ids']=={'dtype':'torch.int64','shape':[1,seq],'sha256':digest},'E1 prompt')
   ctrl=row['generation_controls'];eos=ctrl.get('eos_token_id');eos=[eos] if isinstance(eos,int) else eos or []
   require(all(ctrl.get(k)==v for k,v in ref['generation_scalars'].items()) and eos==ref['eos_ids'],'generation controls')
   require(row['local_rope_adapter'] is True and row['full_user_text_adapter'] is (arm=='user_text_256') and row['prompt_ids_unchanged'] and row['selector_restored'],'adapter')
   st=row['selection_trace'];require(st['calls']==(1 if arm=='user_text_256' else 0),'selector calls')
   if arm=='user_text_256':require(st['input_ids_sha256']==digest and st['returned_exact_selected_embeddings'] is True and len(st['returned_embedding_sha256'])==64,'selector trace')
   ids=row['selected_indices'];pre=row['prefill'];require(len(ids)==1 and len(ids[0])==256 and len(set(ids[0]))==256 and all(type(i) is int and 0<=i<1024 for i in ids[0]),'selected indices')
   require(pre['selected_count']==256 and pre['selected_unique'] and pre['selected_in_range'] and pre['positions_match'] and pre['visual_count']==256 and pre['ds_rows']==[256]*3,'prefill/DS checks')
   require(tok==row['output_ids'] and 0<len(tok)<=2048 and tokenizer.decode(tok,skip_special_tokens=True,clean_up_tokenization_spaces=False)==row['answer'],'decode')
   ended=tok[-1] in eos;truncated=len(tok)==2048 and not ended;require(row['eos'] is ended and row['truncated'] is truncated and (ended or truncated),'termination')
   calls=row['calls'];initial=seq-768;require(len(calls)==len(tok) and calls[0]['input_length']==initial and calls[0]['cache_length_before']==0,'prefill call')
   require(all(c['input_length']==1 and c['cache_length_before']==initial+i-1 for i,c in enumerate(calls) if i),'decode cache')
   final=row['final_cache'];shape=[1,8,initial+len(tok)-1,128];require(len(final)==36 and all(c['key_shape']==shape and c['value_shape']==shape for c in final),'final cache')
   statuses[key]='completed';summaries.append({'task':task,'qid':qid,'arm':arm,'output_count':len(tok),'eos':ended,'truncated':truncated})
  require(len(set(digests))==1,'arm prompt mismatch');expected.append(('question_completed',task,qid))
 require(expected==actual and done['statuses']==statuses and done['status']=='attempts_recorded' and done['no_GT'] is True and done['scoring'] is False,'complete progress')
 require(summaries==f['per_arm'],'formal per-arm differences')
 result={'status':'passed','independent_review':True,'questions':384,'physical_generations':768,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'formal_sha256':sha(formal),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(D/'monitor_smoke001/terminal.json'),'run_exit_sha256':sha(D/'run_smoke001/exit.json'),'reviewer_sha256':sha(__file__),'files_sha256':files,'e1_journal_sha256':sha(journal),'limitations':['DEV only; both arms use local RoPE repair; user-text selection includes full instructions/options','per-call cache lengths cover global/first-layer and final 36 layer shapes only; KV contents and high-dimensional visual/DS not archived','positions and selected embedding equality are recorded online checks, not independent tensor reconstruction','only six stored E1 generation scalars and EOS checked; other generation settings remain unverified','instrumented generation is not a latency benchmark'],'errors':[]}
 with a.out.open('x') as h:json.dump(result,h,indent=2);h.write('\n')
 print('passed')
if __name__=='__main__':main()
