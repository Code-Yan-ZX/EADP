"""Full no-GT acceptance for 384 x 2 locally adapted ZipRerank DEV outputs."""
import argparse,hashlib,json,os,struct,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent.parent;D=B/'e6/ziprerank_full_dev_v2';O=D/'run_smoke001/observations';R=B/'e1_independent_review'
MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
ARMS=('author_prefix_256','user_text_256')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(path):
 s=read(path/'started.json');t=read(path/'terminal.json')
 if not(t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']):raise ValueError('supervisor exit not clean')
 return s

def validate_item(anchor,reference,processor,row,raw,tokenizer):
 if any(str(row.get(k))!=str(anchor[k]) for k in ('task','qid','group_id')) or row.get('status')!='completed' or row.get('arm') not in ARMS:raise ValueError('record identity/status')
 method=row['arm'];seq=raw['input_sequence_length'];full=raw['sequences']
 if len(full)!=1 or type(seq) is not int or seq>len(full[0]):raise ValueError('raw sequence shape')
 prompt=full[0][:seq];tokens=full[0][seq:];digest=hashlib.sha256(struct.pack('<'+'q'*seq,*prompt)).hexdigest()
 expected=reference['processor'];imeta=expected['input_ids']
 if imeta!={'dtype':'torch.int64','shape':[1,seq],'sha256':digest} or row.get('prompt_sha256')!=digest:raise ValueError('prompt identity')
 if processor.get('processor')!=expected or processor.get('image_start')!=reference['image_start'] or processor.get('visual_tokens')!=1024:raise ValueError('processor mismatch')
 ctrl=row['generation_controls'];scalars=reference['generation_scalars'];eos=ctrl.get('eos_token_id');eos=[eos] if isinstance(eos,int) else eos or []
 if any(ctrl.get(k)!=v for k,v in scalars.items()) or eos!=reference['eos_ids']:raise ValueError('recorded E1 recipe mismatch')
 if row.get('local_rope_adapter') is not True or row.get('full_user_text_adapter') is not (method=='user_text_256') or not row.get('prompt_ids_unchanged') or not row.get('selector_restored'):raise ValueError('local adapter/cleanup mismatch')
 trace=row['selection_trace']
 if trace.get('calls')!=(1 if method=='user_text_256' else 0):raise ValueError('selector call mismatch')
 if method=='user_text_256' and (trace.get('input_ids_sha256')!=digest or trace.get('returned_exact_selected_embeddings') is not True or len(trace.get('returned_embedding_sha256',''))!=64):raise ValueError('user selector input mismatch')
 selected=row['selected_indices'];pre=row['prefill']
 if len(selected)!=1 or len(selected[0])!=256 or len(set(selected[0]))!=256 or any(type(x) is not int or not 0<=x<1024 for x in selected[0]):raise ValueError('invalid selected IDs')
 if not(pre.get('selected_count')==256 and pre.get('selected_unique') is True and pre.get('selected_in_range') is True and pre.get('positions_match') is True and pre.get('visual_count')==256 and pre.get('ds_rows')==[256]*3):raise ValueError('prefill mismatch')
 if row.get('output_ids')!=tokens or not 0<len(tokens)<=2048 or row.get('answer')!=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False):raise ValueError('output/decode mismatch')
 ended=tokens[-1] in eos;truncated=len(tokens)>=2048 and not ended
 if row.get('eos') is not ended or row.get('truncated') is not truncated or not(ended or truncated):raise ValueError('termination mismatch')
 calls=row['calls'];initial=seq-768
 if len(calls)!=len(tokens) or calls[0]['input_length']!=initial or any(x['input_length']!=1 or x['cache_length_before']!=initial+i-1 for i,x in enumerate(calls) if i):raise ValueError('global cache length mismatch')
 final=row['final_cache']
 if len(final)!=36 or any(x.get('key_shape')!=[1,8,initial+len(tokens)-1,128] or x.get('value_shape')!=[1,8,initial+len(tokens)-1,128] for x in final):raise ValueError('final layer cache mismatch')
 return {'task':anchor['task'],'qid':anchor['qid'],'arm':method,'output_count':len(tokens),'eos':ended,'truncated':truncated}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(D/'plan001.json');start=terminal(D/'monitor_smoke001');terminal(R/'monitor_zip_full_v2_prelaunch002')
 launch=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json');done=read(O/'worker_complete.json')
 if len(plan['anchors'])!=384 or plan['arms']!=list(ARMS) or done.get('status')!='attempts_recorded' or done.get('no_GT') is not True or done.get('scoring') is not False:raise ValueError('panel completion scope')
 if launch['plan_sha256']!=sha(D/'plan001.json') or launch['controller']!=ex['controller'] or start['worker_pid']!=launch['controller']['pid'] or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('gpu_pids_after')!=[]:raise ValueError('run exit mismatch')
 for f,h in plan['sha256'].items():
  if sha(f)!=h:raise ValueError('source changed: '+f)
 obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State']
 if obj['Id']!=ex['last_inspect']['id'] or obj['Image']!=ex['container_identity']['image'] or st['Running'] or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled']:raise ValueError('container not clean')
 if obj['Config']['Cmd']!=[str(D/'worker.py'),'--out',str(O)] or obj['HostConfig']['NetworkMode']!='none':raise ValueError('container command mismatch')
 devices=obj['HostConfig']['Devices'];wanted=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
 if {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}!={(x,x,'rwm') for x in wanted}:raise ValueError('GPU devices mismatch')
 proof=read(B/'e1/e1full001_completion.json');journal=Path(proof['files']['journal']['path'])
 if sha(journal)!=proof['files']['journal']['sha256']:raise ValueError('E1 records changed')
 refs={}
 with journal.open() as f:
  for line in f:
   r=json.loads(line)
   if r.get('arm')=='P3-D1' and r.get('forced') is False:
    k=(r['task'],str(r['qid']))
    if k in refs:raise ValueError('duplicate E1 reference')
    refs[k]=r
 if len(refs)!=384:raise ValueError('E1 count')
 events=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];expected=[]
 for anchor in plan['anchors']:
  task=anchor['task'];qid=anchor['qid'];expected.append(('question_started',task,qid))
  for arm in ARMS:expected.extend([('arm_started',task+'_'+qid+'_'+arm,None),('arm_recorded',task+'_'+qid+'_'+arm,None)])
  expected.append(('question_completed',task,qid))
 actual=[]
 for x in events:
  e=x['event']
  if e in ('question_started','question_completed'):actual.append((e,x['task'],str(x['qid'])))
  elif e in ('arm_started','arm_recorded'):actual.append((e,x['key'],None))
 if actual!=expected:raise ValueError('all question/arm event order required')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);files={};summaries=[];statuses={}
 for anchor in plan['anchors']:
  task=anchor['task'];qid=anchor['qid'];ref=refs[(task,qid)]
  if any(ref.get(k)!=anchor[k] for k in ('task','group_id','message_sha256')):raise ValueError('accepted E1 question identity mismatch')
  procpath=O/(task+'_'+qid+'_processor.json');processor=read(procpath);files[str(procpath)]=sha(procpath)
  for arm in ARMS:
   key=task+'_'+qid+'_'+arm;rp=O/(key+'.json');gp=O/(key+'_generated.json');row=read(rp);raw=read(gp)
   summaries.append(validate_item(anchor,ref,processor,row,raw,tokenizer));statuses[key]='completed';files[str(rp)]=sha(rp);files[str(gp)]=sha(gp)
 if done.get('statuses')!=statuses:raise ValueError('arm statuses mismatch')
 for path in [D/'plan001.json',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',O/'worker_complete.json',O/'progress.jsonl']:files[str(path)]=sha(path)
 result={'status':'passed','questions':384,'physical_generations':768,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'per_arm':summaries,'files_sha256':files,'source_sha256':plan['sha256'],'verifier_sha256':sha(__file__),'limitations':['DEV only; local RoPE repair in both arms; full user text includes instructions/options','per-call cache lengths are global/first-layer; full KV contents and high-dimensional visual/DS not archived','instrumented generation is not a latency benchmark','only six stored E1 generation scalars and EOS checked, not all possible generation settings'],'errors':[]}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('passed')
if __name__=='__main__':main()
