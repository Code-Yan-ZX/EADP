"""Independent no-GT acceptance of a complete PixelPrune 384-DEV natural panel.

This checks actual journal, raw tokens and recorded runtime diagnostics. It does
not claim off-line reconstruction of high-dimensional visual/DS/KV tensors.
"""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent.parent.parent;D=B/'e6/pixelprune_dev384';O=D/'run_smoke001/observations';I=B/'e1_independent_review';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d):
 s=read(d/'started.json');t=read(d/'terminal.json')
 if not (t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']):raise ValueError('supervisor not clean')
 return s
def tensor_hash_int64(tokens):
 import torch
 x=torch.tensor([tokens],dtype=torch.long);return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
def validate_item(row,raw,anchor,tokenizer):
 if any(str(row.get(k))!=str(anchor[k]) for k in ('task','qid','group_id','message_sha256','image_sha256')):raise ValueError('input identity mismatch')
 if row.get('status')!='completed':raise ValueError('generation failed')
 n=row.get('selected_count');indices=row.get('selected_indices')
 if type(n) is not int or not 1<=n<=1024 or len(indices)!=n or indices!=sorted(set(indices)) or any(type(i) is not int or not 0<=i<1024 for i in indices):raise ValueError('dynamic selection mismatch')
 if row.get('visual_rows')!=n or row.get('ds_rows')!=[n]*3 or row.get('prefill_ds_rows')!=[n]*3 or row.get('prefill_visual_count')!=n or row.get('prefill_ds_match')!=[True]*3 or any(row.get(k) is not True for k in ('prefill_main_match','prefill_text_match','prefill_positions_match','hook_cleanup_valid')):raise ValueError('runtime prefill or DS mismatch')
 kwargs={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
 if row.get('generation_kwargs')!=kwargs:raise ValueError('E1 generation scalars mismatch')
 identity=row.get('processor_identity',{});expected=anchor['processor']
 if identity.get('processor')!=expected or identity.get('image_start')!=anchor['image_start'] or identity.get('visual_tokens')!=1024:raise ValueError('processor reference mismatch')
 full=raw.get('full_sequence_ids');tokens=raw.get('generated_token_ids');seq=raw.get('prompt_length')
 if type(seq) is not int or seq!=row.get('sequence_length') or not isinstance(full,list) or not isinstance(tokens,list) or full[seq:]!=tokens or tensor_hash_int64(full[:seq])!=expected['input_ids']:raise ValueError('raw full token/prompt mismatch')
 if not 0<len(tokens)<=2048 or len(tokens)!=row.get('generated_count') or row.get('ends_eos') is not (tokens[-1] in row['eos_token_ids']) or not (row['ends_eos'] or len(tokens)==2048):raise ValueError('natural output termination mismatch')
 if tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=row.get('decoded'):raise ValueError('independent token decode mismatch')
 trace=row['trace'];initial=seq-1024+n;delta=row['original_rope_delta'];maxpos=row['original_max_position']
 if len(trace)!=len(tokens):raise ValueError('forward count mismatch')
 for i,t in enumerate(trace):
  if t.get('rope_delta')!=delta or t.get('model_rope_delta')!=delta or len(t.get('layers',[]))!=36:raise ValueError('delta/layer mismatch')
  if any(z.get('key')!=[1,8,initial+i,128] or z.get('value')!=[1,8,initial+i,128] for z in t['layers']):raise ValueError('KV shape mismatch')
  if i and (t.get('position_ids')!=[[[maxpos+i]]]*3 or t.get('cache_position')!=[seq+i-1]):raise ValueError('decode position mismatch')
 return {'task':anchor['task'],'qid':anchor['qid'],'selected_count':n,'generated_count':len(tokens),'ends_eos':row['ends_eos']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();plan=read(D/'plan001.json');start=terminal(D/'monitor_smoke001');terminal(I/'monitor_pixel_dev384_prelaunch');launch=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json');done=read(O/'worker_complete.json')
 if len(plan['anchors'])!=384 or done.get('n_expected')!=384 or done.get('n_recorded')!=384 or done.get('GT_opened') is not False or done.get('scoring_performed') is not False:raise ValueError('full inventory incomplete')
 if launch['plan_sha256']!=sha(D/'plan001.json') or start['worker_pid']!=launch['controller']['pid'] or ex['controller']!=launch['controller'] or ex['returncode']!=0 or ex['container_exit_code']!=0 or ex.get('timed_out') is not False or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[]:raise ValueError('run exit mismatch')
 for f,h in plan['sha256'].items():
  if sha(f)!=h:raise ValueError('source changed: '+f)
 obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State']
 if obj['Id']!=ex['last_inspect']['id'] or obj['Image']!=ex['container_identity']['image'] or st['Running'] or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled']:raise ValueError('container not clean')
 if obj['Config']['Cmd']!=[str(D/'worker.py'),'--out',str(O)] or obj['HostConfig']['NetworkMode']!='none':raise ValueError('container process mismatch')
 devices=obj['HostConfig']['Devices'];expected_devices=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
 if {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}!={(x,x,'rwm') for x in expected_devices}:raise ValueError('GPU devices mismatch')
 cmd=launch['command'];mount=[]
 for i,c in enumerate(cmd[:-1]):
  if c=='--mount':
   v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
  if c=='--env' and cmd[i+1] not in obj['Config']['Env']:raise ValueError('container environment mismatch')
 if sorted(mount)!=sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind'):raise ValueError('container mounts mismatch')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);events=[json.loads(line) for line in (O/'progress.jsonl').read_text().splitlines()];started=[(x['task'],str(x['qid'])) for x in events if x['event']=='arm_started'];recorded=[(x['task'],str(x['qid'])) for x in events if x['event']=='arm_recorded'];expected=[(x['task'],str(x['qid'])) for x in plan['anchors']]
 if started!=expected or recorded!=expected or len(set(expected))!=384:raise ValueError('journal denominator/order mismatch')
 files={};summary=[]
 for anchor in plan['anchors']:
  key=anchor['task']+'_'+anchor['qid'];rowpath=O/(key+'.json');rawpath=O/(key+'_generated.json');row=read(rowpath);raw=read(rawpath);summary.append(validate_item(row,raw,anchor,tokenizer));files[str(rowpath)]=sha(rowpath);files[str(rawpath)]=sha(rawpath)
  steps=[x['index'] for x in events if x.get('event')=='forward_complete' and x.get('task')==anchor['task'] and str(x.get('qid'))==anchor['qid']]
  if steps!=list(range(row['generated_count'])):raise ValueError('forward journal mismatch')
 files.update({str(x):sha(x) for x in [D/'plan001.json',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',O/'worker_complete.json',O/'progress.jsonl']});out={'status':'passed','questions':384,'physical_generations':384,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'per_question':summary,'files_sha256':files,'source_sha256':plan['sha256'],'verifier_sha256':sha(__file__),'limitations':['dynamic K depends on input; not an equal-K256 comparison','runtime prefill/DS checks are recorded booleans, not archived arrays','KV after lengths/positions recorded; full KV contents and per-decode KV-before are not independently reconstructable','instrumented generation is not a latency benchmark','accepted E1 DEV journal parsed for input identity'],'errors':[]}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('passed')
if __name__=='__main__':main()
