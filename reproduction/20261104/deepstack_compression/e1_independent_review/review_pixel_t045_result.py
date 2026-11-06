"""Independent review of four posthoc PixelPrune threshold-0.45 natural calls."""
import argparse,hashlib,json,os,struct,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/pixelprune_t045_smoke';O=D/'run_smoke001/observations'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def req(x,m):
 if not x:raise ValueError(m)
def terminal(p):
 s=read(p/'started.json');t=read(p/'terminal.json');req(t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'],'supervisor '+str(p));return s,t
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(D/'plan001.json');pre=read(D/'independent_prelaunch_review.json');s,t=terminal(D/'monitor_smoke001');ps,pt=terminal(R/'monitor_pixel_t045_prelaunch');terminal(D/'monitor_preflight001');launch=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json')
 for path,h in plan['sha256'].items():req(sha(path)==h,'source '+path)
 req(pre['status']=='passed' and pre['independent_review'] is True and pre['plan_sha256']==sha(D/'plan001.json') and pre['reviewer_sha256']==sha(pre['reviewer_path']) and pre['review_started_sha256']==sha(R/'monitor_pixel_t045_prelaunch/started.json'),'prelaunch')
 req(plan['threshold']==.45 and plan['method']=='pred_2d' and plan['metric']=='max' and plan['anchored'] is True and plan['vit_layer']==0 and len(plan['anchors'])==4,'plan')
 req(launch['plan_sha256']==sha(D/'plan001.json') and s['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and launch['controller']==ex['controller'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['timed_out'] is False and ex['client_exited_while_container_running'] is False and ex['gpu_pids_after']==[],'run exit')
 ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=ins['State'];req(ins['Id']==ex['container_identity']['container_id'] and ins['Image']==ex['container_identity']['image'] and not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'],'container')
 req(ins['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)] and ins['HostConfig']['NetworkMode']=='none','container command')
 dev=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'];req({(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in ins['HostConfig']['Devices']}=={(x,x,'rwm') for x in dev},'devices')
 mounts=[]
 for i,c in enumerate(launch['command'][:-1]):
  if c=='--mount':
   v=dict(z.split('=',1) if '=' in z else (z,True) for z in launch['command'][i+1].split(','));mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 req(sorted(mounts)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind'),'mounts')
 cal=read(R/'pixel_threshold32_review.json');terminal(R/'monitor_pixel_threshold32');req(cal['status']=='passed' and cal['independent_review'] and sha(R/'pixel_threshold32_review.json')==plan['calibration_review_sha256'],'calibration')
 done=read(O/'worker_complete.json');req(done['status']=='diagnostic_completed' and done['n_expected']==done['n_recorded']==4 and done['GT_opened'] is False and done['scoring_performed'] is False,'completion')
 progress=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];files={};observed=[];statuses={}
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 for anchor in plan['anchors']:
  task,qid=anchor['task'],str(anchor['qid']);key=task+'_'+qid;rp=O/(key+'.json');gp=O/(key+'_generated.json');r=read(rp);raw=read(gp);files[str(rp)]=sha(rp);files[str(gp)]=sha(gp)
  req(all(str(r[x])==str(anchor[x]) for x in ('task','qid','group_id','message_sha256','image_sha256')) and r['status']=='completed','record identity');req(r['generation_kwargs']==plan['generation'] and r['threshold']==.45 and r['anchored'] is True,'generation/settings')
  n=r['selected_count'];idx=r['selected_indices'];req(type(n) is int and 1<=n<=1024 and len(idx)==n and idx==sorted(set(idx)) and all(type(x) is int and 0<=x<1024 for x in idx),'selected bounds')
  cpath=Path(anchor['cpu_journal']);req(cal['image_files_sha256'].get(str(cpath))==sha(cpath),'calibration source');cpu=[json.loads(x) for x in cpath.read_text().splitlines() if x.strip()];entry=[x for x in cpu if x.get('event')=='completed' and x.get('threshold')==.45];req(len(entry)==1 and entry[0]['selected_indices']==anchor['cpu_selected_indices']==r['cpu_expected_indices']==idx and entry[0]['selected_count']==anchor['cpu_K']==r['cpu_expected_K']==n and r['cpu_gpu_selection_equal'] is True,'CPU GPU indices')
  seq=r['sequence_length'];allids=raw['full_sequence_ids'];tokens=raw['generated_token_ids'];req(raw['prompt_length']==seq and allids[seq:]==tokens and len(tokens)==r['generated_count'] and 0<len(tokens)<=2048,'output sequence');digest=hashlib.sha256(struct.pack('<'+'q'*seq,*allids[:seq])).hexdigest();req(anchor['processor']['input_ids']=={'dtype':'torch.int64','shape':[1,seq],'sha256':digest},'prompt hash');req(r['processor_identity']['processor']==anchor['processor'] and r['processor_identity']['image_start']==anchor['image_start'] and r['processor_identity']['visual_tokens']==1024,'processor identity')
  req((tokens[-1] in r['eos_token_ids'] or len(tokens)==2048) and r['ends_eos'] is (tokens[-1] in r['eos_token_ids']) and tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['decoded'],'decode and EOS')
  req(r['visual_rows']==r['prefill_visual_count']==n and r['ds_rows']==r['prefill_ds_rows']==[n]*3 and all(r[x] is True for x in ('prefill_main_match','prefill_text_match','prefill_positions_match','hook_cleanup_valid')) and r['prefill_ds_match']==[True]*3,'online prefill/cleanup')
  trace=r['trace'];req(len(trace)==len(tokens),'forward count');initial=seq-1024+n
  for i,z in enumerate(trace):
   shape=[1,8,initial+i,128];req(z['rope_delta']==r['original_rope_delta']==z['model_rope_delta'] and len(z['layers'])==36 and all(k['key']==shape and k['value']==shape for k in z['layers']),'rope and KV shape')
   if i:req(z['position_ids']==[[[r['original_max_position']+i]]]*3 and z['cache_position']==[seq+i-1],'decode position')
  req(all(r['diagnostic_checks'].values()) and [e['index'] for e in progress if e.get('event')=='forward_complete' and e.get('task')==task and str(e.get('qid'))==qid]==list(range(len(trace))),'forward journal')
  req(len([e for e in progress if e.get('event')=='arm_started' and e.get('task')==task and str(e.get('qid'))==qid])==1 and len([e for e in progress if e.get('event')=='arm_recorded' and e.get('task')==task and str(e.get('qid'))==qid and e.get('status')=='completed'])==1,'arm events')
  statuses[task+'/'+qid]='completed';observed.append({'task':task,'qid':qid,'visual_tokens':n,'generated_tokens':len(tokens),'eos':r['ends_eos'],'cpu_gpu_indices_equal':True})
 req(done['arm_statuses']==statuses and any(x['visual_tokens']==1 for x in observed),'all attempts including K1')
 for pth in (O/'progress.jsonl',O/'worker_complete.json',D/'plan001.json',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',D/'independent_prelaunch_review.json'):files[str(pth)]=sha(pth)
 out={'status':'passed','independent_review':True,'images':4,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'observed':observed,'files_sha256':files,'reviewer_sha256':sha(__file__),'limitations':['posthoc selected DEV sample, not representative or an accuracy experiment','CPU/GPU exact selected ID equality is established for these four cases only','full visual, DeepStack, position tensors and KV content not archived; actual tensor equalities are online checks and per-layer verification uses cache shapes','no latency or equal-compute claim'],'errors':[]}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 print('four cases independently reviewed')
if __name__=='__main__':main()
