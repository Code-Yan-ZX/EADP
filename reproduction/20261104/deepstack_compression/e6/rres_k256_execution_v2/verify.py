"""No-GT formal acceptance of all three 512px model diagnostics."""
import argparse,hashlib,json,os,struct,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/rres_k256_pilot_v2';O=D/'run001/observations'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def supervise(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd'])
 if s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor mismatch')
 return s

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert not a.out.exists()
 plan=read(D/'gpu_plan001.json');s=supervise(D/'monitor_run001',D/'launch.py');supervise(D/'monitor_snapshot001',D/'snapshot.py');m=read(D/'source_snapshot001/manifest.json')
 assert m['status']=='passed' and m['plan_sha256']==sha(D/'gpu_plan001.json') and set(m['files'])==set(plan['source_sha256'])
 for path,digest in plan['source_sha256'].items():assert m['files'][path]['sha256']==sha(path)==sha(m['files'][path]['path'])==digest,path
 launch=read(D/'run001/launch.json');ex=read(D/'run001/exit.json');assert launch['gpu_pids_before']==[] and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and not ex['timed_out'] and ex['container_stopped_verified'] and ex['gpu_pids_after']==[]
 obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];assert obj['Id']==ex['last_inspect']['id'] and obj['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==st['ExitCode']==0 and not st['OOMKilled'] and obj['HostConfig']['NetworkMode']=='none' and obj['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)]
 cmd=launch['command'];mounts=[]
 for i,c in enumerate(cmd[:-1]):
  if c=='--env':assert cmd[i+1] in obj['Config']['Env']
  if c=='--mount':
   v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 assert sorted(mounts)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind')
 dev=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'];assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in obj['HostConfig']['Devices']}=={(x,x,'rwm') for x in dev}
 ref=read(P/'run_preflight001/processor_probe.json');anchors=ref['anchors'];done=read(O/'worker_complete.json');assert ref['status']=='passed' and len(anchors)==3 and done['expected']==3 and len(done['started'])==len(done['recorded'])==3 and done['status']=='completed' and done['resolution_restored'] and done['GT_opened'] is False and done['scoring_performed'] is False
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 events=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];expected=[(x['task'],str(x['qid'])) for x in anchors];assert len(set(expected))==3 and [(x['task'],str(x['qid'])) for x in events if x['event']=='arm_started']==expected==[(x['task'],str(x['qid'])) for x in events if x['event']=='arm_recorded']
 results=[];files={}
 for anchor in anchors:
  name=anchor['task']+'_'+str(anchor['qid']);rp=O/(name+'.json');gp=O/(name+'_generated.json');r=read(rp);raw=read(gp)
  assert all(str(r[k])==str(anchor[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) and r['processor']==anchor['low_processor'] and r['resolution']==512 and r['visual_tokens']==256 and r['hooks_and_methods_restored'] and r['timing_claims'] is False
  full=raw['full_sequence'];length=raw['input_length'];assert len(full)==1 and all(type(x) is int for x in full[0]) and len(full[0])>length and r['output_ids']==full[0][length:]
  assert anchor['low_processor']['input_ids']=={'dtype':'torch.int64','shape':[1,length],'sha256':hashlib.sha256(struct.pack('<'+'q'*length,*full[0][:length])).hexdigest()}
  ids=r['output_ids'];assert r['generation_scalars']=={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0} and r['eos_ids']==[151645,151643] and 0<len(ids)<=2048 and r['eos']==(ids[-1] in r['eos_ids']) and r['truncated'] is (not r['eos']) and (r['eos'] or len(ids)==2048) and tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer']
  assert r['main_visual']['shape'][0]==256 and len(r['deepstack'])==len(r['injected_ds'])==3 and all(x['shape'][0]==256 for x in r['deepstack']) and r['deepstack']==r['injected_ds'] and r['delta']==anchor['rope_delta'] and r['prefill_position_ids']==anchor['position_ids'] and r['prefill_attention_mask']==anchor['low_processor']['attention_mask'] and r['block_calls'] and all(x==1 for x in r['block_calls']) and r['merger_calls']==1 and r['ds_calls']==[1]*3
  mask=bytearray(length);mask[anchor['image_start']:anchor['image_start']+256]=b'\1'*256;assert r['visual_mask']=={'dtype':'torch.bool','shape':[1,length],'sha256':hashlib.sha256(mask).hexdigest()}
  trace=r['decode_trace'];assert len(trace)==len(ids)
  for i,t in enumerate(trace):
   assert t['rope_delta']==r['delta'] and t['layer_cache_lengths_after']==[length+i]*36
   if i==0:assert t['cache_position']==list(range(length)) and t['layer_cache_lengths'] in ([],[0]*36)
   else:assert t['cache_position']==[length+i-1] and t['position_ids']==[[[length+i-1+int(r['delta'][0][0])]]]*3 and t['layer_cache_lengths']==[length+i-1]*36
  files[str(rp)]=sha(rp);files[str(gp)]=sha(gp);results.append({'task':r['task'],'qid':r['qid'],'visual_tokens':256,'generated_count':len(ids),'eos':r['eos'],'truncated':r['truncated']})
 for path in (O/'progress.jsonl',O/'worker_complete.json',D/'run001/launch.json',D/'run001/exit.json'):files[str(path)]=sha(path)
 out={'status':'passed','images':3,'GT_opened':False,'existing_DEV_predictions_opened':True,'scoring_performed':False,'timing_claims':False,'results':results,'files_sha256':files,'source_manifest_sha256':sha(D/'source_snapshot001/manifest.json'),'plan_sha256':sha(D/'gpu_plan001.json'),'verifier_sha256':sha(__file__),'limitations':['only 3 fixed DEV anchors; no quality or equal-compute conclusion','visual/DeepStack/KV payloads not archived; equality was checked online and hashes/lengths retained','post-run source snapshot and endpoint hashes cannot exclude transient modifications'],'errors':[]}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
