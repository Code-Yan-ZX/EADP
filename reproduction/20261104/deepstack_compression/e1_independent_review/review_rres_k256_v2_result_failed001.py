"""Independent no-GT review of all three real R-res K256 diagnostics."""
import argparse,hashlib,importlib.util,json,os,struct,subprocess
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e6/rres_k256_execution_v2';P=B/'e6/rres_k256_pilot_v2';I=B/'e1_independent_review';ROOT=B.parents[2];O=D/'run001/observations'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor')
 return s
def pos_hash(n,start):
 suffix=n-start-256
 if start<0 or suffix<0:raise ValueError('position shape')
 prefix=list(range(start));after=list(range(start+16,start+16+suffix));axes=[prefix+[start]*256+after,prefix+[start+r for r in range(16) for _ in range(16)]+after,prefix+[start+c for _ in range(16) for c in range(16)]+after];flat=sum(axes,[])
 return hashlib.sha256(struct.pack('<%dq'%len(flat),*flat)).hexdigest(),max(flat)+1-n
def check_record(anchor,row,raw,decode):
 if any(str(row.get(k))!=str(anchor.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')) or row.get('processor')!=anchor['low_processor'] or row.get('resolution')!=512 or row.get('visual_tokens')!=256 or row.get('hooks_and_methods_restored') is not True or row.get('timing_claims') is not False:raise ValueError('runtime input identity or cleanup')
 full=raw.get('full_sequence');n=raw.get('input_length')
 if not isinstance(full,list) or len(full)!=1 or type(n) is not int or len(full[0])<=n or any(type(x) is not int for x in full[0]):raise ValueError('raw sequence')
 info=anchor['low_processor']['input_ids']
 if info!={'dtype':'torch.int64','shape':[1,n],'sha256':hashlib.sha256(struct.pack('<%dq'%n,*full[0][:n])).hexdigest()} or row.get('output_ids')!=full[0][n:]:raise ValueError('input/output token mismatch')
 ids=row['output_ids'];eos=row.get('eos_ids')
 if eos!=[151645,151643] or not 0<len(ids)<=2048 or row.get('eos')!=(ids[-1] in eos) or row.get('truncated')!=(ids[-1] not in eos) or not(row['eos'] or len(ids)==2048) or decode(ids)!=row.get('answer'):raise ValueError('generation termination or independent decode')
 if row.get('generation_scalars')!={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0}:raise ValueError('recipe')
 actual,delta=pos_hash(n,anchor['image_start']);pos=row['prefill_position_ids']
 if pos!={'dtype':'torch.int64','shape':[3,1,n],'sha256':actual} or pos!=anchor['position_ids'] or row['delta']!=[[delta]] or row['delta']!=anchor['rope_delta']:raise ValueError('independent 3-axis mRoPE')
 if row['prefill_attention_mask']!=anchor['low_processor']['attention_mask'] or row['main_visual']['shape'][0]!=256 or len(row['deepstack'])!=3 or row['deepstack']!=row['injected_ds'] or any(x['shape'][0]!=256 for x in row['deepstack']):raise ValueError('main/three DS metadata')
 mask=bytearray(n);mask[anchor['image_start']:anchor['image_start']+256]=bytes([1])*256
 if row['visual_mask']!={'dtype':'torch.bool','shape':[1,n],'sha256':hashlib.sha256(mask).hexdigest()} or row.get('merger_calls')!=1 or row.get('ds_calls')!=[1,1,1] or not row.get('block_calls') or any(x!=1 for x in row['block_calls']):raise ValueError('visual span or hook counts')
 trace=row['decode_trace']
 if len(trace)!=len(ids):raise ValueError('forward count')
 for index,step in enumerate(trace):
  if step.get('rope_delta')!=[[delta]] or step.get('layer_cache_lengths_after')!=[n+index]*36:raise ValueError('KV after/rope')
  if index==0:
   if step.get('cache_position')!=list(range(n)) or step.get('layer_cache_lengths') not in ([],[0]*36):raise ValueError('prefill cache')
  elif step.get('cache_position')!=[n+index-1] or step.get('position_ids')!=[[[n+index-1+delta]]]*3 or step.get('layer_cache_lengths')!=[n+index-1]*36:raise ValueError('decode KV/position')
 return {'task':row['task'],'qid':str(row['qid']),'visual_tokens':256,'generated_count':len(ids),'eos':row['eos'],'truncated':row['truncated']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 if a.out.resolve()!=(I/'rres_k256_v2_result_review.json').resolve():raise ValueError('output path')
 formal=D/'verification001.json';terminal(D/'monitor_verify001',D/'verify.py',formal);terminal(D/'monitor_snapshot001',D/'snapshot.py');started=terminal(D/'monitor_run001',D/'launch.py');f=read(formal);plan=read(D/'gpu_plan001.json');snap=read(D/'source_snapshot001/manifest.json')
 if f.get('status')!='passed' or f.get('images')!=3 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(D/'verify.py') or f.get('plan_sha256')!=sha(D/'gpu_plan001.json') or f.get('source_manifest_sha256')!=sha(D/'source_snapshot001/manifest.json'):raise ValueError('formal identity')
 if snap.get('status')!='passed' or snap.get('plan_sha256')!=sha(D/'gpu_plan001.json') or set(snap['files'])!=set(plan['source_sha256']) or any(sha(k)!=v or sha(snap['files'][k]['path'])!=v or snap['files'][k]['sha256']!=v for k,v in plan['source_sha256'].items()) or any(sha(k)!=v for k,v in f['files_sha256'].items()):raise ValueError('source or formal file drift')
 launch=read(D/'run001/launch.json');ex=read(D/'run001/exit.json');cid=ex['container_identity']['container_id'];obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0]
 if launch['gpu_pids_before']!=[] or started['worker_pid']!=launch['controller']['pid'] or ex['controller']!=launch['controller'] or ex['returncode']!=0 or ex['container_exit_code']!=0 or ex['timed_out'] is not False or ex['container_stopped_verified'] is not True or ex['gpu_pids_after']!=[]:raise ValueError('process exit')
 author_path=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author=read(author_path);loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';sp=importlib.util.spec_from_file_location('_rres_v2_review_loader',loaderpath);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);runtime=m.load_author(plan['source_sha256'])
 cmd=runtime.container_command(author,[str(D/'worker.py'),'--out',str(O)],True,launch['container_name'])
 if cmd.count('--rm')!=1:raise ValueError('fixed command')
 cmd.remove('--rm')
 cfg=obj['Config'];host=obj['HostConfig'];state=obj['State'];image=author['container_image']
 if launch['command']!=cmd or obj['Id']!=cid or obj['Image']!=image or cfg['Image']!=image or obj['Name']!='/'+launch['container_name'] or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False or cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['User']!=cmd[cmd.index('--user')+1] or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1] or cfg['Cmd']!=cmd[cmd.index(image)+1:] or host['NetworkMode']!=cmd[cmd.index('--network')+1]:raise ValueError('actual Docker image/process')
 for value in (cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'):
  if [x for x in cfg['Env'] if x.startswith(value.split('=',1)[0]+'=')]!=[value]:raise ValueError('environment')
 devices=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--device'];actual=host.get('Devices') or []
 if host.get('DeviceRequests') not in (None,[]) or len(actual)!=len(devices) or {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in actual}!={(x,x,'rwm') for x in devices}:raise ValueError('GPU devices')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--mount':
   parts=cmd[i+1].split(',');v=dict(z.split('=',1) for z in parts if '=' in z);mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in parts))
 actual=[(str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind']
 if len(actual)!=len(obj['Mounts']) or sorted(actual)!=sorted(mounts):raise ValueError('mounts')
 probe=read(P/'run_preflight001/processor_probe.json');anchors=probe['anchors'];finished=read(O/'worker_complete.json');events=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];expected=[(x['task'],str(x['qid'])) for x in anchors]
 if len(expected)!=3 or len(set(expected))!=3 or finished.get('status')!='completed' or finished.get('resolution_restored') is not True or finished.get('GT_opened') is not False or finished.get('scoring_performed') is not False or [(x['task'],str(x['qid'])) for x in events if x['event']=='arm_started']!=expected or [(x['task'],str(x['qid'])) for x in events if x['event']=='arm_recorded']!=expected:raise ValueError('complete ordered inventory')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 results=[]
 for anchor in anchors:
  key=anchor['task']+'_'+str(anchor['qid']);record=read(O/(key+'.json'));raw=read(O/(key+'_generated.json'))
  results.append(check_record(anchor,record,raw,lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)))
 if results!=f['results']:raise ValueError('formal per-question mismatch')
 fields={'formal_sha256':formal,'formal_started_sha256':D/'monitor_verify001/started.json','formal_terminal_sha256':D/'monitor_verify001/terminal.json','run_started_sha256':D/'monitor_run001/started.json','run_terminal_sha256':D/'monitor_run001/terminal.json','snapshot_manifest_sha256':D/'source_snapshot001/manifest.json','reviewer_sha256':Path(__file__)}
 value={'status':'passed','independent_review':True,'images':3,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'errors':[],**{k:sha(v) for k,v in fields.items()},'files_sha256':f['files_sha256'],'docker_log_sha256':sha(D/'run001/docker.log'),'results':results,'container_id':cid,'limitations':['three DEV anchors, no quality or equal-compute conclusion','visual/three DS/KV raw tensors not archived; online equality, fingerprints and KV lengths only','hook and finite-logit checks rely on runtime observation','Docker log reports unwritable kernel cache, FlashAttention dtype notice, and sampling parameters ignored under greedy decoding; runtime checked bf16 and flash_attention_2; no timing claim']}
 with a.out.open('x') as dest:json.dump(value,dest,indent=2);dest.write('\n');dest.flush();os.fsync(dest.fileno())
if __name__=='__main__':main()
