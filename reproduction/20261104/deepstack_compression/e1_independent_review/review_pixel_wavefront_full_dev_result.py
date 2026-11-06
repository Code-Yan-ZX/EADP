"""Independent reconstruction of the completed 384-item wavefront comparison; no GT or scoring."""
import argparse,collections,hashlib,json,os,struct,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_wavefront_t020_full_dev';V=B/'e6/pixel_wavefront_t020_full_dev_accept_v2';OLD=B/'e6/pixelprune_t020_dev384';A=B/'e6/pixelprune_t020_accept';I=B/'e1_independent_review';O=D/'run_run001/observations';P=OLD/'run_smoke001/observations';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 assert s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()],str(d)
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s

def validate_record(r,raw,anchor,tokenizer):
 assert all(str(r.get(k))==str(anchor[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) and r['status']=='completed'
 assert r['pixelprune_config']=={'PIXELPRUNE_ENABLED':'true','PIXELPRUNE_METHOD':'pred_2d','PIXELPRUNE_METRIC':'max','PIXELPRUNE_THRESHOLD':'0.2','PIXELPRUNE_ANCHORED':'true','PIXELPRUNE_VIT_LAYER':'0'}
 assert r['generation_kwargs']=={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
 ix=r['selected_indices'];n=r['selected_count'];assert type(n) is int and 1<=n<=1024 and len(ix)==n and ix==sorted(set(ix)) and all(type(i) is int and 0<=i<1024 for i in ix)
 seq=r['sequence_length'];full=raw['full_sequence_ids'];tokens=raw['generated_token_ids'];assert type(seq) is int and raw['prompt_length']==seq and len(full)==seq+len(tokens) and full[seq:]==tokens and all(type(x) is int for x in full)
 inputhash=hashlib.sha256(struct.pack('<'+'q'*seq,*full[:seq])).hexdigest();assert anchor['processor']['input_ids']=={'dtype':'torch.int64','shape':[1,seq],'sha256':inputhash}
 pi=r['processor_identity'];assert pi['processor']==anchor['processor'] and pi['image_start']==anchor['image_start'] and pi['visual_tokens']==1024
 assert len(tokens)==r['generated_count'] and 0<len(tokens)<=2048 and r['ends_eos']==(tokens[-1] in r['eos_token_ids']) and (r['ends_eos'] or len(tokens)==2048)
 assert tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['decoded']
 assert r['visual_rows']==r['prefill_visual_count']==n and r['ds_rows']==r['prefill_ds_rows']==[n]*3 and r['prefill_ds_match']==[True]*3 and all(r[k] is True for k in ('prefill_main_match','prefill_text_match','prefill_positions_match','hook_cleanup_valid'))
 assert len(r['trace'])==len(tokens);initial=seq-1024+n
 for i,t in enumerate(r['trace']):
  assert t['rope_delta']==r['original_rope_delta']==t['model_rope_delta'] and len(t['layers'])==36
  assert all(z['key']==z['value']==[1,8,initial+i,128] for z in t['layers'])
  if i:assert t['position_ids']==[[[r['original_max_position']+i]]]*3 and t['cache_position']==[seq+i-1]
 return ix,tokens

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();assert not args.out.exists()
 formal=V/'verification002.json';f=read(formal);terminal(V/'monitor_verify002',V/'verify.py',formal);terminal(V/'monitor_snapshot002',V/'snapshot.py');start=terminal(D/'monitor_run001',D/'launch.py')
 assert f['status']=='passed' and f['questions']==f['physical_generations']==384 and f['GT_opened'] is False and f['scoring_performed'] is False and f['errors']==[] and f['verifier_sha256']==sha(V/'verify.py')
 plan=read(D/'plan002.json');assert f['plan_sha256']==sha(D/'plan002.json') and f['acceptance_plan_sha256']==sha(V/'acceptance_plan002.json')
 snapshot=V/'snapshot002/manifest.json';m=read(snapshot);assert f['snapshot_manifest_sha256']==sha(snapshot) and m['status']=='passed' and m['plan_sha256']==sha(D/'plan002.json') and set(m['files'])==set(plan['source_sha256'])==set(f['source_sha256'])
 for path,expected in plan['source_sha256'].items():
  assert f['source_sha256'][path]==expected==m['files'][path]['sha256'] and sha(path)==expected and sha(m['files'][path]['path'])==expected,path
 oldformal=A/'verification001.json';oldproof=I/'pixel_t020_dev384_result_review.json';of=read(oldformal);op=read(oldproof);oldreviewer=I/'review_pixel_t020_dev384_result.py';terminal(A/'monitor_verify001',A/'verify.py',oldformal);terminal(I/'monitor_pixel_t020_dev384_result',oldreviewer,oldproof);terminal(OLD/'monitor_smoke001',OLD/'launch.py')
 assert of['status']==op['status']=='passed' and op['independent_review'] is True and of['questions']==op['questions']==384 and op['GT_opened'] is False and op['scoring_performed'] is False and op['errors']==[]
 for k,path in {'formal_sha256':oldformal,'formal_started_sha256':A/'monitor_verify001/started.json','formal_terminal_sha256':A/'monitor_verify001/terminal.json','reviewer_sha256':oldreviewer,'run_started_sha256':OLD/'monitor_smoke001/started.json','run_terminal_sha256':OLD/'monitor_smoke001/terminal.json'}.items():assert op[k]==sha(path),k
 anchors=read(OLD/'plan001.json')['anchors'];keys=[(a['task'],str(a['qid'])) for a in anchors];assert len(keys)==len(set(keys))==384 and collections.Counter(k[0] for k in keys)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
 launch=read(D/'run_run001/launch.json');ex=read(D/'run_run001/exit.json');assert launch['plan_sha256']==sha(D/'plan002.json') and start['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['container_stopped_verified'] is True and ex['gpu_pids_after']==[]
 obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];assert obj['Id']==ex['last_inspect']['id'] and obj['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==st['ExitCode']==0 and not st['OOMKilled'] and obj['HostConfig']['NetworkMode']=='none' and obj['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)]
 dev=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'];assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in obj['HostConfig']['Devices']}=={(x,x,'rwm') for x in dev}
 mounts=[];cmd=launch['command']
 for i,c in enumerate(cmd[:-1]):
  if c=='--mount':
   v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
  if c=='--env':assert cmd[i+1] in obj['Config']['Env']
 assert sorted(mounts)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind')
 for path,digest in f['files_sha256'].items():assert sha(path)==digest,path
 done=read(O/'worker_complete.json');assert done['n_expected']==done['n_recorded']==384 and done['wavefront_selector_restored'] is True and done['GT_opened'] is False and done['scoring_performed'] is False
 events=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];assert [(e['task'],str(e['qid'])) for e in events if e['event']=='arm_started']==keys==[(e['task'],str(e['qid'])) for e in events if e['event']=='arm_recorded'] and all(e.get('status')=='completed' for e in events if e['event']=='arm_recorded')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);comparison=[];files={};eos=0
 for anchor in anchors:
  key=anchor['task']+'_'+str(anchor['qid']);cp=O/(key+'.json');cg=O/(key+'_generated.json');bp=P/(key+'.json');bg=P/(key+'_generated.json')
  for path in (bp,bg):assert of['files_sha256'][str(path)]==op['per_question_files_sha256'][str(path)]==sha(path)
  for path in (cp,cg):assert f['files_sha256'][str(path)]==sha(path);files[str(path)]=sha(path)
  c=read(cp);cr=read(cg);b=read(bp);br=read(bg);ci,ct=validate_record(c,cr,anchor,tok);bi,bt=validate_record(b,br,anchor,tok)
  assert all(c[k]==b[k] for k in ('task','qid','group_id','image_sha256','message_sha256','processor_identity','pixelprune_config','generation_kwargs','eos_token_ids')) and c['wavefront_selector_calls']==1 and all(c['diagnostic_checks'].get(k) is True for k in ('unique_selection','vision_and_three_ds','same_main_text_positions','natural_complete','calls_match_tokens','hooks_removed','positions_and_kv','wavefront_called_once'))
  steps=[e['index'] for e in events if e.get('event')=='forward_complete' and e.get('task')==anchor['task'] and str(e.get('qid'))==str(anchor['qid'])];assert steps==list(range(len(ct)))
  comparison.append({'task':anchor['task'],'qid':anchor['qid'],'group_id':anchor['group_id'],'selection_order_equal':ci==bi,'selection_set_equal':set(ci)==set(bi),'output_tokens_equal':ct==bt,'decoded_equal':c['decoded']==b['decoded'],'eos_termination_equal':c['ends_eos']==b['ends_eos'],'generated_count_equal':len(ct)==len(bt),'baseline_selected_count':len(bi),'candidate_selected_count':len(ci)});eos+=int(c['ends_eos'])
 assert f['comparisons']==comparison
 fields=('selection_order_equal','selection_set_equal','output_tokens_equal','decoded_equal','eos_termination_equal','generated_count_equal');summary={'questions':384,'equal_counts':{x:sum(y[x] for y in comparison) for x in fields},'per_task':{t:{'questions':128,'equal_counts':{x:sum(y[x] for y in comparison if y['task']==t) for x in fields}} for t in sorted(set(k[0] for k in keys))}};assert f['summary']==summary
 result={'status':'passed','independent_review':True,'questions':384,'physical_generations':384,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'formal_sha256':sha(formal),'formal_started_sha256':sha(V/'monitor_verify002/started.json'),'formal_terminal_sha256':sha(V/'monitor_verify002/terminal.json'),'snapshot_manifest_sha256':sha(snapshot),'snapshot_started_sha256':sha(V/'monitor_snapshot002/started.json'),'snapshot_terminal_sha256':sha(V/'monitor_snapshot002/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'run_exit_sha256':sha(D/'run_run001/exit.json'),'prior_formal_sha256':sha(oldformal),'prior_review_sha256':sha(oldproof),'reviewer_sha256':sha(__file__),'files_sha256':files,'summary':summary,'candidate_eos_count':eos,'limitations':['source snapshot taken after GPU run; pre/post source hashes do not exclude transient changes','full visual/DeepStack and KV tensors were not archived; their identity checks are online runtime booleans and KV shapes','existing DEV predictions are parsed for identity and comparison; no GT or scoring','dynamic K, post-hoc exploratory identity diagnostic; not a latency benchmark'],'errors':[]}
 with args.out.open('x') as out:json.dump(result,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
 print('independent 384 review passed')
if __name__=='__main__':main()
