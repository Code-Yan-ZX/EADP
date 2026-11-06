"""Independent six-case post-run identity audit; no GT or timing analysis."""
import argparse,hashlib,json,os,struct,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_identity3_execution';T=B/'e6/rres_eadp_timing_pilot_v4';V=B/'e6/rres_eadp_identity3_v2'
ANCHORS=(('DocVQA_VAL','64588'),('TextVQA_VAL','36511'),('AI2D_TEST','207406'))
RECIPE={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def clean(m,script=None,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if cwd.resolve()!=B or s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('supervisor '+str(m))
 if script is not None and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor script')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor out')
 return s
def audit(record,raw,ref,row,arm,task,qid):
 if raw['arm']!=arm or raw['task']!=task or str(raw['qid'])!=qid or record['arm']!=arm or record['task']!=task or str(record['qid'])!=qid:raise ValueError('identity')
 for k in ('group_id','image_sha256'):
  if record[k]!=ref[k] or record[k]!=row[k]:raise ValueError(k)
 full=raw['full_sequence'];out=record['output_ids'];n=record['input_length'];length=n if arm=='rres' else n-768
 if len(full)!=1 or not out or any(type(x)!=int for x in full[0]) or (full[0][n:] if arm=='rres' else full[0])!=out or out!=ref['output_ids'] or record['response']!=ref['answer']:raise ValueError('raw/output/response')
 if arm=='rres':
  pref=full[0][:n];meta={'dtype':'torch.int64','shape':[1,len(pref)],'sha256':hashlib.sha256(struct.pack('<'+'q'*len(pref),*pref)).hexdigest()}
  if meta!=ref['processor']['input_ids']:raise ValueError('prompt prefix')
 if record['processor']!=ref['processor'] or record['visual']!=ref['main_visual'] or record['deepstack']!=ref['deepstack' if arm=='rres' else 'full_deepstack'] or record['injected_deepstack']!=ref['injected_ds' if arm=='rres' else 'injected_deepstack']:raise ValueError('processor/visual/DS')
 if record['prefill_inputs_embeds']!=ref['prefill_inputs_embeds' if arm=='rres' else 'prefill_embeds'] or record['prefill_attention_mask']!=ref['prefill_attention_mask' if arm=='rres' else 'prefill_mask']:raise ValueError('prefill')
 if record['prefill_logits_reference_compared'] is not (arm=='rres') or arm=='rres' and record['prefill_logits']!=ref['prefill_logits']:raise ValueError('logits')
 c=record['visual_execution_counts']
 if c['block_calls']!=ref['block_calls'] or c['merger_calls']!=ref['merger_calls' if arm=='rres' else 'main_merger_calls'] or c['ds_calls']!=ref['ds_calls']:raise ValueError('vision counts')
 pos=record['prefill_position_ids'];flat=[x for a in pos for b in a for x in b];m={'dtype':'torch.int64','shape':[3,1,length],'sha256':hashlib.sha256(struct.pack('<'+'q'*len(flat),*flat)).hexdigest()}
 if len(pos)!=3 or any(len(a)!=1 or len(a[0])!=length for a in pos) or m!=record['prefill_position_meta']:raise ValueError('position bytes')
 delta=record['delta'];mask=record['prefill_visual_mask']
 if arm=='rres':
  if m!=ref['prefill_position_ids'] or delta!=ref['delta'] or mask!=[[row['image_start']<=j<row['image_start']+256 for j in range(n)]]:raise ValueError('rres position')
 else:
  if pos!=ref['prefill_position_ids'] or delta!=ref['adjusted_delta'] or mask!=ref['injected_mask'] or record['ordered_ids']!=ref['ordered_ids'] or len(record['ordered_ids'])!=256:raise ValueError('eadp position/selection')
 trace=record['decode_trace'];d=int(delta[0][0])
 if len(trace)!=len(out) or trace[0]['cache_position']!=list(range(length)) or trace[0]['rope_delta']!=[[d]] or trace[0]['layer_cache_lengths'] not in ([],[0]*36) or trace[0]['layer_cache_lengths_after']!=[length]*36:raise ValueError('prefill KV')
 for j,t in enumerate(trace[1:],1):
  k=length+j-1
  if t['cache_position']!=[k] or t['position_ids']!=[[[k+d]]]*3 or t['rope_delta']!=[[d]] or t['layer_cache_lengths']!=[k]*36 or t['layer_cache_lengths_after']!=[k+1]*36:raise ValueError('decode KV')
 obs=record['generation_observation'];kw=obs['explicit_kwargs'];cfg=obs['effective_generation_config']
 if obs['prepare_calls']!=1 or any(type(kw.get(k)) is not type(v) or type(cfg.get(k)) is not type(v) or kw.get(k)!=v or cfg.get(k)!=v for k,v in RECIPE.items()):raise ValueError('generation configuration')
 eos=cfg['eos_token_id'];eos=[eos] if type(eos) is int else list(eos or [])
 if eos!=ref['eos_ids'] or record['criteria_calls']!=len(out) or record['first_criteria_length']!=(n+1 if arm=='rres' else 1) or record['status']!='passed' or record['errors'] or not all(record['checks'].values()) or record['checks'].get('hooks_and_methods_restored') is not True or record['timing_claims'] is not False:raise ValueError('EOS/hooks/boundary')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();dest=I/'rres_eadp_identity3_gpu_result_review.json'
 if a.out.resolve()!=dest or dest.exists():raise ValueError('output')
 run=D/'run001';formal=D/'verification001.json';fs=clean(D/'monitor_verify001',D/'verify.py',formal);rs=clean(D/'monitor_run001',D/'controller.py');clean(D/'monitor_snapshot001',D/'snapshot.py');verification=read(formal)
 if verification['status']!='passed' or verification['generations']!=6 or verification['images']!=3 or verification['errors'] or verification['GT_opened'] is not False or verification['timing_claims'] is not False:raise ValueError('formal result')
 manifest=D/'run001_sources/manifest.json';snap=read(manifest);plan=read(D/'gpu_plan001.json')
 if sha(manifest)!=verification['source_manifest_sha256'] or set(snap['source_files'])!=set(plan['source_sha256']):raise ValueError('snapshot')
 for k,v in plan['source_sha256'].items():
  item=snap['source_files'][k]
  if sha(k)!=v or item['sha256']!=v or sha(item['copy'])!=v:raise ValueError('source drift')
 for k,v in {**verification['files_sha256'],**verification['run_artifact_sha256']}.items():
  if sha(k)!=v:raise ValueError('artifact hash')
 sys.path[:0]=[str(V),str(T)];import reference_inputs
 refs=reference_inputs.load();cases=[]
 for arm in ('rres','eadp'):
  ad=run/arm;launch=read(ad/'launch.json');ex=read(ad/'exit.json');identity=read(ad/'container_identity.json');cid=identity['container_id'];obj=json.loads(subprocess.run(['docker','inspect',cid],text=True,capture_output=True,check=True,timeout=30).stdout)[0];state=obj['State'];hc=obj['HostConfig'];cfg=obj['Config'];ob=ad/'observations'
  if obj['Id']!=cid or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False or hc['NetworkMode']!='none' or len(hc['Devices'])!=4 or hc.get('DeviceRequests') not in (None,[]) or cfg['Cmd']!=[str(D/'gpu_worker.py'),'--arm',arm,'--out',str(ob)] or launch['gpu_pids_before']!=[] or ex['gpu_pids_after']!=[] or launch['controller']['pid']!=rs['worker_pid'] or ex['container_identity']!=identity:raise ValueError('actual container identity/exit')
  for task,qid in ANCHORS:
   row=refs['rows'][(task,qid)];ref=refs['references'][(arm,task,qid)];p=ob/(task+'_'+qid+'.json');raw=ob/(task+'_'+qid+'_raw.json');audit(read(p),read(raw),ref,row,arm,task,qid);cases.append({'arm':arm,'task':task,'qid':qid,'record_sha256':sha(p),'raw_sha256':sha(raw)})
 if len(cases)!=6 or len(verification['files_sha256'])!=12:raise ValueError('denominator')
 result={'status':'passed','independent_review':True,'images':3,'generations':6,'formal_sha256':sha(formal),'formal_started_sha256':sha(D/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'source_manifest_sha256':sha(manifest),'reviewer_sha256':sha(__file__),'files_sha256':verification['files_sha256'],'cases':cases,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'errors':[],'limits':['Three anchors only; diagnostic hooks change timing.','EADP prior prefill logits are not available.','High-dimensional tensors not archived, identity comparisons use saved fingerprints and actual online checks.','Endpoint hashes/stats and post-run source snapshot do not rule out transient modifications.']}
 with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
