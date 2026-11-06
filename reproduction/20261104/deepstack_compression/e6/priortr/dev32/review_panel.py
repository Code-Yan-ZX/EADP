"""Independent CPU readback of a complete PriorTR DEV32 diagnostic."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PREP=HERE.parent/'preparation'
ARMS=('stock_native','author_default','priortr_k256')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('partial JSONL')
 return [json.loads(x) for x in b.splitlines()]
def req(value,msg):
 if not value:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser()
 for k in ('run','formal','formal-started','formal-terminal','run-started','run-terminal','out'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();run=a.run.resolve();proof=read(a.formal)
 req(proof['status']=='passed' and proof['questions']==32 and proof['generations']==96 and proof['GT_opened'] is False and proof['scoring_performed'] is False,'formal proof')
 for path,digest in {**proof['files_sha256'],**proof['array_sha256']}.items():req(sha(path)==digest,'accepted artifact changed')
 for start,end,script,out in ((a.formal_started,a.formal_terminal,HERE/'panel_verify_v2.py',a.formal),(a.run_started,a.run_terminal,HERE/'launch.py',None)):
  s=read(start);t=read(end);cwd=Path(s['cwd']);cmd=s['command']
  req(t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and t['worker_pid']==s['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()],'supervisor/command')
  if out is not None:req(cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve(),'formal output')
 req(proof['verifier_sha256']==sha(HERE/'panel_verify_v2.py'),'verifier source')
 exit=read(run/'exit.json');req(exit['returncode']==0 and exit['container_exit_code']==0 and exit['timed_out'] is False and exit['interrupted'] is None and exit['container_stopped_verified'] is True and exit['gpu_pids_after']==[] and exit['failure'] is None,'launcher exit')
 obj=json.loads(subprocess.check_output(['docker','inspect',exit['container_identity']['container_id']],timeout=20))[0]
 req(obj['Id']==exit['container_identity']['container_id'] and not obj['State']['Running'] and obj['State']['Pid']==0 and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled'],'actual container exit')
 frozen=read(run/'frozen.json');req(frozen['GT_opened'] is False and frozen['scoring_performed'] is False and all(sha(p)==h for p,h in frozen['source_sha256'].items()),'frozen code')
 sample=read(HERE/'sample_001.json')['sample'];records=lines(run/'data/records.jsonl');events=lines(run/'data/generated.jsonl');req(len(sample)==32 and len(records)==96 and len(events)==96,'denominator')
 e1path=BASE/'e1/e1full001_completion.json';req(frozen['upstream']['e1']==sha(e1path),'E1 proof');e1=read(e1path)
 for name in ('inputs','journal'):
  x=e1['files'][name];req(sha(x['path'])==x['sha256'],'E1 '+name)
 ref={(r['task'],str(r['qid'])):r for r in lines(e1['files']['journal']['path']) if r.get('arm')=='P3-D1' and r.get('forced') is False};req(len(ref)==384,'E1 references')
 plan=read(BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');tok=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 anchors={next((r['task'],str(r['qid'])) for r in sample if r['task']==t) for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')};arrays_seen=set();stock={};checks=0
 for ai,arm in enumerate(ARMS):
  for i,row in enumerate(sample):
   r=records[32*ai+i];event=events[32*ai+i];key=(row['task'],str(row['qid']))
   for field in ('task','qid','group_id','arm','output_ids','generation_scalars','forced'):req(event[field]==r[field],'journal '+field);checks+=1
   req((r['task'],str(r['qid']),r['group_id'],r['arm'])==(row['task'],str(row['qid']),row['group_id'],arm) and r['processor']==ref[key]['processor'] and r['generation_scalars']==ref[key]['generation_scalars'] and r['forced'] is False and r['backend']=='sdpa' and r['precision']=='bfloat16','input/arm')
   ids=r['output_ids'];end=bool(ids and ids[-1] in r['eos_ids']);req(ids and len(ids)<=2048 and r['eos']==end and r['truncated']==(len(ids)==2048 and not end) and (end or r['truncated']) and tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'],'natural decode')
   k=1024 if arm=='stock_native' else 113 if arm=='author_default' else 256;t=r['trace'];seq=t['calls'][0]['tokens'];length=seq-(1024-k)
   req([x['tokens'] for x in t['layers'] if x['step']==0]==[seq]*3+[length]*33 and len(t['calls'])==len(ids) and t['calls'][0]['cache_after']==[length]*36,'actual layer/cache lengths')
   for j,c in enumerate(t['calls'][1:]):req(c['cache_before']==[length+j]*36 and c['cache_after']==[length+j+1]*36,'decode cache');checks+=72
   req(len(t['injections'])==3 and all(z['rows']==(1024 if j<2 else k) and z['mask_count']==(1024 if j<2 else k) for j,z in enumerate(t['injections'])),'injection counts')
   if arm!='stock_native':
    z=t['prunes'][0];start,end=z['range_before'];small=r['small_diagnostics'];chosen=np.asarray(small['selected_ids_0']);scores=np.asarray(small['selection_scores_0']);req(len(t['prunes'])==1 and z['layer']==2 and end-start==1024 and z['range_after']==[start,start+k] and len(chosen)==k and len(set(chosen.tolist()))==k and np.all(chosen>=0) and np.all(chosen<1024) and np.all(np.diff(chosen)>=0) and scores.shape==(1024,) and np.isfinite(scores).all(),'selection')
    others=np.ones(1024,dtype=bool);others[chosen]=False;req(not others.any() or scores[chosen].min()>=scores[others].max(),'topk')
    keep=np.concatenate((np.arange(start),start+chosen,np.arange(end,seq)));before=np.asarray(small['prune_before_position_0']);after=np.asarray(small['prune_after_position_0']);req(np.array_equal(before[:,:,keep],after),'P3 gathering')
   else:stock[key]=r
   arc=r['arrays'];req(arc['kind']==('full_arrays' if key in anchors else 'summary_only'),'retention policy')
   if key in anchors:
    path=Path(arc['path']).resolve();req(path.parent==(run/'data/arrays').resolve() and path not in arrays_seen and sha(path)==arc['sha256'],'anchor path');arrays_seen.add(path)
    with np.load(path,allow_pickle=False) as z:
     req(set(z.files)==set(arc['tensors']),'anchor inventory')
     for name,meta in arc['tensors'].items():req(hashlib.sha256(np.ascontiguousarray(z[name]).tobytes()).hexdigest()==meta['sha256'],'anchor tensor '+name);checks+=1
     if arm!='stock_native':
      chosen=np.asarray(r['small_diagnostics']['selected_ids_0']);start,end=t['prunes'][0]['range_before'];keep=np.concatenate((np.arange(start),start+chosen,np.arange(end,seq)))
      req(np.array_equal(np.take(z['prune_before_hidden_0'],keep,axis=1),z['prune_after_hidden_0']),'anchor hidden gather')
      for j in range(3):req(np.array_equal(z['injected_ds_'+str(j)],z['full_ds_'+str(j)] if j<2 else z['full_ds_'+str(j)][chosen]),'anchor DS gather');checks+=1
     else:
      for j in range(3):req(np.array_equal(z['injected_ds_'+str(j)],z['full_ds_'+str(j)]),'stock DS')
 req(len(arrays_seen)==9 and arrays_seen=={Path(p).resolve() for p in proof['array_sha256']},'anchor set')
 e4rec=BASE/'e4_panel_v2/panel_003/records.jsonl';req(proof['files_sha256'].get(str(e4rec.resolve()))==sha(e4rec),'E4 reference')
 full={(r['task'],str(r['qid'])):r for r in lines(e4rec) if r['budget']==1024 and r['ds_on'] is True};matches=sum(stock[k]['output_ids']==full[k]['output_ids'] for k in stock)
 req(len(stock)==32 and matches==proof['stock_same_output_as_e4_full'],'E4 output comparison')
 out={'status':'passed','independent_review':True,'questions':32,'generations':96,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(a.formal),'formal_started_sha256':sha(a.formal_started),'formal_terminal_sha256':sha(a.formal_terminal),'reviewer_sha256':sha(__file__),'anchor_arrays_verified':9,'checks':checks,'stock_same_output_as_e4_full':matches,'cross_backend_first_logits':'unavailable for frozen 32 samples','finished_unix':time.time(),'limits':['Diagnostic only, no quality scoring.','Non-anchor high-dimensional arrays are not reconstructible.','Different SDPA/FA2 backends; no latency comparison.']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','questions':32,'records':96,'checks':checks}))
if __name__=='__main__':main()
