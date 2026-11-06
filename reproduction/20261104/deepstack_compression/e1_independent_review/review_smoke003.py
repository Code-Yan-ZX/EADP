"""Read-only independent legacy/P1 evidence audit; writes only its own report."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np

BASE=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression')
ROOT=Path('/media/disk2/YZX/research/EADP_v2')
MANIFEST=BASE/'data_manifest'
PANEL=ROOT/'reproduction/20261028/qwen/full1024_two_path_panel/plan.json'
PLAN=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
HERE=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True);p.add_argument('--proof',type=Path,required=True)
 p.add_argument('--inputs',type=Path,required=True);p.add_argument('--verifier-terminal',type=Path,required=True)
 p.add_argument('--n',type=int,choices=(1,16),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 errors=[]; checks=0;start=time.time()
 def req(ok,msg):
  nonlocal checks
  checks+=1
  if not ok:errors.append(msg)
 try:
  run=a.run.resolve(); proof=read(a.proof);frozen=read(run/'frozen.json'); records=lines(run/'comparisons.jsonl'); arms=lines(run/'arms.jsonl')
  vt=read(a.verifier_terminal)
  req(proof['status']=='passed' and proof['n_expected']==proof['n_verified']==a.n and not proof['errors'],'E1 verifier status/count')
  req(vt['reason']=='completed' and vt['worker_returncode']==0 and not vt['remaining_process_group'],'E1 verifier true exit')
  for field,path in [('frozen_sha256',run/'frozen.json'),('comparisons_sha256',run/'comparisons.jsonl'),('worker_complete_sha256',run/'worker_complete.json'),('verifier_sha256',BASE/'e1/verify_legacy.py')]:req(proof[field]==sha(path),'proof hash '+field)
  req(frozen['input_sha256']==sha(a.inputs) and frozen['panel_sha256']==sha(MANIFEST/'panel_manifest.jsonl') and frozen['full_manifest_sha256']==sha(MANIFEST/'full_manifest.jsonl'),'input/split hashes')
  old_qids={str(x['index']) for x in read(PANEL)['panel']};full=lines(MANIFEST/'full_manifest.jsonl')
  excluded={x['group_id'] for x in full if x['task']=='DocVQA_VAL' and str(x['qid']) in old_qids}
  req(len(excluded)==16 and sorted(excluded)==frozen['excluded_old_anchor_groups'],'exclude old anchor groups')
  source=lines(a.inputs); candidates=[x for x in source if x['task']=='DocVQA_VAL' and x['group_id'] not in excluded]
  chosen=sorted(candidates,key=lambda x:(hashlib.sha256(('20261104|'+x['group_id']).encode()).hexdigest(),x['group_id']))[:a.n]
  req(frozen['selected_dev']==[{k:r[k] for k in ('task','qid','group_id')} for r in chosen],'deterministic DEV selection')
  dev={(r['task'],str(r['qid']),r['group_id']) for r in lines(MANIFEST/'panel_manifest.jsonl') if r['split']=='DEV'}
  req(len({r['group_id'] for r in chosen})==a.n and all((r['task'],str(r['qid']),r['group_id']) in dev for r in chosen),'DEV-only unique groups')
  req(len(records)==2*a.n and len(arms)==4*a.n,'record count')
  recipe=read(PLAN)['recipe'];model=Path(recipe['model_path']);config=read(model/'config.json');gen=read(model/'generation_config.json')
  eos=gen['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True); forced_id=tokenizer.encode(' the',add_special_tokens=False)[0]
  natural_eos=natural_truncated=0
  for i,row in enumerate(chosen):
   req(sha(row['image_path'])==row['image_sha256'],'image SHA '+str(row['qid']))
   req(hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==row['message_sha256'],'message SHA '+str(row['qid']))
   for k,forced in enumerate((False,True)):
    rec=records[2*i+k];req((str(rec['qid']),rec['group_id'],rec['forced'])==(str(row['qid']),row['group_id'],forced),'qid/condition order')
    vals=[rec['legacy'],rec['p1']]; stores=[]
    for j,(label,v) in enumerate(zip(('legacy','p1'),vals)):
     req(arms[(i*2+k)*2+j]=={'qid':row['qid'],'group_id':row['group_id'],'forced':forced,'arm':label,'observation':v},'arm journal link')
     path=Path(v['array_path']);req(path.parent==run and sha(path)==v['array_sha256'],'archive SHA '+str(row['qid']))
     stores.append(np.load(path,allow_pickle=False))
    try:
     req(set(stores[0].files)==set(stores[1].files),'same array keys')
     for key in stores[0].files:req(np.array_equal(stores[0][key],stores[1][key]),'byte tensor '+key)
     x=vals[0]; ids=np.frombuffer(stores[0]['processor_input_ids'].tobytes(),dtype=np.int64)
     pos=np.flatnonzero(ids==config['image_token_id']);ordered=x['ordered_ids']
     req(len(pos)==1024 and np.array_equal(pos,np.arange(pos[0],pos[0]+1024)),'original image span')
     req(len(ordered)==len(set(ordered))==256 and all(0<=v<1024 for v in ordered),'greedy ids')
     chosen_main=stores[0]['visual_main'][np.sort(np.asarray(ordered))]
     req(np.array_equal(chosen_main,stores[0]['pruned_main']),'reconstruct selected visual')
     req(np.array_equal(chosen_main,stores[0]['prefill_inputs_embeds'][0,int(pos[0]):int(pos[0])+256]),'actual prefill selected visual')
     req(x['output_ids']==vals[1]['output_ids'] and x['answer']==vals[1]['answer'],'generated identity')
     tokens=x['output_ids'][0]
     req(tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)==x['answer'],'independent decode')
     trace=x['llm'];prefill=len(ids)-768
     req(trace[0]['position_ids']==[[list(range(prefill))]]*3 and trace[0]['rope_delta']==[[0]],'P1 prefill position')
     req(trace[0]['visual_pos_masks'] is None and trace[0]['deepstack_visual_embeds'] is False,'legacy/P1 no DS')
     if forced:
      req(tokens==[forced_id]*33 and len(trace)==33,'forced 32 decode')
      for step,t in enumerate(trace[1:],1):
       cp=prefill+step-1
       req(t['cache_position']==[cp] and t['position_ids']==[[[cp]]]*3 and t['rope_delta']==[[0]] and t['layer_cache_lengths']==[cp]*36,'decode and per-layer cache')
     else:
      ended=bool(tokens and tokens[-1] in eos); truncated=bool(len(tokens)>=recipe['max_new_tokens'] and not ended)
      req(ended or truncated,'natural output termination');natural_eos+=int(ended);natural_truncated+=int(truncated)
    finally:
     for z in stores:z.close()
 except Exception as exc:errors.append('review exception: '+repr(exc))
 report={'status':'passed' if not errors else 'failed','n_expected':a.n,'checks':checks,'errors':errors,'natural_eos':natural_eos if 'natural_eos' in locals() else None,'natural_truncated':natural_truncated if 'natural_truncated' in locals() else None,'start_unix':start,'end_unix':time.time(),'run':str(a.run.resolve()),'run_proof_sha256':sha(a.proof),'verifier_terminal_sha256':sha(a.verifier_terminal),'reviewer_sha256':sha(__file__),'scope':'independent read-only CPU verification; no model generation, ground truth or score'}
 with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(report,ensure_ascii=False));return not errors
if __name__=='__main__':raise SystemExit(0 if main() else 1)
