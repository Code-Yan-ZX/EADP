"""Independent re-computation from archived one-image pilot tensors, CPU only."""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
B=Path(__file__).resolve().parent.parent;ROOT=B.parents[2];D=B/'e5/selector_fused_cpu';sys.path.insert(0,str(ROOT/'_reproduction/author_eadp/Qwen_vl'))
import torch
from transformers import AutoTokenizer,AutoConfig
from model.pruner import VisualTokenPruner,_sim_visual_impl,_greed_select_impl
MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();term(D/'monitor001');term(B/'e1_independent_review/monitor_component_pilot001');plan=get(D/'plan001.json');result=get(D/'result001.json');assert result['status']=='completed' and result['plan_sha256']==sha(D/'plan001.json')
for f,h in plan['sha256'].items():assert sha(f)==h,f
artifact=D/'numeric_inputs001.pt';assert sha(artifact)==result['artifact_sha256'];expected=torch.load(artifact,map_location='cpu',weights_only=True);review=get(B/'e1_independent_review/component_pilot001_review.json');assert review['status']=='passed' and sha(B/'e4_c_microbench_pilot/pilot_001/reference.pt')==review['reference_sha256'];act=torch.load(B/'e4_c_microbench_pilot/pilot_001/reference.pt',map_location='cpu',weights_only=True)['actual'];rows=[json.loads(s) for s in (B/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];rows=[r for r in rows if r['task']=='DocVQA_VAL' and str(r['qid'])=='16539'];assert len(rows)==1 and rows[0]['message_sha256']==plan['message_sha256'];text=' '.join(x['value'] for x in rows[0]['message'] if x['type']=='text').strip()
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);instruction=tok(text,return_tensors='pt',add_special_tokens=True,padding=True,truncation=True,max_length=tok.model_max_length).input_ids[0].tolist();assert len(instruction)==result['instruction_tokens'] and hashlib.sha256(json.dumps(instruction).encode()).hexdigest()==result['full_sequence_sha256'];ids=act['processor']['input_ids'][0].tolist();loc=[i for i in range(len(ids)-len(instruction)+1) if ids[i:i+len(instruction)]==instruction];visual=[i for i,x in enumerate(ids) if x==cfg.image_token_id];assert len(loc)==1 and len(visual)==1024 and loc[0]>visual[-1];start=loc[0]-768;stop=start+len(instruction);pre=act['prefill'];assert start>=0 and not bool(pre['visual_pos_masks'][0,start:stop].any());seq=pre['inputs_embeds'][0,start:stop,:].unsqueeze(0);image=act['visual'][0];assert image.shape==(1024,4096) and act['processor']['image_grid_thw'].tolist()==[[1,64,64]]
torch.set_num_threads(4);pr=VisualTokenPruner(visual_token_num=256,alpha=.5,beta=2.,visual_dim=4096,spatial_merge_size=2,device='cpu')
with torch.inference_mode():
 sim=_sim_visual_impl(image.unsqueeze(0));imp=pr.compute_importance(image.unsqueeze(0),seq.mean(dim=1),seq,32,32);order,quota=_greed_select_impl(imp,sim,256)
assert torch.isfinite(sim).all() and torch.isfinite(imp).all() and torch.equal(sim,expected['similarity']) and torch.equal(imp,expected['importance']) and torch.equal(order[0],expected['cpu_ordered']) and torch.equal(expected['gpu_accepted_ordered'],act['ordered'].long()) and torch.equal(order[0],act['ordered'].long()) and quota.tolist()==result['quota'] and result['cpu_vs_gpu_same_order'] is True and result['first_mismatch'] is None and not torch.cuda.is_initialized()
r={'status':'passed','independent_review':True,'image':'DocVQA_VAL/16539','similarity_shape':list(sim.shape),'importance_shape':list(imp.shape),'selection_count':len(order[0]),'cpu_order_matches_recorded_gpu':True,'GT_opened':False,'gpu_used':False,'reviewer_sha256':sha(__file__),'artifact_sha256':sha(artifact),'result_sha256':sha(D/'result001.json'),'limitations':['one image only; CPU reconstruction is not a universal CPU/GPU identity guarantee','historical GPU selector internal similarity/importance were not archived and cannot be directly compared','no performance or model-quality claim'],'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
