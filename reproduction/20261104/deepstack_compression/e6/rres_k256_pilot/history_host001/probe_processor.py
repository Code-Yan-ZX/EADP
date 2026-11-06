"""Author 1024 vs scoped 512 input probe; CPU-only, no model/GT."""
import argparse,hashlib,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def meta(t):
 t=t.detach().contiguous().cpu();return {'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('requires fresh output and no visible CUDA')
 sys.path[:0]=[str(D),str(B/'e4_c_latency'),str(AUTHOR),str(AUTHOR/'VLMEvalKit')]
 import torch
 from transformers import AutoProcessor,AutoConfig
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
 import vlmeval.vlm.qwen3_vl.model_fixed_res as module
 from input_preflight import load_accepted_e1_references,validate_processor
 from resolution import fixed_resolution
 expected=(AUTHOR/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
 if Path(inspect.getfile(module.Qwen3VLChatFixedRes)).resolve()!=expected or module.QWEN3_FIXED_RESOLUTION!=1024 or torch.cuda.is_initialized():raise ValueError('runtime/author identity')
 e1=B/'e1';proof=e1/'e1full001_completion.json';refs=(e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');review=B/'e1_independent_review/e1full001_review.json';rv=json.loads(review.read_text());term=json.loads((B/'e1_independent_review/monitor_e1full001/terminal.json').read_text())
 if rv.get('status')!='passed' or term.get('reason')!='completed' or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[]:raise ValueError('E1 independent proof')
 rows,e1refs=load_accepted_e1_references(proof,*refs);index={(r['task'],str(r['qid'])):r for r in rows}
 design=json.loads((B/'e6/pixel_wavefront_t020_nvml_nine/design001.json').read_text());anchors={}
 for j in design['jobs']:anchors[(j['task'],str(j['qid']))]={k:j[k] for k in ('task','qid','group_id','image_sha256','message_sha256')}
 if len(index)!=384 or len(anchors)!=3 or {k[0] for k in anchors}!={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'}:raise ValueError('anchor inventory')
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);wrapper=object.__new__(module.Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
 if config.vision_config.spatial_merge_size!=2:raise ValueError('merge size')
 results=[]
 for key,identity in anchors.items():
  row=index[key];r=e1refs[key]
  if any(str(row.get(k))!=str(identity[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) or sha(row['image_path'])!=identity['image_sha256']:raise ValueError('input identity')
  messages=wrapper._build_messages(row['message'],dataset=row['task']);original=wrapper._processor_inputs(messages);validate_processor(torch,original,r['processor'],config.image_token_id,r['image_start'])
  with fixed_resolution(module,512):low=wrapper._processor_inputs(messages)
  if module.QWEN3_FIXED_RESOLUTION!=1024 or low['image_grid_thw'].tolist()!=[[1,32,32]] or not bool(torch.all(low['attention_mask']==1)):raise ValueError('low resolution grid or padding')
  def span(v,n):
   ids=v['input_ids'][0];idx=torch.where(ids==config.image_token_id)[0];start=int(idx[0]);
   if idx.numel()!=n or not torch.equal(idx,torch.arange(start,start+n)):raise ValueError('visual span')
   return ids,start
  oldids,oldstart=span(original,1024);newids,start=span(low,256)
  if oldstart!=start or not torch.equal(oldids[:oldstart],newids[:start]) or not torch.equal(oldids[oldstart+1024:],newids[start+256:]):raise ValueError('non-image prompt changed')
  pos,delta=Qwen3VLModel.get_rope_index(SimpleNamespace(config=config),input_ids=low['input_ids'],image_grid_thw=low['image_grid_thw'],attention_mask=low['attention_mask'])
  prefix=torch.arange(start).view(1,-1).expand(3,-1);h=torch.arange(16).view(-1,1).expand(16,16).reshape(-1);w=torch.arange(16).view(1,-1).expand(16,16).reshape(-1);visual=torch.stack((torch.zeros(256,dtype=torch.int64),h,w))+start;suffix_n=int(newids.numel()-start-256);suffix=torch.arange(start+16,start+16+suffix_n).view(1,-1).expand(3,-1);expected_pos=torch.cat((prefix,visual,suffix),dim=1).unsqueeze(1)
  if not torch.equal(pos.cpu(),expected_pos) or int(delta.item())!=int(pos.max().item()+1-newids.numel()):raise ValueError('mRoPE independent reconstruction')
  results.append({**identity,'original_visual_tokens':1024,'low_visual_tokens':256,'grid':low['image_grid_thw'].tolist(),'image_start':start,'text_prefix_suffix_identical':True,'low_processor':{k:meta(low[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'position_ids':meta(pos),'rope_delta':delta.tolist(),'position_formula_verified':True})
  del original,low,pos,delta
 if torch.cuda.is_initialized():raise ValueError('unexpected CUDA')
 value={'status':'passed','anchors':results,'images':3,'resolution':512,'original_resolution':1024,'GT_opened':False,'existing_DEV_predictions_opened':True,'scoring_performed':False,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'e1_formal_sha256':sha(proof),'e1_independent_review_sha256':sha(review),'design_sha256':sha(B/'e6/pixel_wavefront_t020_nvml_nine/design001.json'),'author_source_sha256':sha(expected),'scope':'CPU processor + native mRoPE only; DS/model/GPU not tested','errors':[]}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
