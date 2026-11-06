"""Independent 384 template and three image-processor selector reconstruction."""
import argparse,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
B=Path(__file__).resolve().parent.parent;D=B/'e6/ziprerank_qtext';ROOT=B.parents[2];sys.path[:0]=[str(D),str(ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit')]
import torch
from transformers import AutoProcessor,AutoConfig
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
from selector import locate_user_text,user_text_selector
MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();term(D/'monitor_real001');term(D/'monitor_test001');plan=get(D/'plan001.json');result=get(D/'result001.json');assert result['status']=='passed' and result['plan_sha256']==sha(D/'plan001.json')
for f,h in plan['sha256'].items():assert sha(f)==h,f
proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);w=object.__new__(Qwen3VLChatFixedRes);w.system_prompt=None;w.min_pixels=w.max_pixels=1048576;w.total_pixels=None;w.processor=proc;w.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
rows=[json.loads(s) for s in (B/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];record={(x['task'],str(x['qid'])):x for x in result['records']};assert len(record)==len(rows)==384;real=0
for row in rows:
 key=(row['task'],str(row['qid']));r=record.pop(key);messages=w._build_messages(row['message'],dataset=row['task']);text=row['message'][1]['value'];render=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);assert text and render.count(text)==1;start=render.index(text);end=start+len(text);enc=proc.tokenizer(render,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];images=[i for i,x in enumerate(ids) if x==cfg.image_token_id];assert len(images)==1;idx=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>a and a>=start and b<=end];assert idx and all(not (a<end and b>start) or i in idx for i,(a,b) in enumerate(enc['offset_mapping']) if b>a);positions=[i+1023 for i in idx];tokens=[ids[i] for i in idx];h=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest();assert h==r['user_token_sha256'] and len(tokens)==r['user_token_count']
 raw=ids[:images[0]]+[cfg.image_token_id]*1024+ids[images[0]+1:];synthetic=torch.tensor([raw]);assert locate_user_text(proc,messages,text,synthetic,cfg.image_token_id).tolist()==positions
 if r['real_processor_verified']:
  assert sha(row['image_path'])==row['image_sha256'];actual=w._processor_inputs(messages)['input_ids'];assert actual.tolist()==synthetic.tolist() and locate_user_text(proc,messages,text,actual,cfg.image_token_id).tolist()==positions;real+=1
assert not record and real==3 and not torch.cuda.is_initialized()
class M:
 def _extract_query_text_embeds(self,*args,**kwargs):return 'original'
m=M();input_ids=torch.tensor([[1,2,3,4]]);emb=torch.arange(8).reshape(1,4,2);pos=torch.tensor([1,3]);
try:
 with user_text_selector(m,input_ids,pos):
  assert torch.equal(m._extract_query_text_embeds(emb,input_ids),emb[0,pos]);raise RuntimeError('restore')
except RuntimeError:pass
assert '_extract_query_text_embeds' not in m.__dict__ and m._extract_query_text_embeds()=='original'
r={'status':'passed','independent_review':True,'templates':384,'actual_processors':3,'GT_opened':False,'gpu_used':False,'plan_sha256':sha(D/'plan001.json'),'result_sha256':sha(D/'result001.json'),'reviewer_sha256':sha(__file__),'limitations':['full user text includes instructions and choices; not necessarily question-only','local query-selection adaptation, no model inference or quality test','381 examples used synthetic image token expansion'],'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
