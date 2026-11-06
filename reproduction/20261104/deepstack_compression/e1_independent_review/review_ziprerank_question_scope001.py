"""Independent token-span reconstruction; CPU only, no answers or weights."""
import hashlib,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=Path(__file__).resolve().parents[1];V=R.parents[2];H=R/'e6/ziprerank_question_scope';I=R/'e1_independent_review';sys.path.insert(0,str(V/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit'))
import torch,transformers
from transformers import AutoProcessor,AutoConfig
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
p=read(H/'plan001.json');res=read(H/'result001.json');s=read(H/'monitor001/started.json');t=read(H/'monitor001/terminal.json')
assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and s['cwd']==str(R) and s['command'][1:]==['e6/ziprerank_question_scope/probe.py']
assert all(sha(k)==v for k,v in p['sources'].items()) and res['plan_sha256']==sha(H/'plan001.json') and res['status']=='completed' and res['cuda_initialized'] is False
assert torch.__version__==res['torch'] and transformers.__version__==res['transformers']
model=next(Path(k).parent for k in p['sources'] if k.endswith('/config.json'));proc=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False);cfg=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
w=object.__new__(Qwen3VLChatFixedRes);w.system_prompt=None;w.min_pixels=w.max_pixels=1048576;w.total_pixels=None;w.processor=proc;w.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
rows=[json.loads(x) for x in (R/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];records=res['records'];assert len(rows)==len(records)==384
anchors={(x['task'],str(x['qid'])):x for x in p['processor_anchors']};assert len(anchors)==3;seen=set()
for row,rec in zip(rows,records):
 key=(row['task'],str(row['qid']));assert key not in seen;seen.add(key);assert rec['task']==key[0] and rec['qid']==key[1] and rec['message_sha256']==row['message_sha256']==hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
 text=row['message'][1]['value'];assert [x['type'] for x in row['message']]==['image','text'];messages=w._build_messages(row['message'],dataset=row['task']);rendered=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);assert rendered.count(text)==1
 start=rendered.index(text);end=start+len(text);enc=proc.tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offset=enc['offset_mapping'];selected=[i for i,(a,b) in enumerate(offset) if a<b and a<end and b>start]
 assert selected and selected==list(range(selected[0],selected[-1]+1)) and all(start<=offset[i][0]<offset[i][1]<=end for i in selected)
 assert len([i for i,v in enumerate(ids) if v==cfg.image_token_id])==1 and selected[0]>ids.index(cfg.image_token_id)
 assert rec['question_token_count']==len(selected) and rec['question_ids_sha256']==hashlib.sha256(json.dumps([ids[i] for i in selected],separators=(',',':')).encode()).hexdigest() and rec['question_is_contiguous_and_after_image'] and rec['boundaries_do_not_cross_template']
 if key in anchors:
  anchor=anchors[key];assert sha(row['image_path'])==row['image_sha256']==anchor['image_sha256'] and row['message_sha256']==anchor['message_sha256'];inp=w._processor_inputs(messages);actual=inp['input_ids'][0].tolist();vis=[i for i,v in enumerate(actual) if v==cfg.image_token_id];assert len(vis)==1024
  mapped=[i+len(vis)-1 for i in selected];assert [actual[i] for i in mapped]==[ids[i] for i in selected] and rec['real_processor_verified'] is True and rec['expanded_question_start']==mapped[0] and rec['expanded_question_end_exclusive']==mapped[-1]+1 and rec['image_tokens']==len(vis)
 else:assert rec['real_processor_verified'] is False
assert not torch.cuda.is_initialized();out=I/'ziprerank_question_scope001_review.json';proof={'status':'passed','independent_review':True,'tokenizer_prompts_verified':384,'real_processor_images_verified':3,'plan_sha256':sha(H/'plan001.json'),'result_sha256':sha(H/'result001.json'),'started_sha256':sha(H/'monitor001/started.json'),'terminal_sha256':sha(H/'monitor001/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'cuda_initialized':False,'limitations':['Full message text is selected, including answer-format instructions and any multiple-choice options; not strictly the question stem.','Only 3 image processor inputs were checked; mapping feasibility, not quality or original author baseline.'],'errors':[]}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
