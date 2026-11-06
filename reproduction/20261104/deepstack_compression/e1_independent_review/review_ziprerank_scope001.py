"""Independent CPU reconstruction of fixed QI text-position scope, not quality."""
import hashlib,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=Path(__file__).resolve().parents[1];V=R.parents[2];H=R/'e6/ziprerank_scope';I=R/'e1_independent_review'
sys.path.insert(0,str(V/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit'))
import torch,transformers
from transformers import AutoProcessor,AutoConfig
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
p=load(H/'plan001.json');r=load(H/'result001.json');s=load(H/'monitor001/started.json');t=load(H/'monitor001/terminal.json')
assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and s['cwd']==str(R)
assert s['command'][1:]==['e6/ziprerank_scope/diagnose.py'] and all(sha(k)==v for k,v in p['source_sha256'].items()) and r['plan_sha256']==sha(H/'plan001.json') and r['cuda_initialized'] is False
assert torch.__version__==r['torch'] and transformers.__version__==r['transformers'] and inspect.getfile(Qwen3VLChatFixedRes)==r['author_wrapper_path']
rows=[json.loads(x) for x in (R/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];model=Path(p['model_snapshot']);proc=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False);cfg=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
w=object.__new__(Qwen3VLChatFixedRes);w.system_prompt=None;w.min_pixels=w.max_pixels=1048576;w.total_pixels=None;w.processor=proc;w.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
assert len(p['anchors'])==len(r['anchors'])==3
for a,actual in zip(p['anchors'],r['anchors']):
 row=next(x for x in rows if x['task']==a['task']);assert row['qid']==a['qid'] and actual['qid']==a['qid'] and actual['task']==a['task']
 assert sha(row['image_path'])==row['image_sha256']==a['image_sha256']==actual['image_sha256']
 assert hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==row['message_sha256']==a['message_sha256']==actual['message_sha256']
 messages=w._build_messages(row['message'],dataset=row['task']);rendered=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=w._processor_inputs(messages);ids=inputs['input_ids'][0].tolist();pos=ids.index(cfg.image_token_id)
 assert ids.count(cfg.image_token_id)==actual['image_token_count'] and pos==actual['first_image_token_index'] and len(ids)==actual['input_token_count'] and inputs['image_grid_thw'].tolist()==actual['image_grid_thw']
 excluded={cfg.image_token_id,cfg.video_token_id,cfg.vision_start_token_id,cfg.vision_end_token_id};selected=[i for i in range(pos) if ids[i] not in excluded]
 assert selected==actual['selected_token_indices'] and len(selected)==actual['selected_token_count'] and max(selected)<pos
 marker=rendered.index('<|image_pad|>');qpos=rendered.index(row['message'][1]['value']);assert marker<qpos
 prefix=proc.tokenizer.decode(ids[:pos],skip_special_tokens=False);assert prefix==rendered[:marker] and hashlib.sha256(prefix.encode()).hexdigest()==actual['selected_prefix_sha256']
 assert all(actual[k] is True for k in ('question_after_image_in_template','prefix_tokenization_matches_processor')) and actual['question_text_in_selected_scope'] is False
assert not torch.cuda.is_initialized()
out=I/'ziprerank_scope001_review.json';proof={'status':'passed','independent_review':True,'anchors_verified':3,'plan_sha256':sha(H/'plan001.json'),'result_sha256':sha(H/'result001.json'),'started_sha256':sha(H/'monitor001/started.json'),'terminal_sha256':sha(H/'monitor001/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'cuda_initialized':False,'limitations':['Three input diagnostics only; no accuracy, latency or general-task conclusion.'],'errors':[]}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
