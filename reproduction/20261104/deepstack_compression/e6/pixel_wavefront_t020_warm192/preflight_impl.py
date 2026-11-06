"""Real no-GPU processor and weight preflight; requires external admission."""
import hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify_sources(source):
 if not isinstance(source,dict) or not source or any(sha(k)!=v for k,v in source.items()):raise ValueError('source identity')
def inventory(design):
 if design.get('selected_questions')!=192 or design.get('measured_records')!=384 or design.get('warmup_records')!=8:raise ValueError('design size')
 items={}
 for block in design['blocks']:
  for x in block:
   key=(x['task'],str(x['qid']))
   if key in items:raise ValueError('duplicate measured question')
   items[key]=x
 warm=design['processes'][0]['warmup'];wkey=(warm['task'],str(warm['qid']))
 if len(items)!=192 or wkey in items or any(p['warmup']!=warm for p in design['processes']):raise ValueError('warmup or measured inventory')
 items[wkey]=warm;return items

def run(design,source,model_module,out):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('no-device preflight only')
 verify_sources(source);candidate,prepared,input_preflight,rows,refs=model_module.prepare_cpu(source);torch=prepared['torch']
 from transformers import AutoProcessor,AutoConfig
 processor=AutoProcessor.from_pretrained(candidate.MODEL,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(candidate.MODEL,local_files_only=True,trust_remote_code=False)
 wrapper=object.__new__(prepared['wrapper_class']);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16);wrapper.verbose=False;wrapper.post_process=False
 checked=[]
 for key,expected in inventory(design).items():
  row=rows[key];ref=refs[key]
  if any(str(row.get(k))!=str(expected.get(k)) for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')) or ref['processor']!=expected['processor'] or ref['image_start']!=expected['image_start'] or sha(row['image_path'])!=expected['image_sha256']:raise ValueError('input image/reference mismatch')
  actual=wrapper._processor_inputs(wrapper._build_messages(row['message'],dataset=row['task']));observed=input_preflight.validate_processor(torch,actual,ref['processor'],config.image_token_id,ref['image_start'])
  if not bool(torch.all(actual['attention_mask']==1)):raise ValueError('unexpected input padding')
  checked.append({'task':key[0],'qid':key[1],'group_id':row['group_id'],'image_sha256':row['image_sha256'],'processor':observed['processor'],'image_start':observed['image_start']});del actual
 weight_manifest=ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json';items=json.loads(weight_manifest.read_text())['files'];weights=[]
 if len(items)!=4 or len({x['path'] for x in items})!=4:raise ValueError('weight inventory')
 for item in items:
  p=Path(item['path']);st=p.stat()
  if st.st_size!=item['bytes'] or sha(p)!=item['expected_lfs_sha256']:raise ValueError('model weight hash')
  after=p.stat();before=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
  if before!=[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns]:raise ValueError('weight changed while hashing')
  weights.append({'path':str(p),'stat':before,'sha256':item['expected_lfs_sha256']})
 if len(checked)!=193 or torch.cuda.is_initialized():raise ValueError('preflight CUDA or input inventory')
 verify_sources(source)
 result={'status':'passed','stage':'193 actual inputs/processor and four weight payloads, no GPU/model','inputs_checked':len(checked),'processor_inputs':checked,'weights':weights,'cuda_initialized':False,'model_loaded':False,'GPU_authorized':False,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted E1 and prior six-arm DEV records solely for identity','limitations':['no model generation or candidate equivalence tested','four weight hashes precede later GPU run; endpoint stats cannot exclude transient replacement','warm file cache established by input processing']}
 with Path(out).open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return result
