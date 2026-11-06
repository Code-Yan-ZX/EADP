"""No-GT host-side acceptance for the complete three-anchor smoke."""
import argparse,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from admission import HERE,BASE,admit,sources,validate_container
from real_worker import sha
from arm import validate_trace,compare_keepall,validate_recipe,validate_fingerprints
from latency_admission import terminal
DTYPES={'torch.bfloat16':torch.bfloat16,'torch.float16':torch.float16,'torch.float32':torch.float32,'torch.float64':torch.float64,'torch.int64':torch.int64,'torch.int32':torch.int32,'torch.uint8':torch.uint8,'torch.bool':torch.bool}
def readlines(path):return [json.loads(x) for x in path.read_text().splitlines() if x]
def unpack(arrays,name,meta):
 a=arrays[name];dtype=DTYPES.get(meta['dtype']);shape=meta['shape']
 if dtype is None or a.dtype!=np.uint8 or sha_bytes(a.tobytes())!=meta['sha256']:raise ValueError('tensor hash/dtype '+name)
 x=torch.frombuffer(bytearray(a.tobytes()),dtype=dtype)
 if x.numel()!=math.prod(shape):raise ValueError('tensor shape '+name)
 return x.reshape(shape)
def sha_bytes(b):
 import hashlib
 return hashlib.sha256(b).hexdigest()
def native_array(record,_path,arrays):
 fields={'visual':record['main_visual'],'prefill':record['prefill_embeds'],'prefill_logits':record['prefill_logits'],**{'ds'+str(i):m for i,m in enumerate(record['full_deepstack'])},**{'injected'+str(i):m for i,m in enumerate(record['injected_deepstack'])},**{'processor_'+k:m for k,m in record['processor'].items()}}
 if set(fields)!=set(arrays.files):raise ValueError('native tensor inventory')
 x={k:unpack(arrays,k,v) for k,v in fields.items()}
 if not torch.isfinite(x['prefill_logits']).all():raise ValueError('nonfinite native logits')
 return x
def local_arrays(r,arrays):
 fields={'premerger':r['premerger'],'averaged_premerger':r['averaged_premerger'],'visual':r['main_visual'],'prefill':r['prefill_embeds'],'prefill_logits':r['prefill_logits'],'question_ids':r['question_ids'],'question_embeddings':r['question_embeddings'],'prefill_mask':r['prefill_mask'],'prefill_positions':r['prefill_positions'],'visual_mask':r['visual_mask'],'retained_indices':r['retained_indices'],**{'ds'+str(i):m for i,m in enumerate(r['full_deepstack'])},**{'injected'+str(i):m for i,m in enumerate(r['injected_deepstack'])},**{'processor_'+k:m for k,m in r['processor'].items()}}
 if set(fields)!=set(arrays.files):raise ValueError('local tensor inventory')
 x={k:unpack(arrays,k,v) for k,v in fields.items()}
 if not torch.isfinite(x['prefill_logits']).all() or not torch.equal(x['premerger'].reshape(-1,4,x['premerger'].shape[-1]).mean(1),x['averaged_premerger']):raise ValueError('nonfinite logits or premerger grouping')
 return x

def local_identity(native,r,x,native_arrays):
 ids=unpack(native_arrays,'processor_input_ids',native['processor']['input_ids']);full=unpack(native_arrays,'prefill',native['prefill_embeds']);mask=unpack(native_arrays,'processor_attention_mask',native['processor']['attention_mask']);positions=torch.tensor(native['prefill_position_ids']);vm=torch.tensor(native['injected_mask']);spatial=r['spatial_ids'];k=r['budget']
 if len(spatial)!=k or spatial!=sorted(set(spatial)) or any(i<0 or i>=1024 for i in spatial):raise ValueError('selected ids invalid')
 if k==256 and (len(r['greedy_ids'])!=256 or sorted(r['greedy_ids'])!=spatial or r['selector_repeat_exact'] is not True):raise ValueError('author selector sequence')
 image_positions=torch.nonzero(vm[0],as_tuple=True)[0]
 if image_positions.numel()!=1024:raise ValueError('native visual mask')
 start=int(image_positions[0]);keep=torch.tensor(list(range(start))+[start+i for i in spatial]+list(range(start+1024,ids.shape[1])))
 if not torch.equal(x['retained_indices'],keep) or not torch.equal(x['prefill'],full.index_select(1,keep)) or not torch.equal(x['prefill_mask'],mask.index_select(1,keep)) or not torch.equal(x['prefill_positions'],positions.index_select(2,keep)) or not torch.equal(x['visual_mask'],vm.index_select(1,keep)):raise ValueError('actual P3 prefill differs from native reconstruction')
 for i in range(3):
  d=unpack(native_arrays,'ds'+str(i),native['full_deepstack'][i])
  if not torch.equal(x['ds'+str(i)],d) or not torch.equal(x['injected'+str(i)],d[spatial]):raise ValueError('actual DS alignment')
 if not torch.equal(x['visual'],unpack(native_arrays,'visual',native['main_visual'])) or r['processor']!=native['processor'] or r['rope_delta']!=native['rope_delta']:raise ValueError('native input/visual identity')
 validate_trace(r['decode_trace'],r['output_ids'],ids.shape[1],k,int(r['rope_delta'][0][0]))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run=HERE/'run_001';s=HERE/'monitor_run001/started.json';t=HERE/'monitor_run001/terminal.json';started=terminal(s,t);launch=json.loads((run/'launch.json').read_text());ex=json.loads((run/'exit.json').read_text())
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or launch.get('controller')!=ex.get('controller') or launch['controller']['pid']!=started['worker_pid'] or launch['approval_bundle']!=admit(preflight_only=False,host=True):raise ValueError('run exit/approval')
 obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];validate_container(launch,ex,obj,gpu=True)
 frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text());expected={str(p.resolve()):sha(p) for p in sources()}
 if frozen.get('source_sha256')!=expected or done.get('status')!='completed' or done.get('generations')!=18 or done.get('comparisons')!=6 or frozen.get('GT_opened') is not False:raise ValueError('incomplete execution')
 rows=readlines(run/'records.jsonl');gen=readlines(run/'generated.jsonl');pairs=readlines(run/'comparisons.jsonl');plan=json.loads((HERE/'plan001.json').read_text())
 if len(rows)!=18 or len(gen)!=18 or len(pairs)!=6:raise ValueError('incomplete records')
 from transformers import AutoTokenizer
 from real_worker import PLAN
 tok=AutoTokenizer.from_pretrained(json.loads(PLAN.read_text())['recipe']['model_path'],local_files_only=True);fixed=tok.encode(' the',add_special_tokens=False)[0];array_hash={};i=0
 for anchor in plan['anchors']:
  for forced in (False,True):
   block=rows[i:i+3];events=gen[i:i+3];i+=3
   recipe=json.loads(PLAN.read_text())['recipe'];natural_eos=block[0]['eos_ids'] if not forced else None
   if forced:
    natural_eos=rows[i-6]['eos_ids']
   for rec,event,path in zip(block,events,('stock_native','local_keepall','local_stacktok')):
    validate_recipe(rec,recipe,forced,natural_eos)
    if rec['decode_trace'][0]['position_ids']!=rec['prefill_position_ids']:raise ValueError('actual prefill position trace mismatch')
    full=rec['processor']['input_ids']['shape'][1];validate_fingerprints(rec['cache_fingerprints'],forced,full-(0 if path!='local_stacktok' else 768))
    if any(rec.get(k)!=anchor[k] for k in ('task','qid','group_id','image_sha256','message_sha256')) or rec['path']!=path or rec['forced']!=forced:raise ValueError('wrong question or path')
    if any(rec.get(k)!=event.get(k) for k in ('task','qid','group_id','path','forced','output_ids','eos_ids','generation_scalars')):raise ValueError('generation journal mismatch')
    if tok.decode(rec['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=rec['answer'] or not rec['output_ids'] or (forced and rec['output_ids']!=[fixed]*33) or (not forced and not (rec['eos'] or rec['truncated'])):raise ValueError('output decode/termination')
    file=Path(rec['arrays']['path']).resolve()
    if file.parent!=(run/'arrays').resolve() or sha(file)!=rec['arrays']['sha256'] or str(file) in array_hash:raise ValueError('tensor archive identity')
    array_hash[str(file)]=sha(file)
   compare_keepall(block[0],block[1])
   with np.load(block[0]['arrays']['path'],allow_pickle=False) as na:
    native_array(block[0],block[0]['arrays']['path'],na)
    for rec in block[1:]:
     with np.load(rec['arrays']['path'],allow_pickle=False) as aa:local_identity(block[0],rec,local_arrays(rec,aa),na)
   pair=pairs[i//3-1]
   if pair!={'task':anchor['task'],'qid':anchor['qid'],'forced':forced,'status':'passed'}:raise ValueError('keepall comparison journal')
 value={'status':'passed','images':3,'generations':18,'keepall_pairs':6,'GT_opened':False,'scoring_performed':False,'scope':'Local Qwen3 adaptation; original selector GPU repeat and reconstructed prefill/DS identity. KV byte arrays not archived; fingerprints only. Not author Qwen3 implementation.','arrays_sha256':array_hash,'run_started_sha256':sha(s),'run_terminal_sha256':sha(t),'run_exit_sha256':sha(run/'exit.json'),'verifier_sha256':sha(__file__)}
 with a.out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
