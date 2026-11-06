"""Exercise the actual Transformers stopping path with a tiny CPU-only model."""
import hashlib, importlib.util, json, os
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, StoppingCriteriaList

B=Path(__file__).resolve().parents[2]
source=B/'e6/pixel_wavefront_t020_model_timing_v2/t020_timing_boundaries.py'
spec=importlib.util.spec_from_file_location('warm192_boundary_cpu_probe',source)
boundary=importlib.util.module_from_spec(spec);spec.loader.exec_module(boundary)
torch.manual_seed(1)
model=GPT2LMHeadModel(GPT2Config(vocab_size=31,n_positions=16,n_embd=16,n_layer=1,n_head=2,bos_token_id=1,eos_token_id=2,pad_token_id=0)).eval()
prompt=torch.tensor([[1,3,4]])
def run(b):
 return model.generate(input_ids=prompt,do_sample=False,max_new_tokens=1,stopping_criteria=StoppingCriteriaList([b]))
with torch.inference_mode():output,record=boundary.measure(run,lambda:None)
assert output.shape==(1,4) and output[0,:3].equal(prompt[0])
assert record['criteria_calls']==1 and record['first_criteria_length']==4
assert not torch.cuda.is_initialized()
proof={'status':'passed','synthetic_random_cpu_model':True,'actual_transformers_generate':True,'model':'GPT2 tiny random; not Qwen or Pixel','prompt_length':3,'output_length':4,'boundary_calls':record['criteria_calls'],'cuda_initialized':False,'model_weights_loaded':False,'GT_opened':False,'predictions_opened':False,'boundary_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out=Path(__file__).with_name('one_token_generate001.json')
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
