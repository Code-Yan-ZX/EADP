"""Freeze CPU-only preflight source inventory, not a GPU plan."""
import hashlib,inspect,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def main():
 cpu=json.loads((D/'cpu_proof002.json').read_text());terminal=json.loads((D/'monitor_cpu002/terminal.json').read_text());prior=B/'e4_c_budget_pilot_v2/preflight_plan001.json';p=json.loads(prior.read_text())
 if cpu['status']!='passed' or cpu['source_sha256_before']!=cpu['source_sha256_after'] or terminal['reason']!='completed' or terminal['worker_returncode']!=0 or terminal['remaining_process_group']!=[]:raise ValueError('CPU proof')
 source=dict(p['source_sha256']);source.update(cpu['source_sha256_after']);
 extras=[prior,D/'PLAN.md',D/'processor_host002.json',D/'monitor_processor_host002/started.json',D/'monitor_processor_host002/terminal.json',D/'cpu_proof002.json',D/'monitor_cpu002/started.json',D/'monitor_cpu002/terminal.json',B/'monitor/supervise.py',B/'initial_plan.json',B/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py',B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py',B/'e6/pixel_wavefront_t020_nvml_nine/design001.json',ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json']
 extras += [MODEL/x for x in ('config.json','generation_config.json','chat_template.json','preprocessor_config.json','video_preprocessor_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors.index.json')]
 from transformers.models.qwen3_vl import modeling_qwen3_vl,processing_qwen3_vl
 from transformers.models.qwen2_vl import image_processing_qwen2_vl_fast
 extras += [Path(inspect.getfile(x)).resolve() for x in (modeling_qwen3_vl,processing_qwen3_vl,image_processing_qwen2_vl_fast)]
 for path in extras:source[str(path.resolve())]=sha(path)
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('prior/source changed')
 out={'scope':'three-anchor low-res processor noGPU only','source_sha256':source,'images':3,'target_resolution':512,'target_visual_tokens':256,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'limitations':['source plan includes accepted DEV mixed-prediction hashes','only processor/mRoPE and weight preflight; no model or actual DS','no GPU authority']}
 with (D/'preflight_plan001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
