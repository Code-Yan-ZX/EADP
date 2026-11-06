"""Freeze descriptive analysis after GPU/formal plans, before measurements."""
import json,os
from pathlib import Path
import freeze
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';A=B/'e6/pixel_wavefront_t020_nvml_execution_accept';S=B/'e6/pixel_wavefront_t020_nvml_execution_analysis'
def main():
 out=S/'analysis_plan001.json';gpu=E/'gpu_plan001.json';formal=A/'verification_plan001.json'
 if out.exists():raise ValueError('analysis plan already exists')
 p=freeze.read(gpu);f=freeze.read(formal);c=freeze.read(S/'cpu_proof002.json');source=p['source_sha256']
 if p.get('GPU_used') is not False or f.get('gpu_plan_sha256')!=freeze.sha(gpu) or c.get('status')!='passed' or c.get('GPU_used') is not False or c.get('GT_opened') is not False or c.get('timing_results_opened') is not False or c.get('source_sha256_before')!=c.get('source_sha256_after') or any(source.get(k)!=v or freeze.sha(k)!=v for k,v in c['source_sha256_after'].items()) or any(freeze.sha(k)!=v for k,v in source.items()):raise ValueError('analysis sources or plan changed')
 v={'scope':'nine_process_pixel_nvml_descriptive','gpu_plan_sha256':freeze.sha(gpu),'formal_plan_sha256':freeze.sha(formal),'source_sha256':source,'GPU_used':False,'GT_opened':False,'scoring_performed':False,'limits':['all nine isolated processes retained; post-hoc three-image pilot','process, device and allocator quantities differ; sampling can miss peaks','no timing or equal-compute claim']}
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
