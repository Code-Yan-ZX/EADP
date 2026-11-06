"""Bind frozen CPU verifier and GPU plan before any real GPU measurements."""
import json,os
from pathlib import Path
import freeze
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';A=B/'e6/pixel_wavefront_t020_nvml_execution_accept'
def main():
 out=A/'verification_plan001.json';gpu=E/'gpu_plan001.json'
 if out.exists():raise ValueError('formal plan already exists')
 p=freeze.read(gpu);cpu=freeze.read(A/'cpu_proof006.json');source=p['source_sha256']
 if p.get('GPU_used') is not False or cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after') or any(source.get(k)!=v or freeze.sha(k)!=v for k,v in cpu['source_sha256_after'].items()) or any(freeze.sha(k)!=v for k,v in source.items()):raise ValueError('GPU/formal source drift')
 value={'gpu_plan_sha256':freeze.sha(gpu),'source_sha256':source,'CPU_proof_sha256':freeze.sha(A/'cpu_proof006.json'),'GPU_used':False,'GT_opened':False,'scoring_performed':False,'scope':'post-run verification of nine isolated NVML jobs; no timing inference'}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
