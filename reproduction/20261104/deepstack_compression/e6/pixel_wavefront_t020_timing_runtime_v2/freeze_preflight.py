"""Freeze only no-GPU container preflight; does not authorize timing."""
import json,os
from pathlib import Path
import preflight_gate as g
D=Path(__file__).resolve().parent

def main():
 out=D/'preflight_plan001.json'
 if out.exists():raise ValueError('existing plan')
 _,prior=g.prior();g.cpu();before=g.sources();after=g.sources()
 if before!=after:raise ValueError('source changed during freeze')
 value={'scope':'runtime bridge import in retained container without GPU; no model load and no timing','source_sha256':after,'cpu_proof_sha256':g.sha(D/'cpu_proof003.json'),'prior_noGPU_review_sha256':g.sha(g.I/'pixel_wavefront_t020_timing_execution_v2_preflight_result_review.json'),'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'limitations':['prior reviewer authored the wavefront candidate','this preflight does not implement an executable GPU controller or NVML sampler','the timed run requires separate source freeze, fixed acceptance, GPU review and root scheduling']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
