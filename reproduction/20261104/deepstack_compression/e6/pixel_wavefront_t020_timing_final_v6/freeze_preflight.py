"""Freeze run-source noGPU preflight only; never authorize GPU."""
import json,os
from pathlib import Path
import preflight_gate as g
D=Path(__file__).resolve().parent
def main():
 out=D/'preflight_plan001.json'
 if out.exists():raise ValueError('existing plan')
 g.prior();g.cpu();before=g.sources();after=g.sources()
 if before!=after:raise ValueError('source changed during freeze')
 value={'scope':'actual noGPU import of final candidate 24-process controller and worker; no model or generation','source_sha256':after,'cpu_proof_sha256':g.sha(D/'cpu_proof001.json'),'prior_noGPU_review_sha256':g.sha(g.I/'pixel_wavefront_t020_timing_run_v2_preflight_result_review.json'),'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'limitations':['prospective GPU CLI remains locked; no model/GPU/NVML','independent candidate author provenance recorded in earlier identity chain','source and formal acceptance must be frozen before any GPU launch']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
