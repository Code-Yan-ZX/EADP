"""Candidate memory worker continuation, deliberately without a GPU CLI.

A future independently reviewed entry must validate a real prelaunch proof,
container configuration and source/weight manifest before calling this helper.
This module itself cannot establish those prerequisites or grant GPU authority.
"""
import json,os,time
from pathlib import Path
from bridge import make_measure
from worker_barrier import handshake

def save_new(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def run_after_preflight(*,stage,mode,nonce,directory,prepared,expected_schedule,identity,torch,native,wrapper_class,preflight_one,execute_one,memory_measure,run_memory_jobs,durable_journal,sha,handshake_fn=handshake):
 """Continue after separately established preflight; not an admission gate."""
 d=Path(directory).resolve()
 if stage not in ('smoke','full') or mode not in ('C-ON','C-DROP','C-SKIP') or not d.is_dir():raise ValueError('worker stage, mode or directory')
 if any((d/x).exists() for x in ('frozen.json','journal.jsonl','worker_complete.json')):raise ValueError('worker run directory reused')
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized before memory barrier')
 expected,release=handshake_fn(d,nonce,torch.cuda.is_initialized)
 if release.get('status')!='release' or expected.get('mode')!=mode or expected.get('nonce')!=nonce or release.get('mode')!=mode or release.get('nonce')!=nonce or release.get('container_id')!=expected.get('container_id'):raise ValueError('barrier identity mismatch')
 sources=prepared['source_sha256'];weights=prepared['weights_before']
 if prepared['schedule']!=expected_schedule:raise ValueError('memory schedule differs from independent reconstruction')
 for path,before in weights.items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed before model')
 if any(sha(path)!=digest for path,digest in sources.items()):raise ValueError('worker source changed before model')
 frozen={k:prepared[k] for k in ('schedule','source_sha256','e1_acceptance_sha256','e1_started_sha256','e1_terminal_sha256','anchor_sha256')}
 frozen.update(stage=stage,mode=mode,role='memory_isolated',identity=identity,barrier_expected=expected,barrier_release=release,weight_stats_before=weights,
               scope='separate process memory measurement; no TTFT or accuracy inference',GT_opened=False,scoring_performed=False)
 save_new(d/'frozen.json',frozen)
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 wrapper=wrapper_class(**prepared['plan']['recipe'])
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
 bykey={(r['task'],str(r['qid'])):r for r in prepared['rows']}
 selected={(j['task'],str(j['qid'])) for j in prepared['schedule']['schedule'] if j['mode']==mode}
 if any(k not in bykey or k not in prepared['refs'] for k in selected):raise ValueError('memory inputs missing')
 for key in sorted(selected):preflight_one(torch,wrapper,bykey[key],prepared['refs'][key])
 torch.cuda.synchronize()
 measure=make_measure(torch,wrapper,native,execute_one,memory_measure,mode)
 stream,journal=durable_journal(d/'journal.jsonl')
 try:result=run_memory_jobs(prepared['schedule'],expected_schedule,prepared['rows'],stage,mode,measure,journal)
 finally:stream.close()
 expected_count={'smoke':24,'full':1980}[stage]
 if result.get('status')!='completed' or result.get('jobs')!=expected_count or result.get('mode')!=mode:raise ValueError('memory schedule incomplete')
 for path,before in weights.items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
 if any(sha(path)!=digest for path,digest in sources.items()):raise ValueError('worker source changed')
 save_new(d/'worker_complete.json',{'status':'completed','stage':stage,'mode':mode,'jobs':result['jobs'],'finished_unix':time.time(),'scope':'memory only'})
 return result

def main():raise SystemExit('locked: independently reviewed worker/controller/sampler admission is not implemented')
if __name__=='__main__':main()
