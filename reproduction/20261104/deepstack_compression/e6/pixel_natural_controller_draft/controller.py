"""Draft bounded sequential container orchestration; GPU entry hard locked.

The injectable loop is CPU-testable and doesn't itself launch Docker. A future
reviewed runtime must establish the externally supervised controller identity,
time budget, frozen sources, preflight/reviewer proof and container profile.
"""
import importlib.util,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];PILOT=BASE/'e6/pixel_natural_timing_pilot_v3'
spec=importlib.util.spec_from_file_location('_pixel_controller_contract',PILOT/'pilot_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)

def run_sequential(jobs,anchors,device_pids,launch_one,emit):
 contract.validate_schedule(jobs,anchors)
 identities=set()
 for job in jobs:
  if device_pids()!=[]:raise ValueError('GPU not exclusive before job')
  emit({'event':'launching','job':job})
  try:
   exit_record=launch_one(job)
   host=(exit_record.get('container_identity') or {}).get('host_init') or {}
   ident=(host.get('pid'),host.get('start_ticks'),host.get('boot_id'))
   if (exit_record.get('returncode')!=0 or exit_record.get('container_exit_code')!=0 or
       exit_record.get('container_stopped_verified') is not True or exit_record.get('timed_out') is not False or
       exit_record.get('interrupted') is not None or exit_record.get('failure_type') is not None or
       exit_record.get('client_exited_while_container_running') is not False or exit_record.get('gpu_pids_after')!=[] or
       type(ident[0]) is not int or ident[0]<=0 or type(ident[1]) is not int or ident[1]<=0 or not ident[2] or ident in identities):raise ValueError('container exit or process identity failed')
   identities.add(ident);emit({'event':'finished','job':job,'host_init':host})
  except BaseException as exc:
   emit({'event':'failed','job':job,'error_type':type(exc).__name__,'error':str(exc)})
   raise
 return len(identities)

def main():
 raise ValueError('GPU controller locked: no reviewed Docker worker, admission, timing supervisor or deadlines')
if __name__=='__main__':main()
