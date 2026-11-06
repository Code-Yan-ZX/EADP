"""Existing bounded supervisor with conservative GPU snapshot time windows.

The window brackets the complete health() call, including the nvidia-smi query
and subsequent CPU checks. It therefore conservatively overmarks, never
pretends to know the narrower GPU query interval. No supervisor source changes.
"""
import importlib.util,os,sys,time
from pathlib import Path

def timed_health(health,clock=time.monotonic_ns,namespace=lambda:os.readlink('/proc/self/ns/time')):
 def run(*args,**kwargs):
  start=clock();record=health(*args,**kwargs);end=clock()
  if end<start:raise ValueError('nonmonotonic health clock')
  record['health_snapshot_window_monotonic_ns']=[start,end]
  record['time_namespace']=namespace()
  return record
 return run

def main():
 path=Path(__file__).resolve().parent.parent/'monitor/supervise.py'
 spec=importlib.util.spec_from_file_location('eadp_timing_supervise_base',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.health=timed_health(module.health)
 return module.main()
if __name__=='__main__':sys.exit(main())
