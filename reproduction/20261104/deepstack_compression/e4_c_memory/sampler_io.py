"""Host sampling wiring; callable only after future reviewed memory admission.

No CLI is provided. The external controller must supervise this function,
provide inspected host identity, validate its ready event before releasing the
worker, and write the stopped event only after its own Docker inspection.
"""
import json,os,subprocess,time
from pathlib import Path
from barrier import publish
from host_evidence import process_alive,stopped_ns
from sampler import sample_process

def sample_only(expected,journal_path,ready_path,stopped_path,inspect_path,interval_ns,timeout_ns,
                alive=process_alive,clock=time.monotonic_ns,sleep=time.sleep,run=subprocess.run):
 journal_path=Path(journal_path);stopped_path=Path(stopped_path)
 with journal_path.open('x') as stream:
  def record(event):
   stream.write(json.dumps(event,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())
  def query():
   result=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
   return result.stdout
  def ready(started,first):publish(ready_path,{'started':started,'first_sample':first})
  def stopped(_expected):
   try:event=json.loads(stopped_path.read_text())
   except FileNotFoundError:return None
   return stopped_ns(expected,event,inspect_path)
  return sample_process(expected,alive,query,clock,lambda ns:sleep(ns/1e9),record,ready,stopped,interval_ns,timeout_ns)
