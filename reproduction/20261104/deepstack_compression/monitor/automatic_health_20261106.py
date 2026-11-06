#!/usr/bin/env python3
"""Bounded, read-only periodic health records for the user-authorized study.

The process does not kill or signal any experiment. Every health query is
bracketed in a shared manual-window log for later timing-interference audits.
"""
import datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
DEADLINE=1794065414
INTERVAL=2700
EVENTS=BASE/'monitor/automatic_health_20261106_events.jsonl'
WINDOWS=BASE/'monitor/automatic_health_20261106_windows.jsonl'
stop=False
def event(kind,**extra):
 value={'event':kind,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'monotonic_ns':time.monotonic_ns(),**extra}
 with EVENTS.open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
def on_signal(sig,_frame):
 global stop
 stop=True
 event('signal',signal=sig)
def main():
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,on_signal)
 event('monitor_started',deadline_unix=DEADLINE,interval_seconds=INTERVAL,window_log=str(WINDOWS),policy='read-only; no external process signalling')
 due=time.monotonic()+INTERVAL
 while not stop and time.time()<DEADLINE:
  remaining=min(due-time.monotonic(),DEADLINE-time.time())
  if remaining>0:
   time.sleep(min(remaining,60));continue
  if time.time()>=DEADLINE:break
  try:
   result=subprocess.run([sys.executable,str(BASE/'monitor/hourly_timing_health_eadp.py'),'--manual-log',str(WINDOWS),'--reason','bounded periodic project health'],capture_output=True,text=True,timeout=45)
   event('health_completed',returncode=result.returncode,stdout=result.stdout[-3000:],stderr=result.stderr[-1000:])
  except (OSError,subprocess.TimeoutExpired) as exc:event('health_failed',error=repr(exc))
  due=time.monotonic()+INTERVAL
 event('monitor_finished',reason='signal' if stop else 'deadline')
if __name__=='__main__':main()
