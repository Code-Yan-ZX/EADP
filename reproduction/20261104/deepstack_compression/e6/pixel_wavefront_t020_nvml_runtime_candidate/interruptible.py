"""Signal-aware lifecycle wrapper; no Docker or GPU entry point."""
import signal,threading
from controller_io import contract
from lifecycle import run

def run_control(ops):
 if threading.current_thread() is not threading.main_thread():raise RuntimeError('signal lifecycle must run in main thread')
 signals=(signal.SIGTERM,signal.SIGINT);previous={sig:signal.getsignal(sig) for sig in signals}
 def interrupt(sig,frame):
  # Once interrupted, give cleanup one uninterrupted bounded opportunity.
  for other in signals:signal.signal(other,signal.SIG_IGN)
  raise InterruptedError('controller received signal '+str(sig))
 try:
  for sig in signals:signal.signal(sig,interrupt)
  return run(ops,contract.admit_release,contract.audit)
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)
