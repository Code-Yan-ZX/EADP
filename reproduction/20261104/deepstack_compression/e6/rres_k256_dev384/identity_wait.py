"""Wait only for transient missing or partially written container identity."""
import json,time
def wait(read,paths,*,clock=time.monotonic,sleep=time.sleep,timeout=60):
 deadline=clock()+timeout
 while True:
  try:return tuple(read(path) for path in paths)
  except (FileNotFoundError,json.JSONDecodeError):
   if clock()>deadline:raise TimeoutError('container identity publication')
   sleep(.2)
