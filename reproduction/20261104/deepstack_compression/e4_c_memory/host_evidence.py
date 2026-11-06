"""Host identity/stopped-event checks for future isolated memory controller.

Checking an event's fields does not prove who produced it; a reviewed launcher
must bind the event and raw Docker inspect to its own process and supervisor.
"""
import hashlib,json
from pathlib import Path
from contract import identity

def process_alive(expected,proc=Path('/proc')):
 identity(expected)
 try:
  if (proc/'sys/kernel/random/boot_id').read_text().strip()!=expected['boot_id']:return False
  root=proc/str(expected['host_pid']);stat=(root/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
  if tail[0]=='Z' or int(tail[19])!=expected['start_ticks']:return False
  return expected['container_id'] in (root/'cgroup').read_text()
 except (FileNotFoundError,ProcessLookupError,IndexError,ValueError):return False

def stopped_ns(expected,event,inspect_path):
 identity(expected)
 if not isinstance(event,dict) or any(event.get(k)!=expected[k] for k in ('nonce','container_id','host_pid','start_ticks','boot_id')) or event.get('status')!='stopped' or type(event.get('monotonic_ns')) is not int:raise ValueError('stopped event identity mismatch')
 raw=Path(inspect_path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=event.get('inspect_sha256'):raise ValueError('stopped inspect hash mismatch')
 inspect=json.loads(raw);inspect=inspect[0] if isinstance(inspect,list) and len(inspect)==1 else inspect
 state=inspect.get('State',{})
 if inspect.get('Id')!=expected['container_id'] or state.get('Running') is not False or state.get('Pid')!=0:raise ValueError('container not observed stopped')
 return event['monotonic_ns']
