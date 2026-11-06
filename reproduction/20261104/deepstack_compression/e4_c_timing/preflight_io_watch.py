"""Host-only, low-frequency I/O/progress metadata during CPU weight hashing.

Samples the mounted block device counters and E2/E3 journal file size; never
opens model predictions, truth, or a GPU. Aggregate device I/O is not attributed
to an individual process. Runs until the supervised preflight terminal appears.
"""
import argparse,json,os,time
from pathlib import Path

def disk_counters(device):
 major,minor=os.major(device),os.minor(device)
 for line in Path('/proc/diskstats').read_text().splitlines():
  f=line.split()
  if int(f[0])==major and int(f[1])==minor:
   return {'major':major,'minor':minor,'device':f[2],'reads_completed':int(f[3]),'sectors_read':int(f[5]),
           'writes_completed':int(f[7]),'sectors_written':int(f[9]),'io_milliseconds':int(f[12])}
 return {'major':major,'minor':minor,'unavailable':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--started',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--progress',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 begin=time.monotonic();device=Path('/media/disk2').stat().st_dev;seen=False
 with a.out.open('x') as f:
  while time.monotonic()-begin<1500:
   seen=seen or a.started.exists()
   try:st=a.progress.stat();progress={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
   except FileNotFoundError:progress=None
   record={'unix':time.time(),'monotonic':time.monotonic(),'preflight_started':seen,'preflight_terminal_exists':a.terminal.exists(),
           'panel_progress_file':str(a.progress),'panel_progress_metadata':progress,'disk':disk_counters(device)}
   f.write(json.dumps(record,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
   if seen and record['preflight_terminal_exists']:return 0
   time.sleep(15)
 return 1
if __name__=='__main__':raise SystemExit(main())
