"""Conservative observer lifetime admission, before any GPU query."""
import os,time
from pathlib import Path

def lifetime(start_ticks,hz,monotonic_ns,boottime_ns,duration_seconds,needed_seconds):
 if any(type(x) is not int for x in (start_ticks,hz,monotonic_ns,boottime_ns,duration_seconds,needed_seconds)) or min(start_ticks,hz,monotonic_ns,duration_seconds,needed_seconds)<=0 or boottime_ns<monotonic_ns:raise ValueError('invalid observer clock')
 # /proc starttime is rounded to clock ticks and expressed in boot time.
 # Current boot-minus-monotonic offset is at least the offset at process start,
 # so subtracting it and one tick yields a conservative monotonic lower bound.
 tick_ns=(1_000_000_000+hz-1)//hz
 start_lower=start_ticks*1_000_000_000//hz-(boottime_ns-monotonic_ns)-tick_ns
 if start_lower<=0 or start_lower>monotonic_ns:raise ValueError('invalid observer start')
 deadline_lower=start_lower+duration_seconds*1_000_000_000
 if deadline_lower<monotonic_ns+needed_seconds*1_000_000_000:raise ValueError('observer insufficient remaining lifetime')
 return {'observer_start_lower_monotonic_ns':start_lower,'deadline_lower_monotonic_ns':deadline_lower,'checked_monotonic_ns':monotonic_ns,'required_remaining_seconds':needed_seconds,'clock_ticks_per_second':hz}

def proc_stat(pid):
 s=(Path('/proc')/str(pid)/'stat').read_text(); tail=s[s.rfind(')')+2:].split()
 if len(tail)<20:raise ValueError('invalid proc stat')
 return {'state':tail[0],'parent_pid':int(tail[1]),'start_ticks':int(tail[19])}

def verify(observer_pid,supervisor_pid,supervisor_script,run_dir,outer_seconds,grace_seconds,*,clock=time.clock_gettime_ns):
 if type(observer_pid) is not int or type(supervisor_pid) is not int or min(observer_pid,supervisor_pid)<=0:raise ValueError('invalid supervisor identity')
 proc=proc_stat(observer_pid);parent=proc_stat(supervisor_pid)
 if proc['parent_pid']!=supervisor_pid or proc['state'] in ('Z','X') or parent['state'] in ('Z','X'):raise ValueError('observer or supervisor not live')
 cmd=[x.decode() for x in (Path('/proc')/str(supervisor_pid)/'cmdline').read_bytes().split(b'\0') if x]
 if not any(Path(x).resolve()==Path(supervisor_script).resolve() for x in cmd) or cmd.count('--run-dir')!=1 or Path(cmd[cmd.index('--run-dir')+1]).resolve()!=Path(run_dir).resolve():raise ValueError('observer supervisor identity')
 if os.readlink('/proc/self/ns/time')!=os.readlink(f'/proc/{observer_pid}/ns/time'):raise ValueError('observer clock namespace')
 now=clock(time.CLOCK_MONOTONIC);boot=clock(time.CLOCK_BOOTTIME)
 value=lifetime(proc['start_ticks'],os.sysconf('SC_CLK_TCK'),now,boot,23000,outer_seconds+grace_seconds)
 value.update({'observer_pid':observer_pid,'supervisor_pid':supervisor_pid,'time_namespace':os.readlink('/proc/self/ns/time'),'GPU_queried':False})
 return value
