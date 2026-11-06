"""Injected synchronized first-token timing with explicit clock provenance."""
import os,time
class Boundary:
 def __init__(self,sync,mono,wall):self.sync=sync;self.mono=mono;self.wall=wall;self.first=None;self.first_wall=None;self.first_length=None;self.calls=0
 def __call__(self,input_ids,scores,**kwargs):
  self.calls+=1
  if self.first is None:self.sync();self.first=self.mono();self.first_wall=self.wall();self.first_length=int(input_ids.shape[1])
  return False
def measure(run,synchronize,mono=time.perf_counter_ns,wall=time.time_ns):
 synchronize();start=mono();start_wall=wall();b=Boundary(synchronize,mono,wall);result=run(b);synchronize();end=mono();end_wall=wall()
 if b.first is None or not 0<start<b.first<=end:raise ValueError('invalid first-token boundary')
 return result,{'t0_before_image_read_ns':start,'first_token_selected_ns':b.first,'generation_complete_ns':end,'t0_unix_ns':start_wall,'first_token_unix_ns':b.first_wall,'completion_unix_ns':end_wall,'logits_ready_ns':None,'explicit_sync_boundaries':['before_t0','first_token','generation_end'],'first_criteria_length':b.first_length,'criteria_calls':b.calls,'clock_implementation':time.get_clock_info('perf_counter').implementation,'time_namespace':os.readlink('/proc/self/ns/time'),'wall_clock_note':'wall-clock reads immediately follow monotonic reads and are not simultaneous'}
