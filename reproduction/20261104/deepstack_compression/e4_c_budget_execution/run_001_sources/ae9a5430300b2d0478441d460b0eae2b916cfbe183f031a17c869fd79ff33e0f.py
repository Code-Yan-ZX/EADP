"""TTFT wall-clock boundary probe; time source and GPU synchronization injected."""
import os,time
class FirstTokenBoundary:
 """Transformers stopping criterion called after one selected token is appended.

 Requires source/version and runtime identity proof; does not itself stop generation.
 """
 def __init__(self,synchronize,clock=time.perf_counter_ns):self.synchronize=synchronize;self.clock=clock;self.first_token_ns=None;self.first_length=None;self.calls=0
 def __call__(self,input_ids,scores,**kwargs):
  self.calls+=1
  if self.first_token_ns is None:
   self.synchronize();self.first_token_ns=self.clock();self.first_length=int(input_ids.shape[1])
  return False

def measure(run,synchronize,clock=time.perf_counter_ns):
 """run(boundary) must do image read, preprocess and model generation.

 Only 3 explicit synchronizations: before T0, at first chosen token, and end.
 Profiling logits-ready and components is intentionally separate.
 """
 synchronize();start=clock();boundary=FirstTokenBoundary(synchronize,clock);result=run(boundary);synchronize();end=clock()
 if boundary.first_token_ns is None or not(start<=boundary.first_token_ns<=end):raise ValueError('first chosen token not observed')
 return result,{'t0_before_image_read_ns':start,'first_token_selected_ns':boundary.first_token_ns,'generation_complete_ns':end,'logits_ready_ns':None,'explicit_sync_boundaries':['before_t0','first_token','generation_end'],'first_criteria_length':boundary.first_length,'criteria_calls':boundary.calls,'clock_implementation':time.get_clock_info('perf_counter').implementation,'time_namespace':os.readlink('/proc/self/ns/time')}
