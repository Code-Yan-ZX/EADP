"""Synchronous natural request boundaries; CPU contracts, no GPU authority."""
import time
class Boundary:
 def __init__(self,sync,clock):self.sync=sync;self.clock=clock;self.first_ns=None;self.first_length=None;self.calls=0;self.raw=None;self.generate_end_ns=None
 def __call__(self,input_ids,scores,**kwargs):
  self.calls+=1
  if self.first_ns is None:
   self.sync();self.first_ns=self.clock();self.first_length=int(input_ids.shape[1])
  return False
 def generation_returned(self,raw):
  if self.raw is not None or raw is None:raise ValueError('invalid or duplicate generation')
  self.raw=raw
  if self.first_ns is None:raise ValueError('missing first token')
  self.sync();self.generate_end_ns=self.clock()
  return raw
def measure(invoke,sync,clock=time.perf_counter_ns,persist=None):
 if not callable(persist):raise ValueError('durable raw output required')
 sync();start=clock();boundary=Boundary(sync,clock)
 try:
  response=invoke(boundary);sync();end=clock()
  if boundary.raw is None or not 0<start<boundary.first_ns<=boundary.generate_end_ns<=end:raise ValueError('invalid synchronized boundaries')
 except BaseException:
  if boundary.raw is not None:persist(boundary.raw)
  raise
 persist(boundary.raw)
 return response,{'t0_ns':start,'first_ns':boundary.first_ns,'generate_end_ns':boundary.generate_end_ns,'response_end_ns':end,'first_criteria_length':boundary.first_length,'criteria_calls':boundary.calls,'scope':'before image read; first token; generate return; decoded response return','raw_persisted_after_all_endpoints':True}
def normalize(arm,raw,prompt):
 if arm not in ('eadp','rres') or not isinstance(raw,list) or not raw or any(type(x) is not int for x in raw):raise ValueError('invalid generated ids')
 if arm=='rres':
  if not isinstance(prompt,list) or not prompt or any(type(x) is not int for x in prompt) or raw[:len(prompt)]!=prompt:raise ValueError('R-res prompt prefix mismatch')
  raw=raw[len(prompt):]
 if not raw:raise ValueError('empty generated continuation')
 return raw
