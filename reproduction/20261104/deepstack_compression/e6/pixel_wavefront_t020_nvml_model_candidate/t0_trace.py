"""Observe the actual author t0 fast path without altering its returned indices."""
from contextlib import contextmanager
@contextmanager
def observe(selector_class,expected):
 original=selector_class._select_2d_loco_fast;seen=[]
 def selected(self,tokens,height,width,device):
  if self.method!='max' or self.threshold!=0 or self.anchored is not True:raise ValueError('wrong t0 selector')
  value=original(self,tokens,height,width,device);snapshot=value.detach().clone();seen.append((value,snapshot));return value
 selector_class._select_2d_loco_fast=selected
 try:
  yield seen
  if len(seen)!=1:raise ValueError('t0 selector call count')
  value,snapshot=seen[0]
  if not value.equal(snapshot) or snapshot.cpu().tolist()!=expected:raise ValueError('t0 selector identity')
 finally:selector_class._select_2d_loco_fast=original
