"""Candidate selector observation for timed identity; no GPU entry or model import.

Clones returned indices on-device in the measured path, then verifies against
the returned object and copies the snapshot to host only after generation.
Both arms incur the same Python dispatch and clone cost; inference tensors do
not provide a usable version counter. A later real
worker must separately prove the selector is the one called by actual patch.
"""
from contextlib import contextmanager
@contextmanager
def observe(selector_class,candidate,arm,expected_indices):
 if arm not in ('author','wavefront'):raise ValueError('unknown arm')
 original=selector_class._select_2d_loco_anchored;seen=[]
 def selected(self,tokens,height,width,device):
  if self.method!='max' or self.threshold!=.2 or self.anchored is not True:raise ValueError('wrong real selector')
  value=original(self,tokens,height,width,device) if arm=='author' else candidate(tokens,height,width,self.threshold)
  snapshot=value.detach().clone();seen.append((value,snapshot));return value
 selector_class._select_2d_loco_anchored=selected
 try:
  yield seen
  if len(seen)!=1:raise ValueError('selector call count')
  value,snapshot=seen[0]
  if not value.equal(snapshot):raise ValueError('selector indices mutated')
  actual=snapshot.cpu().tolist()
  if actual!=expected_indices:raise ValueError('selector indices differ from accepted model identity')
 finally:selector_class._select_2d_loco_anchored=original
