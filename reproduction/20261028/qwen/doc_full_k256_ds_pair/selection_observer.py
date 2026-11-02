"""Scoped, passive observation of the author's actual greedy return values.

No replacement selection, changed arithmetic, modified return or extra call.
The wrapper is process-global; only use in a single-threaded diagnostic worker.
"""
class SelectionObserver:
 def __init__(self,native_module):
  self.module=native_module;self.original=None;self.calls=0;self.results=[];self.errors=[]
 def __enter__(self):
  if self.original is not None:raise RuntimeError('observer already installed')
  self.original=self.module._greed_select_impl
  def observe(*args,**kwargs):
   actual=self.original(*args,**kwargs)
   self.calls+=1
   try:
    ids,quotas=actual
    self.results.append(dict(ordered_ids=ids.detach().clone(),quotas=quotas.detach().clone()))
   except Exception as error:self.errors.append(repr(error))
   return actual
  self.observe=observe;self.module._greed_select_impl=observe
  return self
 def __exit__(self,typ,value,traceback):
  if self.module._greed_select_impl is not self.observe:
   self.errors.append('global greedy binding unexpectedly replaced')
  self.module._greed_select_impl=self.original
  return False
