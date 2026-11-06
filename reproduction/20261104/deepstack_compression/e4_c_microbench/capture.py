"""Diagnostic-only real component inputs; never attach to main TTFT timing.

Capture runs separately, clones tensors outside measurement, and must later be
bound to independently accepted real-model identities before any GPU use.
"""
from contextlib import ExitStack

def clone_tree(torch,value):
 if torch.is_tensor(value):return value.detach().clone()
 if isinstance(value,tuple):return tuple(clone_tree(torch,x) for x in value)
 if isinstance(value,list):return [clone_tree(torch,x) for x in value]
 if isinstance(value,dict):return {k:clone_tree(torch,x) for k,x in value.items()}
 if value is None or isinstance(value,(str,int,float,bool)):return value
 raise TypeError('unsupported component input')

def equal_tree(torch,a,b):
 if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a,b)
 if type(a)!=type(b):return False
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(torch,x,y) for x,y in zip(a,b))
 if isinstance(a,dict):return set(a)==set(b) and all(equal_tree(torch,a[k],b[k]) for k in a)
 return a==b

class CaptureInputs:
 def __init__(self,torch,wrapper,adapter):self.t=torch;self.w=wrapper;self.adapter=adapter;self.state={};self.stack=None
 def __enter__(self):
  stack=ExitStack();self.stack=stack;s=self.state;t=self.t
  try:
   def pre(label):
    def hook(_m,args,kw):
     if label in s:raise ValueError('repeated component capture: '+label)
     s[label]={'args':clone_tree(t,args),'kwargs':clone_tree(t,kw)}
    return hook
   def post(label):
    def hook(_m,_args,_kw,out):
     if label not in s or 'result' in s[label]:raise ValueError('component capture order')
     s[label]['result']=clone_tree(t,out)
    return hook
   def attach(module,label):
    h=module.register_forward_pre_hook(pre(label),with_kwargs=True);stack.callback(h.remove)
    h=module.register_forward_hook(post(label),with_kwargs=True);stack.callback(h.remove)
   attach(self.w.pruner,'selector')
   mergers=list(self.w.model.visual.deepstack_merger_list)
   if len(mergers)!=3:raise ValueError('expected three DS mergers')
   for i,module in enumerate(mergers):attach(module,'merger'+str(i))
   def visual(_m,_args,out):
    if 'full_ds' in s or out[1] is None or len(out[1])!=3 or any(x is None for x in out[1]):raise ValueError('expected full DS outputs once')
    s['full_ds']=clone_tree(t,out[1])
   h=self.w.model.visual.register_forward_hook(visual);stack.callback(h.remove)
   def prefill(_m,_args,kw):
    if 'injection' in s:return
    mask=kw.get('visual_pos_masks');streams=kw.get('deepstack_visual_embeds')
    if mask is None or streams is None or len(streams)!=3:raise ValueError('expected actual DS injection')
    s['injection']={'attention_mask':clone_tree(t,kw['attention_mask']),'mask':clone_tree(t,mask),'streams':clone_tree(t,streams)}
   h=self.w.model.model.language_model.register_forward_pre_hook(prefill,with_kwargs=True);stack.callback(h.remove)
   return self
  except BaseException:stack.close();self.stack=None;raise
 def __exit__(self,typ,value,tb):
  try:return self.stack.__exit__(typ,value,tb)
  finally:self.stack=None

 def prepare_operations(self,image_start):
  """Validate replay equality once, then return uninstrumented component calls."""
  if self.stack is not None:raise ValueError('remove diagnostic hooks before replay')
  t=self.t;s=self.state;spatial=self.adapter.state['ordered'].detach().sort().values
  if spatial.numel()!=256 or any(x not in s for x in ('selector','merger0','merger1','merger2','full_ds','injection')):raise ValueError('capture incomplete')
  def selector():return self.w.pruner(*s['selector']['args'],**s['selector']['kwargs'])
  mergers=list(self.w.model.visual.deepstack_merger_list)
  def ds_merger():return [m(*s['merger'+str(i)]['args'],**s['merger'+str(i)]['kwargs']) for i,m in enumerate(mergers)]
  def injection():
   mask=t.zeros_like(s['injection']['attention_mask'],dtype=t.bool);mask[:,image_start:image_start+256]=True
   return mask,[x.index_select(0,spatial) for x in s['full_ds']]
  if not equal_tree(t,selector(),s['selector']['result']):raise ValueError('selector replay differs from actual')
  if not equal_tree(t,ds_merger(),[s['merger'+str(i)]['result'] for i in range(3)]):raise ValueError('DS merger replay differs from actual')
  if not equal_tree(t,injection(),(s['injection']['mask'],s['injection']['streams'])):raise ValueError('DS injection replay differs from actual')
  return {'selector':selector,'ds_merger_three_streams':ds_merger,'ds_injection':injection}
