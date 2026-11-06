"""Read-only Qwen3 visual instrumentation prototype; never patches model output."""
import torch
from local_adapter import group_premerger,user_question

class VisualCapture:
 def __init__(self,visual,grid_thw):
  self.visual=visual;self.grid_thw=tuple(grid_thw);self.state={};self.handles=[];self.closed=False
 def __enter__(self):
  if self.handles or self.closed:raise ValueError('capture context cannot be reused')
  if getattr(self.visual,'spatial_merge_size',None)!=2 or not hasattr(self.visual,'merger'):raise ValueError('unsupported vision architecture')
  try:
   self.handles.append(self.visual.merger.register_forward_pre_hook(self._pre))
   self.handles.append(self.visual.register_forward_hook(self._post))
  except BaseException:
   for h in reversed(self.handles):h.remove()
   self.handles.clear();self.closed=True
   raise
  return self
 def _pre(self,_module,args):
  if 'premerger' in self.state or len(args)!=1 or not torch.is_tensor(args[0]):raise ValueError('missing or duplicate main merger input')
  self.state['premerger']=args[0].detach().clone()
 def _post(self,_module,_args,output):
  if 'postmerger' in self.state or not isinstance(output,(tuple,list)) or len(output)!=2:raise ValueError('unexpected or duplicate visual output')
  main,ds=output
  if not torch.is_tensor(main) or not isinstance(ds,(tuple,list)) or len(ds)!=3 or any(not torch.is_tensor(x) for x in ds):raise ValueError('expected main and three DS streams')
  self.state['postmerger']=main.detach().clone();self.state['deepstack']=[x.detach().clone() for x in ds]
 def result(self):
  if 'premerger' not in self.state or 'postmerger' not in self.state:raise ValueError('visual capture incomplete')
  averaged=group_premerger(self.state['premerger'],self.grid_thw);main=self.state['postmerger'];ds=self.state['deepstack']
  if main.ndim!=2 or main.shape[0]!=averaged.shape[0] or any(x.shape!=main.shape for x in ds):raise ValueError('pre/post/DS row misalignment')
  return {'premerger':self.state['premerger'],'averaged_premerger':averaged,'postmerger':main,'deepstack':ds}
 def __exit__(self,_type,_value,_trace):
  for h in reversed(self.handles):h.remove()
  self.handles.clear();self.closed=True

def encode_question(tokenizer,embed_tokens,message,device):
 text=user_question(message);enc=tokenizer(text.split(),is_split_into_words=True,return_tensors='pt',padding=True,truncation=True);ids=enc['input_ids'].to(device)
 if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1:raise ValueError('unsupported text tokens')
 with torch.no_grad():emb=embed_tokens(ids)[0]
 start=1 if ids.shape[1]>1 and tokenizer.bos_token_id is not None and ids[0,0].item()==tokenizer.bos_token_id else 0
 return {'text':text,'input_ids':ids,'embeddings':emb[start:]}
