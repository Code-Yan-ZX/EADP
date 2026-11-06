"""Diagnostic capture with exact hook/monkeypatch restoration evidence."""
from contextlib import ExitStack

def hook_inventory(modules):
 names=('_forward_pre_hooks','_forward_hooks','_forward_pre_hooks_with_kwargs','_forward_hooks_with_kwargs','_forward_hooks_always_called')
 return [(m,{n:dict(getattr(m,n,{})) for n in names}) for m in modules]
def hooks_equal(snapshot):
 return all(all(dict(getattr(m,n,{}))==value for n,value in saved.items()) for m,saved in snapshot)
def instance_attr(obj,name):return (name in obj.__dict__,obj.__dict__.get(name))
def restored(obj,name,before):return instance_attr(obj,name)==before

def capture_one(torch,wrapper,native,row,author_generate,Adapter,Capture,clone,journal):
 modules=list(wrapper.model.modules())+list(wrapper.pruner.modules());seen=set();modules=[m for m in modules if not (id(m) in seen or seen.add(id(m)))];hooks=hook_inventory(modules)
 model=wrapper.model;before_generate=instance_attr(model,'generate');before_processor=instance_attr(wrapper,'_processor_inputs');before_greedy=native._greed_select_impl;generate=model.generate;actual={};success=False
 def once(*args,**kwargs):
  kwargs['max_new_tokens']=1;out=generate(*args,**kwargs);journal({'status':'capture_generated','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'output_ids':out.detach().cpu().tolist(),'scope':'diagnostic one-token exposure, not timing'});return out
 model.generate=once
 try:
  with torch.inference_mode():
   with Adapter(torch,wrapper,native,'C-ON',row['image_start']) as adapter:
    with Capture(torch,wrapper,adapter) as capture:
     with ExitStack() as extra:
      def vision(_m,_a,out):
       if 'visual' in actual:raise ValueError('visual forward repeated')
       actual['visual']=clone(torch,out)
      def prefill(_m,_a,kw):
       if 'prefill' in actual:return
       actual['prefill']=clone(torch,{k:kw.get(k) for k in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')})
       actual['rope_delta']=clone(torch,wrapper.model.model.rope_deltas)
      extra.callback(wrapper.model.visual.register_forward_hook(vision).remove)
      extra.callback(wrapper.model.model.language_model.register_forward_pre_hook(prefill,with_kwargs=True).remove)
      author_generate(wrapper,row['message'],dataset=row['task'])
     if not adapter.state['prefill_seen'] or set(actual)!={'visual','prefill','rope_delta'}:raise ValueError('missing actual diagnostic capture')
     actual['processor']=clone(torch,dict(adapter.state['inputs']));actual['ordered']=clone(torch,adapter.state['ordered'])
   if not hooks_equal(hooks):raise ValueError('hooks still present after capture')
   operations=capture.prepare_operations(row['image_start']);state=capture.state
   refs={'selector':state['selector']['result'],'ds_merger_three_streams':[state['merger'+str(i)]['result'] for i in range(3)],'ds_injection':(state['injection']['mask'],state['injection']['streams'])}
   success=True
 finally:
  if before_generate[0]:model.generate=before_generate[1]
  else:delattr(model,'generate')
  clean=hooks_equal(hooks) and restored(model,'generate',before_generate) and restored(wrapper,'_processor_inputs',before_processor) and native._greed_select_impl is before_greedy
  journal({'status':'capture_cleanup','passed':bool(clean),'capture_completed':success})
  if not clean:raise RuntimeError('capture hooks or patches were not restored')
 return operations,refs,actual,lambda:hooks_equal(hooks) and restored(model,'generate',before_generate) and restored(wrapper,'_processor_inputs',before_processor) and native._greed_select_impl is before_greedy
