"""Best-effort observation: failed or repeated hooks never alter native return."""
class Recorder:
    def __init__(self):
        self.events={};self.errors=[]
    def record(self,name,producer):
        try:
            value=producer()
        except Exception as e:
            self.errors.append(dict(event=name,error=repr(e)))
            return
        self.events.setdefault(name,[]).append(value)
    def first(self,name,default=None):
        values=self.events.get(name,[])
        return values[0] if values else default
    def count(self,name):return len(self.events.get(name,[]))

def passthrough(recorder,name,function,*args,**kwargs):
    value=function(*args,**kwargs)
    recorder.record(name,lambda:value)
    return value

def validate_capture(arm,counts,errors,native_failure,response,metadata,splits,kwargs):
    required={'visual_features','prefill_inputs_embeds','prefill_position_ids','prefill_attention_mask',
        'processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw','output_ids','output_suffix_ids'}
    assert native_failure is None and isinstance(response,str) and not errors
    assert all(counts.get(k,0)==1 for k in ('vision','prefill','generate_result','processor_inputs'))
    assert required <= set(metadata)
    if arm=='eadp1024':
        assert counts.get('pruner',0)==1 and splits==[1024]
        assert {'pruner_input','pruner_output','generate_inputs_embeds'}<=set(metadata)
    else:assert arm=='plain' and counts.get('pruner',0)==0 and 'input_ids' in kwargs
