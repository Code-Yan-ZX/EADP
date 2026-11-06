"""Observe actual post-forward 36-layer KV content fingerprints, not full KV bytes."""
from pathlib import Path
import sys
E4=Path(__file__).resolve().parent.parent/'e4'
sys.path.insert(0,str(E4))
import identity_records
if Path(identity_records.__file__).resolve()!=(E4/'identity_records.py').resolve():raise ImportError('unexpected tensor metadata source')
tensor_meta=identity_records.tensor_meta

class CacheFingerprints:
    def __init__(self,wrapper,enabled):self.wrapper=wrapper;self.enabled=enabled;self.steps=[];self.handle=None
    def __enter__(self):
        if self.enabled:
            def capture(_module,_args,_kwargs,out):
                cache=out.past_key_values
                if cache is None or len(cache.layers)!=36:raise ValueError('missing 36-layer KV cache')
                self.steps.append([{'key':tensor_meta(layer.keys),'value':tensor_meta(layer.values)} for layer in cache.layers])
            self.handle=self.wrapper.model.model.language_model.register_forward_hook(capture,with_kwargs=True)
        return self
    def __exit__(self,*args):
        if self.handle is not None:self.handle.remove()
