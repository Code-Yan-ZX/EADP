"""Injected real-model timing continuation; no CLI and no GPU authorization."""
from contextlib import contextmanager
import importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def load(name,path):
 p=Path(path).resolve();m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign cached timing module')
  return m
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
timer=load('_pixel_t020_timing_boundaries',D/'t020_timing_boundaries.py')
observe=load('_pixel_t020_selector_observer',D/'selector_instrumentation.py').observe
@contextmanager
def timed_measure(base):
 old=base.measure;base.measure=timer.measure
 try:yield
 finally:base.measure=old

def execute(base,selector_class,candidate,arm,wrapper,patch,torch,row,synchronize,reference,emit_raw,validate_input):
 if arm not in ('author','wavefront') or not callable(validate_input) or not callable(emit_raw):raise ValueError('timed arm contract')
 # Both arms use identical Python observation and an on-device index clone;
 # the host copy happens after timing. Underlying selector may synchronize.
 with observe(selector_class,candidate,arm,reference['selected_indices']) as selected:
  with timed_measure(base):
   result=base.execute_pixel(wrapper,patch,torch,row,synchronize,reference,emit_raw,observe_count=True,validate_input=validate_input)
 result['selected_indices']=selected[0][1].cpu().tolist();result['arm']=arm;result['selection_observer']='same Python dispatch and on-device index clone on both arms; indices copied after generation';result['timing_scope']='one measured generation from before image read to synchronized selected first token; completion includes output persistence';return result
if __name__=='__main__':raise SystemExit('GPU locked: reviewed controller and preflight not implemented')
