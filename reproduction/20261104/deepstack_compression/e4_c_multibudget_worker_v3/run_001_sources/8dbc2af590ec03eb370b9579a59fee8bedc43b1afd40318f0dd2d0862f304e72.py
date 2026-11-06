"""Bind actual first-token generation, criterion and model source files."""
import inspect
from pathlib import Path
def actual_paths():
 from transformers.generation.utils import GenerationMixin
 from transformers.generation.stopping_criteria import StoppingCriteriaList
 from transformers.generation import configuration_utils,logits_process
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
 from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
 objects=(GenerationMixin.generate,inspect.unwrap(GenerationMixin.generate),StoppingCriteriaList.__call__,
          configuration_utils,logits_process,Qwen3VLForConditionalGeneration,Qwen3VLConfig)
 return sorted({str(Path(inspect.getfile(obj)).resolve()) for obj in objects})
def validate(expected,sha):
 paths=actual_paths()
 if set(paths)!=set(expected):raise ValueError('actual generation module origin mismatch')
 if any(sha(path)!=expected[path] for path in paths):raise ValueError('generation source hash mismatch')
 return paths
def validate_model_class(model):
 from transformers.generation.utils import GenerationMixin
 method=getattr(type(model),'generate',None)
 actual=getattr(model,'generate',None);actual_func=getattr(actual,'__func__',None)
 if (method is None or 'generate' in vars(model) or actual_func is None or getattr(actual,'__self__',None) is not model or
     inspect.unwrap(method) is not inspect.unwrap(GenerationMixin.generate) or inspect.unwrap(actual_func) is not inspect.unwrap(GenerationMixin.generate)):
  raise ValueError('actual model generation implementation changed')
