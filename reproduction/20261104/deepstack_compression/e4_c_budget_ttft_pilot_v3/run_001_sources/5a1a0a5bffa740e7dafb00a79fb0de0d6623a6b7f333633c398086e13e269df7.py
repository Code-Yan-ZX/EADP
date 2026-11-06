import hashlib,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import generation_identity as gen
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Tests(unittest.TestCase):
 def test_actual_generation_sources_resolve_without_gpu(self):
  paths=gen.actual_paths();self.assertTrue(any(x.endswith('/transformers/generation/utils.py') for x in paths));self.assertTrue(any(x.endswith('/transformers/generation/stopping_criteria.py') for x in paths))
  self.assertTrue(any(x.endswith('/transformers/models/qwen3_vl/modeling_qwen3_vl.py') for x in paths))
 def test_wrong_origin_and_modified_source_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source.py';p.write_text('original');expected={str(p):sha(p)}
   with patch.object(gen,'actual_paths',return_value=[str(p)]):
    gen.validate(expected,sha);p.write_text('changed')
    with self.assertRaisesRegex(ValueError,'hash mismatch'):gen.validate(expected,sha)
   with patch.object(gen,'actual_paths',return_value=[str(Path(d)/'wrong')]):
    with self.assertRaisesRegex(ValueError,'origin mismatch'):gen.validate(expected,sha)
 def test_foreign_model_generation_method_rejected(self):
  with self.assertRaisesRegex(ValueError,'implementation changed'):gen.validate_model_class(types.SimpleNamespace(generate=lambda:None))
 def test_instance_override_rejected_before_underlying_call(self):
  from transformers.generation.utils import GenerationMixin
  class Model(GenerationMixin):pass
  model=Model();gen.validate_model_class(model);calls=[]
  model.generate=lambda **kw:calls.append(kw)
  with self.assertRaisesRegex(ValueError,'implementation changed'):gen.validate_model_class(model)
  self.assertEqual(calls,[])
