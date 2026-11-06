import unittest
from unittest.mock import patch
from types import SimpleNamespace
from worker_preflight import LATENCY,validate_latency_contract,prepare_memory

class ImportResolutionTests(unittest.TestCase):
 def test_only_exact_latency_contract_is_accepted(self):
  validate_latency_contract(SimpleNamespace(__file__=str(LATENCY/'contract.py')))
  with self.assertRaisesRegex(ValueError,'wrong contract module'):validate_latency_contract(SimpleNamespace(__file__='/tmp/contract.py'))
 def test_actual_preflight_rejects_already_loaded_wrong_contract_before_identity(self):
  import sys
  wrong=SimpleNamespace(__file__='/tmp/contract.py');old=list(sys.path)
  try:
   with patch.dict('sys.modules',{'contract':wrong}):
    with self.assertRaisesRegex(ValueError,'wrong contract module'):prepare_memory('smoke')
  finally:sys.path[:]=old
if __name__=='__main__':unittest.main()
