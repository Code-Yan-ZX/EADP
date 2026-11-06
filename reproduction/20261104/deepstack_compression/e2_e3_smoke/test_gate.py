import json,tempfile,unittest,sys,types
from pathlib import Path
from unittest.mock import patch
from local_imports import load_local
gate=load_local('gate')
class GateTests(unittest.TestCase):
    def test_local_modules_ignore_preloaded_same_names(self):
        fake=types.ModuleType('verify_smoke');fake.__file__='/tmp/other/verify_smoke.py'
        with patch.dict(sys.modules,{'verify_smoke':fake}):
            module=load_local('verify_smoke')
            self.assertEqual(Path(module.__file__).resolve(),gate.HERE/'verify_smoke.py')
            self.assertIsNot(module,fake)
    def test_unapproved_is_locked(self):
        with tempfile.TemporaryDirectory() as d,patch.object(gate,'APPROVAL',Path(d)/'missing'):
            for preflight in (False,True):
                with self.assertRaisesRegex(ValueError,'approval is absent'):gate.check_approval(preflight)
    def test_changed_source_and_missing_preflight_fail(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'source.py';source.write_text('a');digest=gate.sha(source)
            cpu=d/'cpu.json';cpu.write_text(json.dumps({'status':'passed','returncode':0,
                'source_sha256_before':{str(source):digest},'source_sha256_after':{str(source):digest}}))
            review=d/'review.json';review.write_text(json.dumps({'status':'passed','independent_review':True,
                'cpu_proof_sha256':gate.sha(cpu),'source_sha256':{str(source):digest}}))
            approval=d/'approval.json';approval.write_text(json.dumps({'cpu_proof':str(cpu),'review_proof':str(review)}))
            with patch.object(gate,'APPROVAL',approval),patch.object(gate,'SOURCES',[source]),patch.object(gate,'HERE',d):
                gate.check_approval(preflight=True)
                with self.assertRaisesRegex(ValueError,'preflight absent'):gate.check_approval()
                source.write_text('tampered')
                with self.assertRaisesRegex(ValueError,'unreviewed source'):gate.check_approval(preflight=True)
if __name__=='__main__':unittest.main()
