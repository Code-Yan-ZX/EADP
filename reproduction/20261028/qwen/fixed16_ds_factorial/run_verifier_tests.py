"""Run bounded synthetic verifier tests and bind success to plan."""
import sys,unittest
import common as c
import test_verify_terminal
suite=unittest.defaultTestLoader.loadTestsFromModule(test_verify_terminal)
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert 'torch' not in sys.modules
c.save(c.HERE/'cpu_verifier_tests.json',dict(success=result.wasSuccessful(),n_tests=result.testsRun,
 plan_sha256=c.sha(c.PLAN),CPU_only=True,model_loaded=False,GPU_started=False,GT_read=False,scoring_started=False))
sys.exit(0 if result.wasSuccessful() else 1)
