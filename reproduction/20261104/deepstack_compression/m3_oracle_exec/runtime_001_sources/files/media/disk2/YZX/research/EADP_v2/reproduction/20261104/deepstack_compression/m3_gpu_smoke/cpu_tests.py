"""Load each synthetic test by its exact path, immune to E4 sys.path collisions."""
import importlib.util
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
suite=unittest.TestSuite()
for name in ('test_launcher','test_contract','test_core','test_gate'):
    path=HERE/(name+'.py');spec=importlib.util.spec_from_file_location('m3_exact_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
r=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() else 1)
