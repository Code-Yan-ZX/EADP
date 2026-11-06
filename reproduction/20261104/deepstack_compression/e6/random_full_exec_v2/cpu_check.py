"""Only local synthetic/schema tests; no model or GPU."""
import unittest

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromNames(['test_core','test_verify_contract','test_gate'])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 raise SystemExit(not result.wasSuccessful())
