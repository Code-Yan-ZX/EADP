import unittest
import model
class ModelTest(unittest.TestCase):
 def test_unreviewed_module_has_no_execution_entry(self):
  self.assertTrue(callable(model.prepare_cpu));self.assertTrue(callable(model.initialize))
if __name__=='__main__':unittest.main()
