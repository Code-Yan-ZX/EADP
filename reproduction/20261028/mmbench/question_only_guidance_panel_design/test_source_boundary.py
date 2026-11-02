"""CPU AST gates; no model or GPU imports."""
import ast,copy,unittest
import source_boundary as s
class Tests(unittest.TestCase):
 def test_exact_real_source_and_same_wait(self):
  result=s.proof();self.assertTrue(result['full_restored_AST_equal']);self.assertTrue(result['same_prior_three_wait_AST']);self.assertFalse(result['model_loaded'])
 def test_duplicate_source_assignment_is_rejected(self):
  tree=ast.parse(s.SOURCE.read_bytes());f=s.bodyfn(tree);node=s.target(f);f.body.append(copy.deepcopy(node))
  with self.assertRaises(ValueError):s.transform(tree)
 def test_changed_native_assignment_rejected(self):
  tree=ast.parse(s.SOURCE.read_bytes());node=s.target(s.bodyfn(tree));node.value=ast.Constant('unexpected')
  with self.assertRaises(ValueError):s.transform(tree)
if __name__=='__main__':unittest.main()
