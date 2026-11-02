"""Read-only completed-source panel contract; no model or scoring."""
import unittest
from panel_evidence import check
class Tests(unittest.TestCase):
 def test_complete_group_identity_and_history_boundary(self):
  r=check();self.assertEqual((r['n_groups'],r['n_rows']),(16,60));self.assertEqual(r['old_literal_answers_available'],60)
  self.assertEqual(r['source_text_hash_matches'],60);self.assertFalse(r['historical_output_ids_or_masks_available'])
  self.assertFalse(r['GT_columns_loaded']);self.assertFalse(r['model_loaded']);self.assertFalse(r['scoring_started'])
if __name__=='__main__':unittest.main()
