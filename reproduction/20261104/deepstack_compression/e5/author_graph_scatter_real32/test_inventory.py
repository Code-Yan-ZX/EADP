import copy,unittest
from pathlib import Path
from unittest.mock import patch
import inventory

class TestInventory(unittest.TestCase):
    def test_actual_metadata_and_scatter_proof(self):
        rows=inventory.inventory(hash_files=False)
        self.assertEqual(len(rows),96);self.assertEqual({x['budget'] for x in rows},{128,256,512})
        self.assertEqual({x['image_index'] for x in rows},set(range(32)))
        self.assertEqual(inventory.require_scatter_review()['formal_sha256'],inventory.sha(inventory.SCATTER/'smoke001_verification.json'))
    def test_changed_first_archive_rejected(self):
        first=(inventory.REAL/'run_001/image_000/case_128.pt').resolve();real=inventory.sha
        def altered(path):return '0'*64 if Path(path).resolve()==first else real(path)
        with patch.object(inventory,'sha',side_effect=altered):
            with self.assertRaises(ValueError):inventory.inventory(hash_files=True)
    def test_missing_formal_case_rejected(self):
        real=inventory.read
        def altered(path):
            result=real(path)
            if Path(path)==inventory.FORMAL:
                result=copy.deepcopy(result);result['array_sha256'].pop(str((inventory.REAL/'run_001/image_000/case_128.pt').resolve()))
            return result
        with patch.object(inventory,'read',side_effect=altered):
            with self.assertRaises(ValueError):inventory.inventory(hash_files=False)
    def test_prior_scatter_failure_rejected(self):
        real=inventory.read
        def altered(path):
            result=real(path)
            if Path(path)==inventory.SREVIEW:
                result=copy.deepcopy(result)
                for x in result['cases']:
                    if x['case']=='k_512':x['outcome']='failed'
            return result
        with patch.object(inventory,'read',side_effect=altered):
            with self.assertRaises(ValueError):inventory.require_scatter_review()

if __name__=='__main__':unittest.main(verbosity=2)
