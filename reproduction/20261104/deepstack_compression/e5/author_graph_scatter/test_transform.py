import inspect,unittest
from unittest.mock import patch
import torch
import transform

class TestTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.author=transform.load_author();candidate,cls.metadata=transform.build(cls.author);cls.candidate=staticmethod(candidate)
    def compare(self,imp,sim,k):
        old,quota=self.author._greed_select_impl(imp,sim,k)
        new,newquota=self.candidate(imp,sim,k)
        self.assertTrue(torch.equal(old,new));self.assertTrue(torch.equal(quota,newquota))
    def test_ties_zeros_and_multi_batch(self):
        self.compare(torch.ones(3,8),torch.full((3,8,8),.5),8)
        self.compare(torch.zeros(2,8),torch.zeros(2,8,8),8)
    def test_seeded_and_near_ties(self):
        g=torch.Generator().manual_seed(20261105)
        self.compare(torch.rand(2,32,generator=g),torch.rand(2,32,32,generator=g),16)
        imp=torch.ones(1,8);imp[0,1]+=torch.finfo(torch.float32).eps
        self.compare(imp,torch.full((1,8,8),.5),8)
    def test_only_one_exact_statement_is_allowed(self):
        self.assertEqual(self.metadata['replacement_count'],1)
        self.assertIs(self.metadata['source_written'],False)
        source=inspect.getsource(self.author._greed_select_impl).replace('selected_mask[batch_indices, best_idx] = True','pass')
        with patch.object(transform.inspect,'getsource',return_value=source):
            with self.assertRaises(ValueError):transform.build(self.author)
        source=inspect.getsource(self.author._greed_select_impl).replace('selected_mask[batch_indices, best_idx] = True',
                'selected_mask[batch_indices, best_idx] = True\n        selected_mask[batch_indices, best_idx] = True')
        with patch.object(transform.inspect,'getsource',return_value=source):
            with self.assertRaises(ValueError):transform.build(self.author)

if __name__=='__main__':unittest.main(verbosity=2)
