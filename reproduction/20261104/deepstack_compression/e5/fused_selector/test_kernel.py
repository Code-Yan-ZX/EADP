"""CPU validation of API/rejection paths; actual Triton needs GPU verification."""
import unittest
import torch
from kernel import fused_greedy,validate


class Tests(unittest.TestCase):
    def test_valid_layout_and_budget(self):
        validate(torch.eye(4),torch.ones(4),3)
    def test_invalid_layout_dtype_and_budget(self):
        for sim,w,k in [(torch.ones(3,4),torch.ones(3),2), (torch.ones(4,4,dtype=torch.float64),torch.ones(4),2),
                        (torch.ones(4,4),torch.ones(4),0), (torch.ones(4,4).t(),torch.ones(4),2)]:
            with self.assertRaises(ValueError):validate(sim,w,k)
    def test_refuses_cpu_execution(self):
        with self.assertRaisesRegex(ValueError,'CUDA required'):fused_greedy(torch.eye(4),torch.ones(4),2)


if __name__=='__main__':unittest.main(verbosity=2)
