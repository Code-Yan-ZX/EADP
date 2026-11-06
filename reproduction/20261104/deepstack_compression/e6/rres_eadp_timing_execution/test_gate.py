import unittest
import gpu_gate


class Tests(unittest.TestCase):
    def test_gpu_admission_locked_without_plan(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            gpu_gate.admit()
    def test_actual_three_anchor_evidence(self):
        digest = gpu_gate.require_identity()
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)


if __name__ == '__main__': unittest.main()
