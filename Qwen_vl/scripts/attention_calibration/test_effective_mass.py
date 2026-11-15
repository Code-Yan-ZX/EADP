"""CPU checks against analytically identifiable synthetic clusters."""
import unittest

import torch

from effective_mass import cluster_weights


class EffectiveMassTest(unittest.TestCase):
    def test_known_clusters_and_raster_order(self):
        # Keep contains dominant 0 and two anchors 2 and 5.  Anchor 2 has
        # an identical and an orthogonal member; anchor 5 an opposite member.
        key = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [0., 1.],
                            [0., -1.], [0., 1.]])
        keep = torch.tensor([0, 2, 5])
        t = dict(anchor_idx=torch.tensor([2, 5]), assignee_idx=torch.tensor([1, 3, 4]),
                 assignment=torch.tensor([0, 0, 1]), cluster_sizes=torch.tensor([3, 2]))
        count, effective = cluster_weights(key, (keep, None, None, {"tensors": t}))
        torch.testing.assert_close(count, torch.tensor([1., 3., 2.]))
        torch.testing.assert_close(effective, torch.tensor([1., 2., 1.]))

    def test_equal_members_recover_counts(self):
        key = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
        keep = torch.tensor([0])
        t = dict(anchor_idx=torch.tensor([0]), assignee_idx=torch.tensor([1, 2]),
                 assignment=torch.tensor([0, 0]), cluster_sizes=torch.tensor([3]))
        count, effective = cluster_weights(key, (keep, None, None, {"tensors": t}))
        torch.testing.assert_close(count, effective)

    def test_fails_on_wrong_accounting(self):
        key = torch.ones(3, 2)
        t = dict(anchor_idx=torch.tensor([0]), assignee_idx=torch.tensor([1, 2]),
                 assignment=torch.tensor([0, 0]), cluster_sizes=torch.tensor([2]))
        with self.assertRaises(AssertionError):
            cluster_weights(key, (torch.tensor([0]), None, None, {"tensors": t}))


if __name__ == "__main__":
    unittest.main()
