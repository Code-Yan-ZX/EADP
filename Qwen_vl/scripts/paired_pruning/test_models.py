"""Small CPU-only contract/mechanism tests; no VLM, real data, or GPU required."""
from __future__ import annotations

import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .models import PruningScorer, load_checkpoint, select_keep
from .train import (atomic_torch, finite_bank_log_probs, matched_page_indices,
                    paired_utility_loss, set_question_loss, sha256)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.config = dict(feature_dim=12, streams=4, hidden_dim=16,
                           num_heads=4, mode_heads=4, max_question_tokens=12)
        self.features = torch.randn(2, 4, 16, 12)
        self.query = torch.randn(2, 5, 12)
        self.mask = torch.ones(2, 5, dtype=torch.bool)

    def test_bag_permutation_and_ordered_sensitivity(self):
        permutation = torch.tensor([4, 2, 0, 1, 3])
        bag = PruningScorer(dict(self.config, ordered=False)).eval()
        ordered = PruningScorer(dict(self.config, ordered=True)).eval()
        ordered.load_state_dict(bag.state_dict())
        with torch.no_grad():
            a = bag(self.features, self.query, self.mask)
            b = bag(self.features, self.query[:, permutation], self.mask)
            torch.testing.assert_close(a["scores"], b["scores"], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(a["gate_logits"], b["gate_logits"], atol=2e-6, rtol=2e-6)
            c = ordered(self.features, self.query, self.mask)
            d = ordered(self.features, self.query[:, permutation], self.mask)
            self.assertGreater(float((c["scores"]-d["scores"]).abs().max()), 1e-5)
        self.assertEqual(sum(p.numel() for p in bag.parameters()), sum(p.numel() for p in ordered.parameters()))

    def test_padding_invariance_both_modes_and_left_padding(self):
        for is_ordered in (False, True):
            model = PruningScorer(dict(self.config, ordered=is_ordered)).eval()
            padded = torch.cat((self.query, torch.randn(2, 3, 12)*100), dim=1)
            mask = torch.cat((self.mask, torch.zeros(2, 3, dtype=torch.bool)), dim=1)
            with torch.no_grad():
                short = model(self.features, self.query, self.mask)
                right = model(self.features, padded, mask)
                left = model(self.features, torch.roll(padded, 3, dims=1), torch.roll(mask, 3, dims=1))
            for key in short:
                torch.testing.assert_close(short[key], right[key], atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(short[key], left[key], atol=2e-6, rtol=2e-6)

    def test_exact_k_gate_selects_whole_mode_not_average(self):
        scores = torch.tensor([[[9., 8., 0., 0.], [0., 0., 9., 8.]]])
        output = {"scores": scores, "gate_logits": torch.tensor([[0., 2.]])}
        self.assertEqual(select_keep(output, 2).tolist(), [[2, 3]])
        output = {"scores": torch.zeros(2, 4, 16), "gate_logits": torch.zeros(2, 4)}
        self.assertEqual(select_keep(output, 4).tolist(), [[0, 1, 2, 3], [0, 1, 2, 3]])

    def test_save_reload_identity(self):
        model = PruningScorer(self.config).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"selected.pt"
            metadata = {"arm": "multimode", "seed": 11}
            atomic_torch(path, dict(format_version=1, kind="selected", config=model.config,
                                    state_dict=model.state_dict(), metadata=metadata))
            loaded, restored = load_checkpoint(path)
            self.assertEqual(metadata, restored)
            with torch.no_grad():
                before = model(self.features, self.query, self.mask)
                after = loaded(self.features, self.query, self.mask)
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)

    def test_reject_empty_or_overlength_question(self):
        model = PruningScorer(self.config)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(self.features, self.query, torch.zeros_like(self.mask))
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            model(self.features, torch.zeros(2, 13, 12), torch.ones(2, 13, dtype=torch.bool))


class LossTests(unittest.TestCase):
    def setUp(self):
        self.masks = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]], dtype=torch.bool)
        self.success = torch.tensor([True, True, False])
        self.teacher = torch.tensor([4., 3., 2., 1.])
        self.nll = torch.tensor([0.3, 0.1, 2.])

    def test_pair_alignment_and_page_isolation(self):
        torch.manual_seed(1)
        score = torch.randn(4, 8)
        teacher = torch.rand(4, 8)
        original = paired_utility_loss(score, teacher, ["a", "a", "b", "b"])
        perm = torch.tensor([3, 1, 2, 0])
        permuted = paired_utility_loss(score[perm], teacher[perm], ["b", "a", "b", "a"])
        torch.testing.assert_close(original, permuted)
        wrong = paired_utility_loss(score, teacher.flip(0), ["a", "a", "b", "b"])
        self.assertGreater(abs(float(original-wrong)), 1e-5)
        self.assertEqual(float(paired_utility_loss(score, teacher, list(range(4)))), 0.)

    def test_multimode_rewards_complete_successes_over_marginal_hybrid(self):
        good = torch.tensor([[8., 8., -8., -8.], [-8., -8., 8., 8.]], requires_grad=True)
        collapsed = torch.tensor([[8., -8., 8., -8.], [8., -8., 8., -8.]], requires_grad=True)
        gates = torch.zeros(2, requires_grad=True)
        good_loss, _ = set_question_loss(good, gates, self.teacher, self.masks, self.success, self.nll, "multimode", 2)
        bad_loss, _ = set_question_loss(collapsed, gates, self.teacher, self.masks, self.success, self.nll, "multimode", 2)
        self.assertLess(float(good_loss.detach()), float(bad_loss.detach()))
        logp, _ = finite_bank_log_probs(good, gates, self.masks)
        torch.testing.assert_close(logp.exp().sum(), torch.tensor(1.))
        self.assertLess(float(logp[-1].detach().exp()), 0.01)
        bad_loss.backward()
        self.assertTrue(torch.isfinite(collapsed.grad).all())
        self.assertGreater(float(collapsed.grad.abs().sum()), 0.)

    def test_no_success_same_fallback_all_arms(self):
        scores = torch.randn(4, 4)
        gates = torch.randn(4)
        outcomes = [set_question_loss(scores, gates, self.teacher, self.masks,
                    torch.zeros(3, dtype=torch.bool), self.nll, arm, 2)
                    for arm in ("canonical", "token_marginal", "multimode")]
        for loss, info in outcomes:
            torch.testing.assert_close(loss, outcomes[0][0])
            self.assertEqual(info, {"fallback": 1, "success_masks": 0})

    def test_canonical_ignores_lower_nll_failure(self):
        scores = torch.randn(4, 4)
        gates = torch.zeros(4)
        good, _ = set_question_loss(scores, gates, self.teacher, self.masks, self.success, self.nll, "canonical", 2)
        nll = self.nll.clone(); nll[-1] = -100
        unchanged, _ = set_question_loss(scores, gates, self.teacher, self.masks, self.success, nll, "canonical", 2)
        torch.testing.assert_close(good, unchanged)

    def test_all_success_still_has_mask_learning_gradient(self):
        scores = torch.randn(4, 4, requires_grad=True)
        loss, _ = set_question_loss(scores, torch.zeros(4), self.teacher, self.masks,
                                  torch.ones(3, dtype=torch.bool), self.nll, "multimode", 2)
        loss.backward()
        self.assertGreater(float(scores.grad.abs().sum()), 0.)

    def test_single_success_trains_gate_and_every_unassigned_head(self):
        # C_success=1 < H=4; there are no failed bank candidates at all.
        scores = torch.tensor([[8., 8., -8., -8.], [-8., -8., 8., 8.],
                               [1., -1., 1., -1.], [0., 0., 0., 0.]], requires_grad=True)
        gates = torch.zeros(4, requires_grad=True)
        loss, _ = set_question_loss(scores, gates, self.teacher, self.masks[:1],
            torch.ones(1, dtype=torch.bool), self.nll[:1], "multimode", 2)
        loss.backward()
        self.assertLess(float(gates.grad[0]), 0.)
        self.assertTrue(bool((gates.grad[1:] > 0).all()))
        self.assertTrue(bool((scores.grad.abs().sum(-1) > 0).all()))

    def test_conflicting_duplicate_mask_rejected(self):
        masks = torch.stack((self.masks[0], self.masks[0]))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            set_question_loss(torch.zeros(4, 4), torch.zeros(4), self.teacher, masks,
                torch.tensor([True, False]), self.nll[:2], "multimode", 2)

    def test_matched_page_batches_resume_by_step(self):
        first = [matched_page_indices(20, 2, 11, step) for step in range(1, 6)]
        resumed = [matched_page_indices(20, 2, 11, step) for step in range(4, 6)]
        self.assertEqual(first[3:], resumed)
        self.assertNotEqual(first, [matched_page_indices(20, 2, 23, step) for step in range(1, 6)])

    def test_tiny_cpu_all_arms_transaction_resume_and_completed_idempotence(self):
        """Two synthetic updates/arm; interruption after resume commit, before export.

        No real data, VLM, or CUDA is used. Invalid manifest JSON deliberately
        proves the trainer hashes its bytes without parsing held-out answers.
        """
        from . import train
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/"data").mkdir(); (run/"cache").mkdir()
            manifest = run/"data/manifest.json"
            manifest.write_text("training must not parse held-out answers here")
            protocol = dict(steps=2, seeds=[11], n=8, k=2, val_every=1, batch_pages=1,
                split_pages={"fit": 1, "val": 1}, questions_per_page=2,
                source_sha256={}, manifest_sha256=sha256(manifest),
                model_config=dict(feature_dim=8, streams=4, hidden_dim=8, num_heads=2,
                                  mode_heads=4, max_question_tokens=8))
            (run/"protocol.json").write_text(json.dumps(protocol))
            digest = sha256(run/"protocol.json")
            entries = {}
            for role in ("fit", "val"):
                masks = torch.zeros(3, 8, dtype=torch.bool)
                masks[0, :2] = True; masks[1, 2:4] = True; masks[2, 4:6] = True
                rows = [dict(item_id=f"{role}_{j}", question_embeddings=torch.randn(3+j, 8).bfloat16(),
                    teacher_scores=torch.rand(8), bank_masks=masks,
                    bank_success=torch.tensor([True, True, False]), bank_nll=torch.tensor([.3, .2, 2.]),
                    bank_anls=torch.tensor([1., 1., 0.]), bank_names=["a", "b", "c"]) for j in range(2)]
                path = run/"cache"/f"{role}.pt"
                atomic_torch(path, dict(page_id=role, parent_id=role+"_parent", role=role,
                    protocol_sha256=digest, features=torch.randn(4, 8, 8).bfloat16(), questions=rows))
                entries[role] = dict(path=str(path), sha256=sha256(path), role=role, questions=2)
            (run/"cache/index.json").write_text(json.dumps(dict(complete=True, protocol_sha256=digest, pages=entries)))
            arguments = ["--run-dir", str(run), "--device", "cpu"]
            with patch.object(train, "materialize_selected", side_effect=RuntimeError("simulated interruption after commit")):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train.main(arguments+["--experiment", "query"])
            self.assertTrue((run/"checkpoints/bag_rank__seed_11.resume.pt").is_file())
            self.assertFalse((run/"checkpoints/bag_rank__seed_11.pt").exists())
            for experiment in ("query", "sets"):
                train.main(arguments+["--experiment", experiment])
                summary_path = run/"checkpoints"/f"{experiment}_summary.json"
                original_bytes = summary_path.read_bytes()
                summary = json.loads(original_bytes)
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(len(summary["checkpoints"]), 3)
                for row in summary["checkpoints"]:
                    model, metadata = load_checkpoint(row["path"])
                    self.assertEqual(metadata["arm"], row["arm"])
                    self.assertEqual(model.config["mode_heads"], 1 if experiment == "query" else 4)
                    self.assertEqual(sha256(row["path"]), row["sha256"])
                train.main(arguments+["--experiment", experiment])
                self.assertEqual(original_bytes, summary_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
