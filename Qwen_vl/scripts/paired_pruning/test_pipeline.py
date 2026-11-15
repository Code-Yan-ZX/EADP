"""CPU checks for the candidate, label, intervention and launcher contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from .artifacts import atomic_json, atomic_torch_save, sha256
from .build_cache import candidate_bank, mask_from_keep, sufficient, seed_for
from .diagnostics import crossed_mask, jaccard
from . import runtime


class PipelineTests(unittest.TestCase):
    def test_candidates_are_distinct_fixed_budget_and_reproducible(self):
        n, k = 64, 16
        scores = torch.linspace(.1, 2, n)
        teacher = dict(scores={"magnitude_v":scores}, keep_indices={
            "magnitude_v":torch.arange(n-k,n), "magnitude_joint":torch.arange(n-k,n),
            "signed_v":torch.arange(k)})
        args = (teacher, torch.arange(8,8+k), torch.arange(k), n,k,12,17)
        a, b = candidate_bank(*args), candidate_bank(*args)
        self.assertEqual(len(a),12)
        self.assertEqual([name for name,_ in a],[name for name,_ in b])
        for (_,mask),(_,other) in zip(a,b):
            self.assertEqual(int(mask.sum()),k)
            self.assertTrue(torch.equal(mask,other))
        self.assertEqual(len({bytes(mask.tolist()) for _,mask in a}),12)

    def test_candidate_lookups_are_not_success_labels(self):
        full = {"anls":1.,"truncated":False}
        self.assertFalse(sufficient(full,{"anls":.89,"truncated":False}))
        self.assertFalse(sufficient(full,{"anls":1.,"truncated":True}))
        self.assertFalse(sufficient({"anls":.2,"truncated":False},{"anls":1.,"truncated":False}))
        self.assertTrue(sufficient(full,{"anls":1.,"truncated":False}))

    def test_mask_rejects_duplicates_and_bad_budget(self):
        with self.assertRaises(ValueError):
            mask_from_keep([1,1],8,2)
        with self.assertRaises(ValueError):
            mask_from_keep([1,8],8,2)

    def test_crossed_selection_preserves_common_members_and_budget(self):
        a,b = torch.zeros(12,dtype=torch.bool),torch.zeros(12,dtype=torch.bool)
        a[[0,1,2,3,4,5]]=True
        b[[0,1,6,7,8,9]]=True
        mixed=crossed_mask(a,b,19)
        self.assertEqual(int(mixed.sum()),6)
        self.assertTrue(bool(mixed[0] and mixed[1]))
        self.assertFalse(torch.equal(mixed,a))
        self.assertFalse(torch.equal(mixed,b))
        self.assertAlmostEqual(jaccard(a,b),.2)

    def test_frozen_json_never_replaces_existing_content(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"freeze.json"
            atomic_json(path,{"x":1},frozen=True)
            before=sha256(path)
            atomic_json(path,{"x":1},frozen=True)
            with self.assertRaises(RuntimeError):
                atomic_json(path,{"x":2},frozen=True)
            self.assertEqual(sha256(path),before)

    def test_atomic_tensor_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"cached.pt"
            value={"features":torch.arange(12).reshape(3,4),"role":"fit"}
            atomic_torch_save(path,value)
            actual=torch.load(path,weights_only=False)
            self.assertTrue(torch.equal(actual["features"],value["features"]))

    def test_runtime_scoring_matches_official_edge_cases(self):
        from .evaluate import official_anls
        cases=[("",[""]), ("a",["ab"]), ("  A  b ",["a b"]),
               ("12.5",["12.50","12.5"]), ("xyz",["no"])]
        for text,answers in cases:
            self.assertAlmostEqual(runtime.anls(text,answers),official_anls(text,answers))

    def test_seeds_are_outcome_independent(self):
        self.assertEqual(seed_for(1,"page","q"),seed_for(1,"page","q"))
        self.assertNotEqual(seed_for(1,"page","q"),seed_for(1,"page","other_q"))
        context=dict(prep={"n_vis":16},V=torch.zeros(16,2),source_image_sha256="same-source",
                     message=[{"type":"image","value":"/run_a/image.img"},
                              {"type":"text","value":"Which value?"}])
        before=runtime.baseline_keep(None,context,"random",4)
        context["message"][0]["value"]="/run_b/image.img"
        self.assertTrue(torch.equal(before,runtime.baseline_keep(None,context,"random",4)))

    def test_prepare_has_no_answer_dependency(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/"image"
            image.write_bytes(b"frozen source image fixture")
            visual=torch.randn(4,8)
            embedding=torch.nn.Embedding(256,8)
            eng=SimpleNamespace(
                prepare=lambda message,dataset:dict(n_vis=4),
                encode=lambda prep:(visual,[visual.clone() for _ in range(3)]),
                tok=lambda text,**kwargs:{"input_ids":torch.tensor([[ord(c)%256 for c in text]])},
                model=SimpleNamespace(get_input_embeddings=lambda:embedding))
            page=dict(image_path=str(image),image_sha256=sha256(image))
            before=runtime.prepare(eng,page,dict(question="Who pays?",answers=["A"]))
            after=runtime.prepare(eng,page,dict(question="Who pays?",answers=["B"],teacher_scores="poison"))
            self.assertTrue(torch.equal(before["question_embeddings"],after["question_embeddings"]))
            self.assertTrue(torch.equal(before["V"],after["V"]))
            self.assertEqual(before["message"],after["message"])

    def test_native_generation_timing_stops_before_posthoc_audit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        positions=torch.arange(6).view(1,1,6).expand(3,1,6)
        keep=torch.tensor([0,2])
        sequence=torch.tensor([0,1,3,5])
        state=SimpleNamespace(keep_idx=keep,keep_seq=sequence,
            positions_prefill=positions[:,:,sequence],ds_lengths=[2]*3,
            cache=SimpleNamespace(get_seq_length=lambda:4),lk=4,decode_positions=[],n_vis=2)
        eng=SimpleNamespace(model=SimpleNamespace(training=False),
            prefill=lambda *args,**kwargs:state,decode=lambda *args,**kwargs:([7],"ok"),
            keep_sequence=lambda *args:sequence,positions_full=lambda prep:(positions,None))
        ctx=dict(prep={"n_vis":3},V=torch.zeros(3,2),DS=[torch.zeros(3,2)]*3)
        with patch.object(runtime.time,"perf_counter",side_effect=[10.,11.,16.]):
            result=runtime.generate(eng,ctx,keep,max_tokens=1,ignore_eos=True)
        self.assertEqual(result["generation_ready_perf_counter"],11.)
        self.assertEqual(result["cached_generation_ms"],1000.)
        self.assertEqual(result["post_generation_audit_ms"],5000.)
        self.assertEqual(result["keep_idx"],[0,2])


if __name__ == "__main__":
    unittest.main()
