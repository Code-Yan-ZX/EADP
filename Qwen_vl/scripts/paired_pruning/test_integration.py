"""Synthetic CPU end-to-end integration; deliberately no VLM/GPU experiment."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from .artifacts import atomic_json, atomic_torch_save, sha256
from . import evaluate, report, runtime, train


class SyntheticRunTests(unittest.TestCase):
    def test_two_training_suites_evaluation_report_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.is_available", return_value=False):
            run = Path(folder)
            n, k, dim = 1024, 256, 8
            splits = {}
            for number, role in enumerate(("fit", "val", "test", "dev")):
                image = Image.new("RGB", (3, 3), (number*30, 10, 20))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_path = run / f"{role}.png"
                image_path.write_bytes(buffer.getvalue())
                splits[role] = [dict(page_id=role, parent_id="parent_"+role,
                    image_path=str(image_path), image_sha256=sha256(image_path),
                    pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    questions=[dict(item_id=f"{role}_{i}",question=f"Question {i}?", answers=["ok"],
                                    question_id=f"{role}_{i}",numeric=False) for i in range(2)])]
            atomic_json(run/"data/manifest.json", dict(schema_version=1,splits=splits))
            protocol = dict(n=n,k=k,steps=1,seeds=[11],val_every=1,batch_pages=1,
                timing_blocks=1,timing_warmup=0,max_new_tokens=128,max_question_tokens=16,
                split_pages={role:1 for role in splits},questions_per_page=2,
                model_config=dict(feature_dim=dim,streams=4,hidden_dim=8,num_heads=2,
                                  mode_heads=2,max_question_tokens=16),
                mode_heads=2,source_sha256={},model_metadata_sha256={},model_weight_stat={},
                manifest_sha256=sha256(run/"data/manifest.json"))
            atomic_json(run/"protocol.json",protocol)
            digest=sha256(run/"protocol.json")
            index=dict(complete=True,protocol_sha256=digest,pages={})
            generator=torch.Generator().manual_seed(23)
            for role in ("fit","val"):
                masks=torch.zeros(3,n,dtype=torch.bool)
                masks[0,:k]=True
                masks[1,k:2*k]=True
                masks[2,::4]=True
                questions=[]
                for item in splits[role][0]["questions"]:
                    questions.append(dict(item_id=item["item_id"],
                        question_embeddings=torch.randn(4,dim,generator=generator).bfloat16(),
                        teacher_scores=torch.rand(n,generator=generator),bank_masks=masks,
                        bank_success=torch.tensor([True,True,False]),bank_nll=torch.tensor([.1,.2,3.]),
                        bank_anls=torch.tensor([1.,1.,0.]),bank_names=["a","b","c"]))
                path=run/f"cache/pages/{role}.pt"
                atomic_torch_save(path,dict(page_id=role,parent_id="parent_"+role,role=role,
                    protocol_sha256=digest,features=torch.randn(4,n,dim,generator=generator).bfloat16(),
                    questions=questions))
                index["pages"][role]=dict(role=role,path=str(path),sha256=sha256(path),questions=2)
            atomic_json(run/"cache/index.json",index)
            atomic_json(run/"gates/gpu.json",dict(status="PASS",protocol_sha256=digest,
                                                  scope="synthetic CPU fixture only"))
            for experiment in ("query","sets"):
                train.main(["--run-dir",str(run),"--experiment",experiment,"--device","cpu"])
            summaries={experiment:sha256(run/f"checkpoints/{experiment}_summary.json")
                       for experiment in ("query","sets")}
            # A completed training stage must remain byte-for-byte immutable:
            # otherwise a later interrupted evaluation cannot be resumed.
            for experiment in ("query","sets"):
                train.main(["--run-dir",str(run),"--experiment",experiment,"--device","cpu"])
                self.assertEqual(summaries[experiment],sha256(run/f"checkpoints/{experiment}_summary.json"))
            engine=SimpleNamespace(device="cpu",n=n)

            def prepare(_engine,page,question,max_question_tokens=256):
                self.assertNotIn("answers",page)
                self.assertNotIn("questions",page)
                self.assertIsInstance(question,str)
                visual=torch.arange(_engine.n*dim).reshape(_engine.n,dim).bfloat16()/100
                return dict(V=visual,DS=[visual.clone() for _ in range(3)],
                    question_embeddings=torch.ones(4,dim,dtype=torch.bfloat16),
                    prep={"n_vis":_engine.n},message=question)

            @contextmanager
            def low(_engine,pixels=512):
                old=_engine.n
                _engine.n=k
                try:
                    yield
                finally:
                    _engine.n=old

            def generate(_engine,ctx,keep=None,max_tokens=128,ignore_eos=False):
                count=len(ctx["V"])
                selected=list(range(count)) if keep is None else torch.as_tensor(keep).tolist()
                return dict(text="ok",token_ids=[7]*(max_tokens if ignore_eos else 1),
                    truncated=bool(ignore_eos),actual_visual_count=len(selected),keep_idx=selected,
                    ds_lengths=[len(selected)]*3,positions_sha256="a"*64,
                    audit=dict(native=True,ds_streams=3,positions_preserved=True,
                               physical_visual_count=len(selected)))

            with patch.object(runtime,"load_engine",return_value=engine), \
                 patch.object(runtime,"prepare",side_effect=prepare), \
                 patch.object(runtime,"low_resolution",side_effect=low), \
                 patch.object(runtime,"baseline_keep",side_effect=lambda eng,ctx,name,k:torch.arange(k)), \
                 patch.object(runtime,"generate",side_effect=generate):
                arguments=["--run-dir",str(run),"--timing-blocks","1","--timing-warmup","0"]
                evaluate.main(arguments)
                self.assertEqual(report.main(["--run-dir",str(run),"--bootstrap","32"]),0)
                result=json.loads((run/"evaluation/report.json").read_text())
                self.assertEqual(result["status"],"complete")
                self.assertEqual(result["n_observed_outputs"],2*(5+6))
                # Completed QA/timing restart adds neither duplicates nor new
                # predictions, and validates the same checkpoint freeze.
                evaluate.main(arguments)
                self.assertEqual(report.main(["--run-dir",str(run),"--bootstrap","32"]),0)


if __name__ == "__main__":
    unittest.main()
