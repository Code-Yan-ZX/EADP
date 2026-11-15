"""CPU-only synthetic checks for selection, dtype, data isolation and timing summary."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import torch
import student_eval as e
import student_benchmark as b


class StudentEvalTests(unittest.TestCase):
    def test_stable_ties_and_raster_restoration(self):
        scores = torch.tensor([0., 2., 2., -1., 3.])
        self.assertEqual(e.stable_keep(scores,3).tolist(), [1,2,4])
        with self.assertRaises(ValueError): e.stable_keep(torch.tensor([float('nan')]),1)
        with self.assertRaises(ValueError): e.stable_keep(scores,6)

    def test_random_deterministic_and_valid(self):
        s1,k1=e.random_keep('one');s2,k2=e.random_keep('one');s3,k3=e.random_keep('two')
        self.assertEqual(s1,s2);self.assertTrue(torch.equal(k1,k2))
        self.assertNotEqual(s1,s3);self.assertFalse(torch.equal(k1,k3))
        self.assertEqual(len(k1),256);self.assertEqual(len(set(k1.tolist())),256)
        self.assertEqual(k1.tolist(), sorted(k1.tolist()))

    def test_forward_only_message_does_not_use_labels(self):
        one={'image_path':'x','question':'what value?','answers':['112']}
        two=dict(one,answers=['99999'])
        self.assertEqual(e.message(one['image_path'],one['question']),
                         e.message(two['image_path'],two['question']))
        self.assertNotIn('112',repr(e.message(one['image_path'],one['question'])))
        self.assertNotIn('99999',repr(e.message(two['image_path'],two['question'])))

    def test_bfloat16_stream_order_and_live_scorer(self):
        v=torch.zeros((1024,4096),dtype=torch.bfloat16)
        ds=[v+1,v+2,v+3]; q=torch.ones((1,4096),dtype=torch.float32)
        f,q2=e.input_features(v,ds,q)
        self.assertEqual(f.dtype,torch.bfloat16);self.assertEqual(q2.dtype,torch.bfloat16)
        self.assertEqual(f[:, :, 0, 0].tolist(),[[0.,1.,2.,3.]])
        class Dummy(torch.nn.Module):
            def forward(self,f,q): return f[:,0,:,0].float() + q[:,0,None].float()
        keep,scores=e.select_student(Dummy(),v,ds,q)
        self.assertEqual(keep.tolist(),list(range(256)))
        self.assertEqual(scores.shape,(1024,))
        with self.assertRaises(ValueError): e.input_features(v,ds[:2],q)

    def test_summary_uses_paired_raw_samples(self):
        keys=('ttft_ms','total_16_ms','preprocess_ms','vision_ms','question_and_selector_ms',
              'prefill_and_choice_ms','decode_ms','baseline_allocated_bytes','baseline_reserved_bytes',
              'peak_allocated_bytes','peak_reserved_bytes')
        blocks=[{'arms':{arm:{key:float(i) for key in keys} for arm in b.ARMS}} for i in (2,1,3)]
        s=b.summarize(blocks)
        self.assertEqual(s['b2']['ttft_ms']['median'],2.)
        self.assertEqual(s['b2']['ttft_ms']['samples'],[2.,1.,3.])

    def test_runtime_revision_metadata_and_dtype(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);snap=root/'snapshots'/'abc123';snap.mkdir(parents=True)
            (root/'refs').mkdir();(root/'refs'/'main').write_text('abc123')
            (snap/'config.json').write_text('{"ok":true}')
            model=torch.nn.Linear(1,1).to(dtype=torch.bfloat16).eval()
            model.config=SimpleNamespace(_commit_hash='abc123')
            eng=SimpleNamespace(model=model,vlm=SimpleNamespace(model_path='Qwen/Qwen3-VL-8B-Instruct'),
                text=SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa')))
            train={'model':{'revision':'abc123','snapshot':str(snap),
                    'metadata_sha256':{'config.json':e.sha256(snap/'config.json')},'dtype':'torch.bfloat16'},
                   'environment':{'attention_backend':'sdpa'}}
            self.assertEqual(e.verify_runtime_model(eng,train)['revision'],'abc123')
            model.config._commit_hash='wrong'
            with self.assertRaises(RuntimeError): e.verify_runtime_model(eng,train)
            model.config._commit_hash='abc123';model.float()
            with self.assertRaises(RuntimeError): e.verify_runtime_model(eng,train)
            model.to(dtype=torch.bfloat16);(snap/'config.json').write_text('changed')
            with self.assertRaises(RuntimeError): e.verify_runtime_model(eng,train)

if __name__=='__main__': unittest.main()
