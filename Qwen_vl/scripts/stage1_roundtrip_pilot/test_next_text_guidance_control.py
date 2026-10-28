"""CPU semantic gates for opt-in text guidance repairs; no model loading."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'LLaVA'))
from llava.model.llava_arch import LlavaMetaForCausalLM as Arch
from next_text_guidance_control import bounded_clip_chunks, valid_entropy_filter


class TextGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_boundaries_preserve_all_content_and_pool_at_eos(self):
        for n in [0,1,74,75,76,77,150,151,500]:
            with self.subTest(content_length=n):
                content = torch.arange(1,n+1)
                ids = torch.cat([torch.tensor([49406]),content,torch.tensor([49407])])[None,:]
                out = bounded_clip_chunks(ids,torch.ones_like(ids),49406,49407)
                chunks = out['input_ids'].view(-1,77)
                masks = out['attention_mask'].view(-1,77).bool()
                self.assertTrue(torch.equal(chunks.argmax(-1),masks.sum(-1)-1))
                self.assertTrue(bool((chunks[:,0]==49406).all()))
                kept = chunks[masks & (chunks!=49406) & (chunks!=49407)]
                self.assertTrue(torch.equal(kept,content))

    def test_all_valid_is_bit_exact_released_filter(self):
        local = torch.randn(5,20,2,77)*.01
        expected = Arch.entrpy_filter(None,local,T=100.,entropy_keep_ratio=.2)
        actual = valid_entropy_filter(local,torch.ones(2,77,dtype=torch.bool))
        for a,b in zip(expected,actual):
            self.assertTrue(torch.equal(a,b))

    def test_padding_cannot_change_selection_or_aggregation(self):
        local = torch.randn(5,20,2,77)*.01
        valid = torch.zeros(2,77,dtype=torch.bool)
        valid[0,1:8] = True
        valid[1,1:37] = True
        # Make invalid tokens maximally attractive to the released filter.
        bad = local.clone()
        bad.masked_fill_(~valid[None,None,:,:],0.)
        bad[:,0,:,:] = torch.where(~valid[None,:,:],torch.ones_like(bad[:,0,:,:]),bad[:,0,:,:])
        first = valid_entropy_filter(local,valid)
        second = valid_entropy_filter(bad,valid)
        self.assertTrue(torch.equal(first[0],second[0]))
        self.assertTrue(torch.equal(first[1],second[1]))
        out = Arch.local_aggregation(None,*second,M_temp=.01,strategy='negative_entropy')
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertEqual(torch.isfinite(second[1]).sum(-1).tolist(),[[1,7]]*5)

    def test_empty_segment_has_finite_neutral_local(self):
        local = torch.randn(1,10,1,77)
        filtered, entropy = valid_entropy_filter(local,torch.zeros(1,77,dtype=torch.bool))
        out = Arch.local_aggregation(None,filtered,entropy)
        self.assertTrue(torch.equal(out,torch.zeros_like(out)))


if __name__ == '__main__':
    unittest.main()
