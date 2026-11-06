import unittest
from types import SimpleNamespace
import torch
from selector import locate_user_text, user_text_selector


class Processor:
    def apply_chat_template(self, *_args, **_kwargs): return 'a#hi!b'
    def tokenizer(self, *_args, **_kwargs):
        return {'input_ids':[10,99,11,12,13], 'offset_mapping':[(0,1),(1,2),(2,3),(3,4),(4,6)]}


class Model:
    def _extract_query_text_embeds(self,*_args,**_kwargs): return 'original'


class Tests(unittest.TestCase):
    def inputs(self): return torch.tensor([[10]+[99]*1024+[11,12,13]])
    def test_map_and_full_expansion(self):
        self.assertEqual(locate_user_text(Processor(),[],'hi',self.inputs(),99).tolist(),[1025,1026])
        bad=self.inputs();bad[0,-1]=4
        with self.assertRaisesRegex(ValueError,'IDs differ'):locate_user_text(Processor(),[],'hi',bad,99)
    def test_reject_boundary_and_ambiguity(self):
        class Bad(Processor):
            def tokenizer(self,*a,**kw):
                x=super().tokenizer(*a,**kw);x['offset_mapping'][2]=(1,3);return x
        with self.assertRaisesRegex(ValueError,'boundaries'):locate_user_text(Bad(),[],'hi',self.inputs(),99)
        with self.assertRaisesRegex(ValueError,'ambiguous'):locate_user_text(Processor(),[],'',self.inputs(),99)
    def test_selector_and_restoration(self):
        ids=self.inputs();positions=torch.tensor([1025,1026]);model=Model();embeds=torch.arange(ids.shape[1]).reshape(1,-1,1)
        with user_text_selector(model,ids,positions):
            self.assertEqual(model._extract_query_text_embeds(embeds,ids).flatten().tolist(),[1025,1026])
            bad=ids.clone();bad[0,0]=7
            with self.assertRaisesRegex(ValueError,'changed'):model._extract_query_text_embeds(embeds,bad)
        self.assertEqual(model._extract_query_text_embeds(),'original')
        self.assertNotIn('_extract_query_text_embeds',model.__dict__)
    def test_restores_instance_method_on_exception(self):
        ids=self.inputs();model=Model();model._extract_query_text_embeds=lambda *_:'custom';original=model._extract_query_text_embeds
        with self.assertRaisesRegex(RuntimeError,'test'):
            with user_text_selector(model,ids,torch.tensor([1025])): raise RuntimeError('test')
        self.assertIs(model._extract_query_text_embeds,original)
    def test_reject_duplicate_and_bad_indices(self):
        with self.assertRaisesRegex(ValueError,'unique'):
            with user_text_selector(Model(),self.inputs(),torch.tensor([1025,1025])): pass
        with self.assertRaisesRegex(ValueError,'range'):
            with user_text_selector(Model(),self.inputs(),torch.tensor([9999])): pass


if __name__=='__main__':unittest.main(verbosity=2)
