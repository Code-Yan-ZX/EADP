"""Synthetic CPU test of E4 hook wiring and natural-output record checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn

from test_skip_deepstack import FakeVision
from off_adapter import E4OffAdapter
from identity_records import make_record,compare_pair
import identity_worker


def original_greedy(*_args,**_kwargs):
    return torch.arange(255,-1,-1).reshape(1,256),torch.tensor([256])


class FakePruner(nn.Module):
    def __init__(self,namespace):super().__init__();self.namespace=namespace
    def forward(self,visual):
        selected,_=self.namespace._greed_select_impl()
        return visual.index_select(0,selected[0].sort().values),[256]


class FakeLanguage(nn.Module):
    def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None):
        if visual_pos_masks is not None or deepstack_visual_embeds is not None:raise ValueError('not DS-off')
        if position_ids is None:raise ValueError('missing position IDs')
        return inputs_embeds.mean(dim=1)


class FakeCore:
    def __init__(self):self.language_model=FakeLanguage();self.rope_deltas=None
    def get_rope_index(self,input_ids,_grid,_video,attention_mask):
        length=input_ids.shape[1]
        return torch.arange(length).view(1,1,length).expand(3,1,length),torch.tensor([[0]])


class FakeModel:
    def __init__(self):
        self.visual=FakeVision();self.model=FakeCore();self.config=types.SimpleNamespace(image_token_id=999)
        self.embed=nn.Embedding(1000,4)
    def get_input_embeddings(self):return self.embed


class FakeWrapper:
    def __init__(self):
        self.namespace=types.SimpleNamespace(_greed_select_impl=original_greedy)
        self.model=FakeModel();self.pruner=FakePruner(self.namespace)
    def _processor_inputs(self,_message):
        ids=torch.tensor([[1,2]+[999]*1024+[3,4,5]])
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
                'pixel_values':torch.ones(1024,4),'image_grid_thw':torch.tensor([[1,64,64]])}


def fake_natural(w,mode,corrupt=False):
    sample={'task':'DocVQA_VAL','qid':'1','group_id':'g','image_sha256':'image','message_sha256':'prompt'}
    with E4OffAdapter(torch,w,w.namespace,mode) as adapter:
        inputs=w._processor_inputs([])
        main,_=w.model.visual(inputs['pixel_values'])
        pruned,_=w.pruner(main)
        embeddings=w.model.get_input_embeddings()(inputs['input_ids'])
        selected=pruned.clone()
        if corrupt:selected[0,0]+=1
        embeds=torch.cat((embeddings[:,:2],selected.unsqueeze(0),embeddings[:,1026:]),dim=1)
        mask=torch.ones((1,embeds.shape[1]),dtype=torch.long)
        logits=w.model.model.language_model(inputs_embeds=embeds,attention_mask=mask)
        token=int(logits[0].argmax())
        return make_record(adapter,sample,[token],'synthetic output',logits[0],[99])


class OffAdapterTests(unittest.TestCase):
    def test_pair_records_and_full_restoration(self):
        torch.manual_seed(3);w=FakeWrapper()
        drop=fake_natural(w,'C-DROP');skip=fake_natural(w,'C-SKIP')
        self.assertEqual(compare_pair(drop,skip)['status'],'identity_passed_bitwise')
        self.assertEqual(drop['main_visual'],skip['main_visual'])
        self.assertEqual(drop['output_ids'],skip['output_ids'])
        self.assertEqual(drop['ds_results'],[(0,False),(1,False),(2,False)])
        self.assertEqual(skip['ds_results'],[(0,True),(1,True),(2,True)])
        self.assertNotIn('_processor_inputs',w.__dict__)
        self.assertIs(w.namespace._greed_select_impl,original_greedy)
        self.assertTrue(all('forward' not in m.__dict__ for m in w.model.visual.deepstack_merger_list))
        self.assertFalse(w.model.visual._forward_hooks)
        self.assertFalse(w.model.model.language_model._forward_pre_hooks)

    def test_pair_rejects_real_output_or_primary_change(self):
        torch.manual_seed(3);w=FakeWrapper();drop=fake_natural(w,'C-DROP');skip=fake_natural(w,'C-SKIP')
        bad=copy.deepcopy(skip);bad['output_ids'][0]+=1
        with self.assertRaisesRegex(ValueError,'output_ids'):compare_pair(drop,bad)
        bad=copy.deepcopy(skip);bad['main_visual']['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'main_visual'):compare_pair(drop,bad)

    def test_prefill_corruption_fails_and_restores_everything(self):
        w=FakeWrapper()
        with self.assertRaisesRegex(ValueError,'prefill features'):fake_natural(w,'C-SKIP',corrupt=True)
        self.assertNotIn('_processor_inputs',w.__dict__)
        self.assertIs(w.namespace._greed_select_impl,original_greedy)
        self.assertTrue(all('forward' not in m.__dict__ for m in w.model.visual.deepstack_merger_list))
        self.assertFalse(w.model.visual._forward_hooks)

    def test_requires_completed_e1_before_any_device_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof=Path(tmp)/'incomplete.json';proof.write_text('{"status":"passed","stage":"full"}')
            with self.assertRaisesRegex(ValueError,'completed, independently accepted E1'):
                identity_worker.check_e1(proof)

    def test_fixed_dev_selection_excludes_all_old_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp);ex=data/'anchor_exclusion';ex.mkdir()
            image=data/'image';image.write_bytes(b'synthetic image');image_sha=identity_worker.sha(image)
            panel=[];inputs=[]
            for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
                for i in range(128):
                    qid=str(i);gid=task+qid
                    panel.append({'task':task,'qid':qid,'group_id':gid,'split':'DEV'})
                    message=[{'type':'image','value':str(image)},{'type':'text','value':'synthetic'}]
                    inputs.append({'task':task,'qid':qid,'group_id':gid,'image_path':str(image),'image_sha256':image_sha,
                                   'message':message,'message_sha256':hashlib.sha256(json.dumps(message,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
            def write(path,records):path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            panel_path=data/'panel_manifest.jsonl';input_path=data/'inputs.jsonl'
            write(panel_path,panel);write(input_path,inputs)
            full=data/'full_manifest.jsonl';full.write_text('synthetic\n')
            anchor=ex/'anchor_groups.jsonl';write(anchor,[{'group_id':'DocVQA_VAL'+str(i)} for i in range(16)])
            (ex/'build.py').write_text('pass\n');(ex/'verify.py').write_text('pass\n')
            summary={'builder_sha256':identity_worker.sha(ex/'build.py'),
                     'panel_manifest_sha256':identity_worker.sha(panel_path),'full_manifest_sha256':identity_worker.sha(full),
                     'outputs_sha256':{'anchor_groups.jsonl':identity_worker.sha(anchor)}}
            (ex/'summary.json').write_text(json.dumps(summary))
            (ex/'verification.json').write_text(json.dumps({'status':'pass','summary_sha256':identity_worker.sha(ex/'summary.json'),
                                                           'verifier_sha256':identity_worker.sha(ex/'verify.py')}))
            proof={'files':{'panel':{'path':str(panel_path)},'inputs':{'path':str(input_path)}}}
            with patch.object(identity_worker,'DATA',data):selected=identity_worker.select_panel(proof)
            self.assertEqual(len(selected),16)
            self.assertEqual(len({r['group_id'] for r in selected}),16)
            self.assertTrue(all(r['task']=='DocVQA_VAL' and r['group_id'] not in {'DocVQA_VAL'+str(i) for i in range(16)} for r in selected))

if __name__=='__main__':unittest.main()
