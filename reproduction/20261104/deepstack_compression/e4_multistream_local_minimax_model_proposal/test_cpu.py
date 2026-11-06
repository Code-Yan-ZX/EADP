import copy
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE / 'e4'))
import cohort


def dataset():
    identities, archives, rows, hashes = [], [], [], {}
    top = [32*r+c for r in range(0,32,2) for c in range(0,32,2)]
    bottom = sorted(x+32 for x in top)
    for n in range(32):
        task = 'DocVQA_VAL' if n < 11 else 'TextVQA_VAL' if n < 22 else 'AI2D_TEST'
        identity = {'task':task,'qid':str(n),'group_id':'g'+str(n), 'image_sha256':'i'+str(n),
                    'message_sha256':'m'+str(n),'processor':{'input_ids':{'sha256':'s'}}}
        path = '/test/' + str(n); digest = 'h'+str(n)
        archive = dict(identity,index=n,path=path,sha256=digest)
        row = {'index':n,'task':task,'qid':str(n),'group_id':identity['group_id'],
               'archive_sha256':digest,'arms':{'minimax':{'ids':top},'main_minimax':{'ids':bottom}}}
        identities.append(identity); archives.append(archive); rows.append(row); hashes[path]=digest
    return ({'input_identities':identities},{'archives':archives},rows,
            {'status':'passed','images':32,'array_sha256':hashes},
            {'status':'passed','images':32,'independent_review':True,'array_sha256':hashes})


class CohortTests(unittest.TestCase):
    def test_valid_and_full_denominator(self):
        self.assertEqual(len(cohort.construct(*dataset())),32)
    def test_wrong_image_or_archive(self):
        for field in ('image_sha256','message_sha256','qid','group_id'):
            d=dataset();d[1]['archives'][0][field]='wrong'
            with self.subTest(field=field),self.assertRaises(ValueError):cohort.construct(*d)
        d=dataset();d[2][0]['archive_sha256']='wrong'
        with self.assertRaises(ValueError):cohort.construct(*d)
    def test_duplicate_or_reordered(self):
        d=dataset();d[2][0],d[2][1]=d[2][1],d[2][0]
        with self.assertRaises(ValueError):cohort.construct(*d)
        d=dataset();d[0]['input_identities'].pop()
        with self.assertRaises(ValueError):cohort.construct(*d)
    def test_missing_block_duplicate_or_out_of_range(self):
        for change in ('duplicate','block','range','bool','order'):
            ids=copy.copy(dataset()[2][0]['arms']['minimax']['ids'])
            if change=='duplicate':ids[1]=ids[0]
            if change=='block':ids[1]=ids[0]+1
            if change=='range':ids[-1]=1024
            if change=='bool':ids[0]=False
            if change=='order':ids[0],ids[1]=ids[1],ids[0]
            with self.subTest(change=change),self.assertRaises(ValueError):cohort.validate_ids(ids)
    def test_failed_formal_or_independent(self):
        for pos in (3,4):
            d=dataset();d[pos]['status']='failed'
            with self.assertRaises(ValueError):cohort.construct(*d)


class RealAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        from test_off_adapter import FakeWrapper
        cls.wrapper=FakeWrapper
        spec=importlib.util.spec_from_file_location('selected_adapter_cpu', BASE/'m3_oracle_exec/selection_adapter.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.Adapter=module.SelectedAdapter

    def setup_run(self, wrong=False):
        torch=self.torch;w=self.wrapper()
        class Language(torch.nn.Module):
            def forward(self, inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None):
                self.ds=deepstack_visual_embeds;self.positions=position_ids;return inputs_embeds.mean(dim=1)
        w.model.model.language_model=Language()
        ids=dataset()[2][0]['arms']['minimax']['ids']; baseline=list(range(256)) if wrong else list(range(255,-1,-1))
        return w,ids,baseline

    def test_actual_adapter_main_p3_and_three_ds_hooks(self):
        torch=self.torch;w,ids,baseline=self.setup_run()
        with self.Adapter(torch,w,w.namespace,ids,baseline) as a:
            inp=w._processor_inputs([]);main,_=w.model.visual(inp['pixel_values']);pruned,_=w.pruner(main)
            self.assertTrue(torch.equal(pruned,main[ids]))
            embeds=w.model.get_input_embeddings()(inp['input_ids']);embeds=torch.cat((embeds[:,:2],pruned[None],embeds[:,1026:]),dim=1)
            w.model.model.language_model(inputs_embeds=embeds,attention_mask=torch.ones((1,261),dtype=torch.long))
            self.assertTrue(a.verify_calls());self.assertEqual(a.state['spatial'].tolist(),ids)
            for actual,full in zip(a.state['actual_deepstack'],a.state['deepstack']):
                self.assertTrue(torch.equal(actual,full[ids]))
            self.assertEqual(a.state['retained'].tolist(),[0,1]+[2+i for i in ids]+list(range(1026,inp['input_ids'].shape[1])))
        self.assertFalse(w.model.model.language_model._forward_pre_hooks)
        self.assertFalse(w.pruner._forward_hooks)

    def test_actual_adapter_rejects_baseline_change_and_cleans_hooks(self):
        w,ids,baseline=self.setup_run(wrong=True)
        with self.assertRaisesRegex(ValueError,'baseline'):
            with self.Adapter(self.torch,w,w.namespace,ids,baseline):
                x=w._processor_inputs([]);main,_=w.model.visual(x['pixel_values']);w.pruner(main)
        self.assertFalse(w.model.model.language_model._forward_pre_hooks)
        self.assertFalse(w.pruner._forward_hooks)


class ArmOrchestrationTests(unittest.TestCase):
    def prepare(self):
        from types import SimpleNamespace
        import arm
        ids=dataset()[2][0]['arms']['minimax']['ids']
        controls={'do_sample':False,'max_new_tokens':2048,'top_p':1.0,'top_k':1,'temperature':1.0,'repetition_penalty':1.0}
        sample={'split':'DEV','task':'DocVQA_VAL','qid':'0','group_id':'g','message_sha256':'m'}
        ref={**sample,'arm':'P3-D1','forced':False,'generation_scalars':controls,'eos_ids':[2],
             'ordered_ids':list(range(256)), 'processor':{'input_ids':{'shape':[1,1029]}},
             'visual_sha256':{'sha256':'visual'}, 'deepstack_full_sha256':[{'sha256':str(i)} for i in range(3)]}
        class Raw:
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return [[3,2]]
        wrapper=SimpleNamespace(model=SimpleNamespace(generate=lambda **kw:Raw(),generation_config=SimpleNamespace(eos_token_id=2)))
        result={'selection_mode':'oracle_fixed_indices','ordered_ids':ids,'spatial_ids':ids,
                'actual_injection_verified':True,'processor':ref['processor'],'main_visual':ref['visual_sha256'],
                'full_deepstack':ref['deepstack_full_sha256'],'output_ids':[3,2]}
        original_adapter=object()
        class Runner:
            BudgetAdapter=original_adapter
            def run_arm(self,t,w,n,s,k,d,g,on,archive,**kw):
                if self.BudgetAdapter is original_adapter:raise AssertionError('factory missing')
                value=g(w,s);on({'output_ids':value});return copy.deepcopy(result)
        runner=Runner()
        def generate(w,row):return w.model.generate(**controls)
        return arm,ids,sample,ref,wrapper,runner,original_adapter,generate,controls

    def test_64_generation_guard_and_restore(self):
        arm,ids,sample,ref,w,runner,original,generate,controls=self.prepare()
        records=[]
        for i in range(64):
            r=arm.run_arm(None,w,None,sample,'minimax' if i%2==0 else 'main_minimax',ids,ref,runner,object,generate,records.append,None,None,lambda x:None)
            self.assertFalse(r['timing_eligible']);self.assertTrue(r['extra_author_greedy'])
        self.assertEqual(len(records),64);self.assertIs(runner.BudgetAdapter,original)

    def test_bad_kwargs_or_eos_reject_and_restore(self):
        arm,ids,sample,ref,w,runner,original,generate,controls=self.prepare()
        for bad in ('temperature','eos_token_id','stopping_criteria'):
            wrong=dict(controls);wrong[bad]={'temperature':2.0,'eos_token_id':[8],'stopping_criteria':object()}[bad]
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'controls'):
                arm.run_arm(None,w,None,sample,'minimax',ids,ref,runner,object,lambda w,r:w.model.generate(**wrong),lambda x:None,None,None,lambda x:None)
            self.assertIs(runner.BudgetAdapter,original)
            self.assertEqual(w.model.generate().tolist(),[[3,2]])

    def test_reference_and_selection_rejected_before_generate(self):
        arm,ids,sample,ref,w,runner,original,generate,controls=self.prepare()
        ref['arm']='P3-D0'
        with self.assertRaisesRegex(ValueError,'P3'):arm.run_arm(None,w,None,sample,'minimax',ids,ref,runner,object,generate,lambda x:None,None,None,lambda x:None)
        self.assertIs(runner.BudgetAdapter,original)

class RealCandidateEntryTests(unittest.TestCase):
    def invoke(self, corrupt=None):
        import torch,types,arm
        from test_off_adapter import FakeWrapper
        from identity_records import tensor_meta
        spec=importlib.util.spec_from_file_location('selected_adapter_candidate_cpu', BASE/'m3_oracle_exec/selection_adapter.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        w=FakeWrapper()
        class Language(torch.nn.Module):
            def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None):
                return inputs_embeds.mean(dim=1)
        w.model.model.language_model=Language()
        w.model.generation_config=types.SimpleNamespace(eos_token_id=2)
        w.model.generate=lambda **kw:torch.tensor([[3,2]])
        inp=w._processor_inputs([]);visual,streams=w.model.visual(inp['pixel_values'])
        controls={'do_sample':False,'max_new_tokens':2048,'top_p':1.0,'top_k':1,'temperature':1.0,'repetition_penalty':1.0}
        sample={'split':'DEV','task':'DocVQA_VAL','qid':'1','group_id':'g','message_sha256':'m'}
        ref={**sample,'arm':'P3-D1','forced':False,'generation_scalars':controls,'eos_ids':[2],
             'ordered_ids':list(range(255,-1,-1)),
             'processor':{k:tensor_meta(inp[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
             'visual_sha256':tensor_meta(visual), 'deepstack_full_sha256':[tensor_meta(z) for z in streams]}
        if corrupt=='processor':ref['processor']['input_ids']['sha256']='x'
        if corrupt=='visual':ref['visual_sha256']['sha256']='x'
        if corrupt in ('ds0','ds1','ds2'):ref['deepstack_full_sha256'][int(corrupt[-1])]['sha256']='x'
        ids=dataset()[2][0]['arms']['minimax']['ids']
        original=object()
        class Runner:
            BudgetAdapter=original
            def run_arm(self,t,w,n,s,k,d,g,on,archive,**kw):
                with self.BudgetAdapter(t,w,n,k,d) as a:
                    input_=w._processor_inputs([]);main,_=w.model.visual(input_['pixel_values']);pruned,_=w.pruner(main)
                    embeds=w.model.get_input_embeddings()(input_['input_ids']);embeds=torch.cat((embeds[:,:2],pruned[None],embeds[:,1026:]),dim=1)
                    w.model.model.language_model(inputs_embeds=embeds,attention_mask=torch.ones((1,261),dtype=torch.long))
                    a.verify_calls();tokens=g(w,s).tolist()[0];on({'output_ids':tokens})
                    state=a.state
                    return {'selection_mode':state['selection_mode'],'ordered_ids':state['ordered'].tolist(),
                            'spatial_ids':state['spatial'].tolist(),'actual_injection_verified':state['actual_injection_verified'],
                            'processor':{k:tensor_meta(state['inputs'][k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
                            'main_visual':tensor_meta(state['visual']),'full_deepstack':[tensor_meta(z) for z in state['deepstack']],
                            'output_ids':tokens}
        runner=Runner();exposures=[];raws=[]
        try:
            result=arm.run_arm(torch,w,w.namespace,sample,'minimax',ids,ref,runner,module.SelectedAdapter,
                               lambda w,s:w.model.generate(**controls),exposures.append,None,tensor_meta,raws.append)
            self.assertEqual(raws[0]['returned_batch'],[[3,2]])
            self.assertEqual(raws[0]['processor_input_length'],1029)
            return result,exposures
        finally:
            self.assertIs(runner.BudgetAdapter,original)
            self.assertFalse(w.model.model.language_model._forward_pre_hooks)
            self.assertFalse(w.pruner._forward_hooks)
    def test_candidate_real_selected_adapter_and_hooks(self):
        result,exposures=self.invoke();self.assertEqual(result['output_ids'],[3,2]);self.assertEqual(len(exposures),1)
    def test_candidate_rejects_actual_fullstream_identity_before_generate(self):
        for corrupt in ('processor','visual','ds0','ds1','ds2'):
            with self.subTest(field=corrupt),self.assertRaisesRegex(ValueError,'processor or full visual streams'):
                self.invoke(corrupt)
