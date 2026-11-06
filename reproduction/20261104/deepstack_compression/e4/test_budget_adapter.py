"""CPU-only tests: budget-specific greedy, keep-all, DS alignment and failures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch
import torch
import numpy as np
from torch import nn
from test_off_adapter import FakeWrapper
from budget_adapter import BudgetAdapter
from budget_reuse import check_reuse
from budget_runner import run_arm
from verify_k256 import compare_record, FIELDS
import verify_k256, smoke_launch
import smoke_contract


class Language(nn.Module):
    def forward(self, inputs_embeds, attention_mask, position_ids=None, visual_pos_masks=None, deepstack_visual_embeds=None):
        return inputs_embeds.mean(dim=1)


def run(k, on, corrupt=None):
    w = FakeWrapper(); w.model.model.language_model = Language()
    if corrupt == 'mutate_forward':
        class MutatingLanguage(nn.Module):
            def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None):
                inputs_embeds.add_(17);attention_mask.add_(3);position_ids.add_(5)
                w.model.model.rope_deltas.add_(11)
                if visual_pos_masks is not None:visual_pos_masks.logical_not_()
                if deepstack_visual_embeds is not None:
                    for x in deepstack_visual_embeds:x.add_(23)
                return inputs_embeds.mean(dim=1)
        w.model.model.language_model=MutatingLanguage()
    def greedy():
        return torch.arange(k - 1, -1, -1).reshape(1, k), torch.tensor([k])
    w.namespace._greed_select_impl = greedy
    class Pruner(nn.Module):
        def forward(self, x):
            if k == 1024: return x, [1024]
            order, _ = w.namespace._greed_select_impl()
            return x.index_select(0, order[0].sort().values), [k]
    w.pruner = Pruner()
    with BudgetAdapter(torch, w, w.namespace, k, on) as a:
        inp = w._processor_inputs([]); visual, _ = w.model.visual(inp['pixel_values'])
        pruned, _ = w.pruner(visual)
        embed = w.model.get_input_embeddings()(inp['input_ids'])
        if corrupt == 'embed': pruned = pruned.clone(); pruned[0, 0] += 1
        embed = torch.cat((embed[:, :2], pruned.unsqueeze(0), embed[:, 1026:]), dim=1)
        w.model.model.language_model(inputs_embeds=embed, attention_mask=torch.ones((1, k+5), dtype=torch.long))
        a.verify_calls()
        return a.state


class BudgetTests(unittest.TestCase):
    def test_all_budgets_and_ds_modes(self):
        for k in (128, 256, 512, 1024):
            for on in (False, True):
                with self.subTest(k=k, on=on):
                    s = run(k, on)
                    self.assertEqual(s['greedy_calls'], 0 if k == 1024 else 1)
                    self.assertEqual(s['selection_mode'], 'keep_all' if k == 1024 else 'author_greedy')
                    self.assertEqual(len(s['retained']), k+5)
                    self.assertEqual(s['adjusted_delta'].item(), 1024-k)
                    self.assertEqual(s['intended_ds_mask'] is not None, on)
                    for original, selected in zip(s['deepstack'], s['selected_deepstack']):
                        self.assertTrue(torch.equal(selected, original.index_select(0, s['spatial'])))

    def test_corrupt_prefill_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prefill features'): run(128, True, 'embed')

    def test_prefill_snapshot_survives_forward_in_place_mutation(self):
        s=run(128,True,'mutate_forward')
        self.assertTrue(torch.equal(s['prefill_embeds'][0,2:130],s['visual'].index_select(0,s['spatial'])))
        self.assertTrue(torch.equal(s['prefill_mask'],torch.ones_like(s['prefill_mask'])))
        self.assertTrue(torch.equal(s['prefill_position_ids'],s['full_pos'].index_select(-1,s['retained'])))
        self.assertEqual(int(s['actual_ds_mask'].sum()),128)
        self.assertEqual(s['adjusted_delta'].item(),896)
        for original,selected,actual in zip(s['deepstack'],s['selected_deepstack'],s['actual_deepstack']):
            self.assertTrue(torch.equal(selected,original.index_select(0,s['spatial'])))
            self.assertTrue(torch.equal(actual,selected))

    def test_runner_records_and_enforces_dev(self):
        w = FakeWrapper();w.visual_token_num = 128
        class Cache:
            def __init__(self,n):self.n=n
            def get_seq_length(self,_index):return self.n
        class TraceLanguage(nn.Module):
            def __init__(self):super().__init__();self.layers=[None]*36
            def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None,
                        cache_position=None,past_key_values=None):
                return types.SimpleNamespace(past_key_values=Cache(inputs_embeds.shape[1]))
        w.model.model.language_model=TraceLanguage()
        w.model.generation_config=types.SimpleNamespace(eos_token_id=99)
        w.model.generate=lambda **kw:torch.tensor([[99]])
        def greedy(): return torch.arange(127, -1, -1).reshape(1, 128), torch.tensor([128])
        w.namespace._greed_select_impl = greedy
        class Pruner(nn.Module):
            visual_token_num = 128
            def forward(self, x):
                order, _ = w.namespace._greed_select_impl()
                return x.index_select(0, order[0].sort().values), [128]
        w.pruner = Pruner()
        def generate(wrapper, _sample):
            inp = wrapper._processor_inputs([]); visual, _ = wrapper.model.visual(inp['pixel_values'])
            selected, _ = wrapper.pruner(visual); embed = wrapper.model.get_input_embeddings()(inp['input_ids'])
            embed = torch.cat((embed[:, :2], selected.unsqueeze(0), embed[:, 1026:]), dim=1)
            wrapper.model.model.language_model(inputs_embeds=embed, attention_mask=torch.ones((1,133),dtype=torch.long),
                                               cache_position=torch.arange(133),past_key_values=None)
            wrapper.model.generate(do_sample=False,max_new_tokens=2048)
            return ''
        sample = dict(split='DEV', task='AI2D_TEST', qid='1', group_id='g', image_sha256='x', message_sha256='y')
        emitted=[]
        archive=lambda tensors:{'kind':'full_arrays','path':'synthetic','sha256':'x'}
        rec = run_arm(torch, w, w.namespace, sample, 128, True, generate, emitted.append, archive)
        self.assertEqual(rec['answer'], '')
        self.assertTrue(rec['eos']); self.assertEqual(len(rec['selected_deepstack']), 3)
        self.assertEqual(emitted[0]['output_ids'],[99])
        def fail_archive(_):raise RuntimeError('post-generation failure')
        with self.assertRaisesRegex(RuntimeError,'post-generation'):run_arm(torch,w,w.namespace,sample,128,True,generate,emitted.append,fail_archive)
        self.assertEqual(len(emitted),2)
        with self.assertRaisesRegex(ValueError, 'DEV'): run_arm(torch, w, w.namespace, dict(sample, split='FINAL'), 128, True, generate, emitted.append, archive)
        with self.assertRaisesRegex(ValueError, 'budget'): run_arm(torch, w, w.namespace, sample, 256, True, generate, emitted.append, archive)

    def test_forced_33_outputs_capture_real_32_step_trace_and_anchor_arrays(self):
        w=FakeWrapper();w.visual_token_num=128;w.model.generation_config=types.SimpleNamespace(eos_token_id=99)
        w.model.lm_head=nn.Linear(4,100,bias=False)
        class Cache:
            def __init__(self,n):self.n=n
            def get_seq_length(self,_index):return self.n
        class TraceLanguage(nn.Module):
            def __init__(self):super().__init__();self.layers=[None]*36
            def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None,
                        cache_position=None,past_key_values=None):
                n=inputs_embeds.shape[1] if past_key_values is None else past_key_values.n+1
                return types.SimpleNamespace(past_key_values=Cache(n))
        w.model.model.language_model=TraceLanguage()
        def greedy():return torch.arange(127,-1,-1).reshape(1,128),torch.tensor([128])
        w.namespace._greed_select_impl=greedy
        class Pruner(nn.Module):
            visual_token_num=128
            def forward(self,x):
                order,_=w.namespace._greed_select_impl()
                return x.index_select(0,order[0].sort().values),[128]
        w.pruner=Pruner()
        def actual_generate(**kw):
            self.assertEqual(kw['max_new_tokens'],33);self.assertEqual(kw['min_new_tokens'],33);self.assertEqual(kw['eos_token_id'],[])
            scores=kw['logits_processor'](torch.tensor([[0]]),torch.zeros((1,100)))
            self.assertEqual(int(scores.argmax()),17)
            embed=w.pending_embed;lm=w.model.model.language_model
            out=lm(inputs_embeds=embed,attention_mask=torch.ones((1,133),dtype=torch.long),
                   cache_position=torch.arange(133),past_key_values=None)
            w.model.lm_head(torch.ones((1,1,4)))
            for step in range(32):
                pos=133+step
                out=lm(inputs_embeds=torch.ones((1,1,4)),attention_mask=torch.ones((1,pos+1),dtype=torch.long),
                       position_ids=torch.tensor([[[pos+896]]]*3),cache_position=torch.tensor([pos]),past_key_values=out.past_key_values)
            return torch.tensor([[17]*33])
        w.model.generate=actual_generate
        def generate(wrapper,_sample):
            inp=wrapper._processor_inputs([]);visual,_=wrapper.model.visual(inp['pixel_values']);selected,_=wrapper.pruner(visual)
            embed=wrapper.model.get_input_embeddings()(inp['input_ids'])
            wrapper.pending_embed=torch.cat((embed[:,:2],selected.unsqueeze(0),embed[:,1026:]),dim=1)
            wrapper.model.generate(do_sample=False,max_new_tokens=2048)
            return 'forced answer'
        sample=dict(split='DEV',task='AI2D_TEST',qid='1',group_id='g',image_sha256='x',message_sha256='y')
        journal=[];archives=[]
        def archive(tensors):archives.append(set(tensors));return {'kind':'full_arrays','path':'synthetic','sha256':'x'}
        rec=run_arm(torch,w,w.namespace,sample,128,True,generate,journal.append,archive,forced_token=17)
        self.assertEqual(len(rec['decode_trace']),33);self.assertEqual(rec['output_ids'],[17]*33)
        self.assertTrue(journal[0]['forced'])
        self.assertTrue({'processor_input_ids','processor_pixel_values','prefill_logits'}.issubset(archives[0]))

    def test_reuse_missing_or_changed_evidence_rejected(self):
        evidence = {'accepted_e1': True, 'same_protocol': True, 'same_question': True,
                    'same_processor': True, 'same_selection': True, 'same_p3': True,
                    'same_visual_and_ds': True, 'same_generation': True,
                    'same_natural_output': True, 'independent_verifier': True}
        self.assertTrue(check_reuse(evidence))
        for key in evidence:
            bad = copy.deepcopy(evidence); bad.pop(key)
            with self.assertRaises(ValueError): check_reuse(bad)
            bad = copy.deepcopy(evidence); bad[key] = False
            with self.assertRaises(ValueError): check_reuse(bad)
        with self.assertRaises(ValueError): check_reuse(dict(evidence, extra=True))

    def test_k256_independent_record_comparison_rejects_mismatches(self):
        e4={k: 'same' for k in FIELDS}; e1={v: 'same' for v in FIELDS.values()}
        e4.update(budget=256, selection_mode='author_greedy', greedy_calls=1, actual_injection_verified=True,
                  ds_on=True, image_sha256='image', max_new_tokens=2048,
                  generation_scalars={'do_sample':False,'max_new_tokens':2048})
        e1.update(arm='P3-D1', forced=False, generation_scalars=e4['generation_scalars'])
        row={k:e4[k] for k in ('task','qid','group_id','message_sha256')}; row['image_sha256']='image'
        self.assertTrue(compare_record(e4,e1,row))
        for key in FIELDS:
            bad=copy.deepcopy(e4);bad[key]=None
            with self.assertRaises(ValueError, msg=key):compare_record(bad,e1,row)
        for name in ('image_sha256','group_id'):
            bad=copy.deepcopy(row);bad[name]='changed'
            with self.assertRaises(ValueError):compare_record(e4,e1,bad)

    def test_smoke_selection_distinct_and_excludes_old_anchor(self):
        rows=[{'task':task,'qid':str(i),'group_id':task+str(i)} for task in smoke_contract.COUNTS for i in range(128)]
        old=[{'group_id':'DocVQA_VAL'+str(i)} for i in range(16)]
        def read(path):return rows if str(path)=='inputs' else old
        with patch.object(smoke_contract,'select_panel'),patch.object(smoke_contract,'read_jsonl',side_effect=read):
            sample=smoke_contract.select_smoke({'files':{'inputs':{'path':'inputs'}}})
        self.assertEqual(len(sample),32);self.assertEqual(len({r['group_id'] for r in sample}),32)
        self.assertTrue(set(r['group_id'] for r in sample).isdisjoint({r['group_id'] for r in old}))

    def test_smoke_terminal_requires_clean_and_real_container(self):
        here=verify_k256.HERE;run=here/'smoke_abc'
        ident={'pid':123,'start_ticks':4,'boot_id':'old'};host={'pid':124,'start_ticks':5,'boot_id':'old'}
        started={'worker_pid':123,'cwd':str(here),'command':['python',str(here/'smoke_launch.py'),'--run-id','abc']}
        terminal={'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
        launch={'controller':ident,'command':['docker',str(here/'smoke_worker.py'),'--out',str(run)],'worker_sha256':'x','launcher_sha256':'x','approval_bundle':{'approved':True}}
        exited={'controller':ident,'container_identity':{'container_id':'cid','host_init':host},'returncode':0,
                'container_exit_code':0,'timed_out':False,'interrupted':None,'container_stopped_verified':True,
                'client_exited_while_container_running':False,'gpu_pids_after':[]}
        inspection=types.SimpleNamespace(stdout='[{"Id":"cid","State":{"Running":false,"ExitCode":0}}]')
        with patch.object(verify_k256,'sha',return_value='x'),patch.object(verify_k256,'alive',return_value=False), \
             patch.object(smoke_launch,'check_approval',return_value={'approved':True}),patch.object(verify_k256.subprocess,'run',return_value=inspection):
            verify_k256.validate_smoke_terminal(run,started,terminal,launch,exited)
            bad=copy.deepcopy(exited);bad['client_exited_while_container_running']=True
            with self.assertRaises(ValueError):verify_k256.validate_smoke_terminal(run,started,terminal,launch,bad)
            bad=copy.deepcopy(terminal);bad['worker_returncode']=1
            with self.assertRaises(ValueError):verify_k256.validate_smoke_terminal(run,started,bad,launch,exited)
            with patch.object(smoke_launch,'check_approval',side_effect=ValueError('unapproved')):
                with self.assertRaises(ValueError):verify_k256.validate_smoke_terminal(run,started,terminal,launch,exited)

    def test_independent_all_budget_tensor_position_and_decode_checks(self):
        arrays={name:(np.arange(4096,dtype=np.uint8).reshape(1024,4)+i).astype(np.uint8)
                for i,name in enumerate(('visual','ds0','ds1','ds2'))}
        def meta(a,n):return {'dtype':'torch.bfloat16','shape':[n,2],'sha256':__import__('hashlib').sha256(a.tobytes()).hexdigest()}
        class T:
            def decode(self,ids,**kw):return 'answer' if ids and all(x==7 for x in ids) else 'other'
        full=[meta(arrays[n],1024) for n in arrays]
        positions=[[[i for i in range(1027)]] for _ in range(3)]
        reference={'processor':{'identity':'same'},'visual_sha256':full[0],'deepstack_full_sha256':full[1:],
                   'original_position_ids':positions,'full_rope_delta':[[0]],'image_start':2,'original_length':1027,
                   'generation_scalars':{'max_new_tokens':2048,'do_sample':False},'eos_ids':[7]}
        for k in (128,256,512,1024):
            for on in (False,True):
                with self.subTest(k=k,on=on):
                    ordered=list(range(k-1,-1,-1)) if k<1024 else list(range(1024));spatial=sorted(ordered)
                    retained=[0,1]+[2+i for i in spatial]+[1026]
                    picked=[meta(a[spatial],k) for a in arrays.values()]
                    prefill=np.concatenate((np.zeros((2,4),dtype=np.uint8),arrays['visual'][spatial],np.zeros((1,4),dtype=np.uint8)))[None,:,:]
                    stored=dict(arrays,prefill=prefill)
                    if on:stored.update({'injected'+str(i):arrays['ds'+str(i)][spatial] for i in range(3)})
                    rec={'budget':k,'ds_on':on,'image_start':2,'original_length':1027,'ordered_ids':ordered,'spatial_ids':spatial,
                         'greedy_calls':0 if k==1024 else 1,'selection_mode':'keep_all' if k==1024 else 'author_greedy',
                         'retained_ids':retained,'full_position_ids':positions,'prefill_position_ids':[[[x[0][i] for i in retained]] for x in positions],
                         'full_delta':[[0]],'adjusted_delta':[[1024-k]],'processor':reference['processor'],'main_visual':full[0],
                         'full_deepstack':full[1:],'pruned_visual':picked[0],'selected_deepstack':picked[1:],
                         'injected_mask':[[False,False]+[True]*k+[False]] if on else None,
                         'injected_deepstack':picked[1:] if on else None,'output_ids':[7],'answer':'answer',
                         'generation_scalars':reference['generation_scalars'],'eos_ids':[7],'max_new_tokens':2048,
                         'prefill_mask':{'dtype':'torch.int64','shape':[1,len(retained)],
                           'sha256':__import__('hashlib').sha256(np.ones((1,len(retained)),dtype=np.int64).tobytes()).hexdigest()},
                         'prefill_embeds':{'dtype':'torch.bfloat16','shape':[1,len(retained),2],
                           'sha256':__import__('hashlib').sha256(prefill.tobytes()).hexdigest()},
                         'visual_calls':1,'main_merger_calls':1,'ds_calls':[1,1,1],'block_calls':[1,1],
                         'actual_injection_verified':True}
                    rec['forced']=False
                    rec['decode_trace']=[{'cache_position':list(range(len(retained))),'rope_delta':[[1024-k]],
                        'layer_cache_lengths':[],'layer_cache_lengths_after':[len(retained)]*36,'position_ids':rec['prefill_position_ids']}]
                    verify_k256.validate_budget_record(rec,reference,stored,T())
                    summary=copy.deepcopy(rec)
                    def text_meta(a):return {'dtype':'torch.bfloat16','shape':[1,a.shape[1],2],
                        'sha256':__import__('hashlib').sha256(a.tobytes()).hexdigest()}
                    summary['prefill_text_before']=text_meta(prefill[:,:2]);summary['prefill_text_after']=text_meta(prefill[:,2+k:])
                    tensors={'visual':full[0],'prefill':rec['prefill_embeds'],**{'ds'+str(i):full[i+1] for i in range(3)}}
                    if on:tensors.update({'injected'+str(i):picked[i+1] for i in range(3)})
                    summary['arrays']={'kind':'summary_only','tensors':tensors}
                    verify_k256.validate_budget_summary(summary,reference,T())
                    bad_summary=copy.deepcopy(summary);bad_summary['arrays']['kind']='full_arrays'
                    with self.assertRaisesRegex(ValueError,'summary-only'):verify_k256.validate_budget_summary(bad_summary,reference,T())
                    for field in ('prefill_position_ids','selected_deepstack','answer'):
                        bad=copy.deepcopy(rec);bad[field]='wrong'
                        with self.assertRaises((ValueError,TypeError,IndexError)):verify_k256.validate_budget_record(bad,reference,stored,T())
                    if on:
                        bad=copy.deepcopy(stored);bad['injected0'][0,0]^=1
                        with self.assertRaisesRegex(ValueError,'injected DS'):verify_k256.validate_budget_record(rec,reference,bad,T())
                    if k==128 and on:
                        forced=copy.deepcopy(rec);forced.update(forced=True,forced_token=7,output_ids=[7]*33,eos_ids=[],max_new_tokens=33)
                        forced['generation_scalars']=dict(rec['generation_scalars'],max_new_tokens=33)
                        for i in range(32):
                            pos=len(retained)+i
                            forced['decode_trace'].append({'cache_position':[pos],'position_ids':[[[pos+896]]]*3,
                                'rope_delta':[[896]],'layer_cache_lengths':[pos]*36,'layer_cache_lengths_after':[pos+1]*36})
                        extras={};processor={}
                        for name,shape,dtype,size in [('input_ids',[1,2],'torch.int64',16),('attention_mask',[1,2],'torch.int64',16),
                                ('pixel_values',[2,4],'torch.bfloat16',16),('image_grid_thw',[1,3],'torch.int64',24)]:
                            arr=np.arange(size,dtype=np.uint8).reshape(1,size);extras['processor_'+name]=arr
                            processor[name]={'dtype':dtype,'shape':shape,'sha256':__import__('hashlib').sha256(arr.tobytes()).hexdigest()}
                        logits=np.arange(200,dtype=np.uint8).reshape(1,200);extras['prefill_logits']=logits
                        forced['prefill_logits']={'dtype':'torch.bfloat16','shape':[1,100],
                            'sha256':__import__('hashlib').sha256(logits.tobytes()).hexdigest()}
                        forced['processor']=processor;reference_forced=dict(reference,processor=processor)
                        forced_arrays=dict(stored,**extras)
                        verify_k256.validate_budget_record(forced,reference_forced,forced_arrays,T())
                        bad=copy.deepcopy(forced_arrays);bad['prefill_logits'][0,0]^=1
                        with self.assertRaisesRegex(ValueError,'diagnostic original tensor'):verify_k256.validate_budget_record(forced,reference_forced,bad,T())
                        bad=copy.deepcopy(forced_arrays);bad['prefill_logits'][0,:2]=[0x80,0x7f]
                        bad_record=copy.deepcopy(forced)
                        bad_record['prefill_logits']['sha256']=__import__('hashlib').sha256(bad['prefill_logits'].tobytes()).hexdigest()
                        with self.assertRaisesRegex(ValueError,'nonfinite prefill logits'):
                            verify_k256.validate_budget_record(bad_record,reference_forced,bad,T())

    def test_approval_covers_all_dependencies_and_rejects_replacement(self):
        names=('smoke_launch.py','smoke_worker.py','smoke_contract.py','budget_runner.py','budget_adapter.py',
               'off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py','verify_k256.py',
               'decode_observer.py','test_budget_adapter.py','test_off_adapter.py','test_skip_deepstack.py','run_cpu_checks.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);here=root/'e4';here.mkdir();(root/'e1_analysis').mkdir();(root/'e1').mkdir()
            paths=[here/name for name in names]+[root/'e1_analysis/completion.py',root/'e1_analysis/score.py',
                root/'e1_analysis/metrics.py',root/'e1_analysis/paired_stats.py',root/'e1/contract.py']
            import run_cpu_checks
            external=[root/x for x in ('author.py','pruner.py','vision.py','config.py')];paths += external
            launch_deps=[root/x for x in ('plan.json','panel.py','diagnose.py','runtime.py')];paths += launch_deps
            for path in paths:path.write_text(path.name)
            digests={str(path):verify_k256.sha(path) for path in paths}
            cpu=root/'cpu.json';review=root/'review.json'
            cpu.write_text(json.dumps({'status':'passed','timeout':False,'returncode':0,'source_sha256_before':digests,'source_sha256_after':digests}))
            review.write_text(json.dumps({'status':'passed','independent_review':True,'cpu_proof_sha256':verify_k256.sha(cpu),'source_sha256':digests}))
            manifest=root/'approval.json';manifest.write_text(json.dumps({'cpu_proof':str(cpu),'review_proof':str(review)}))
            with patch.object(smoke_launch,'HERE',here),patch.object(smoke_launch,'APPROVAL_PATH',manifest), \
                 patch.object(run_cpu_checks,'AUTHOR',external[0]),patch.object(run_cpu_checks,'PRUNER',external[1]), \
                 patch.object(run_cpu_checks,'VISION',external[2]),patch.object(run_cpu_checks,'CONFIG',external[3]), \
                 patch.object(run_cpu_checks,'LAUNCH_DEPENDENCIES',launch_deps):
                smoke_launch.check_approval()
                (here/'budget_adapter.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'unapproved dependency'):smoke_launch.check_approval()

if __name__ == '__main__': unittest.main()
