import copy
import unittest
import numpy as np
import torch
from local_imports import load_local
v=load_local('verify_interventions');digest=v.digest;expected_permutation=v.expected_permutation;validate_one=v.validate_one
validate_kv_reference=load_local('verify_smoke').validate_kv_reference
validate_e1_endpoint=load_local('verify_smoke').validate_e1_endpoint

class VerifyInterventions(unittest.TestCase):
    def make(self,family,arm):
        rng=np.random.default_rng(1);spatial=list(range(256));start=1;length=1026
        retained=list(range(257))+[1025];full=np.tile(np.arange(length),(3,1));delta=full.max()+1-length+768
        visual=rng.integers(0,256,(1024,4),dtype=np.uint8)
        streams=[rng.integers(0,256,(1024,4),dtype=np.uint8) for _ in range(3)]
        before=[x[spatial] for x in streams];image='0'*64;seed,perm=expected_permutation(image)
        inject=[(z if arm[i]=='1' else np.zeros_like(z)) if family=='E2' else z[perm] for i,z in enumerate(before)]
        arrays={'visual':visual,'prefill':np.concatenate([np.zeros((1,4),dtype=np.uint8),visual[spatial],np.zeros((1,4),dtype=np.uint8)])[None,:,:]}
        for i in range(3):arrays.update({'ds'+str(i):streams[i],'selected_before'+str(i):before[i],'injected'+str(i):inject[i]})
        meta=lambda x:{'sha256':digest(x),'dtype':'torch.float32','shape':[1024,1]}
        prefill=full[:,retained][:,None,:].tolist();n=len(retained);delta=int(delta)
        trace=[{'rope_delta':[[delta]],'layer_cache_lengths':[],'layer_cache_lengths_after':[n]*36,'cache_position':list(range(n)),'position_ids':prefill}]
        for i in range(1,33):
            cp=n+i-1;trace.append({'rope_delta':[[delta]],'layer_cache_lengths':[cp]*36,'layer_cache_lengths_after':[cp+1]*36,'cache_position':[cp],'position_ids':[[[cp+delta]]]*3})
        row={'family':family,'arm':arm,'budget':256,'ds_on':True,'spatial_ids':spatial,'ordered_ids':spatial[::-1],
             'image_start':start,'original_length':length,'retained_ids':retained,'full_position_ids':full[:,None,:].tolist(),
             'prefill_position_ids':prefill,'adjusted_delta':[[delta]],'output_ids':[5]*33,'decode_trace':trace,
             'actual_injection_verified':True,'injected_mask':[[False]+[True]*256+[False]],'image_sha256':image,
             'main_visual':meta(visual),'pruned_visual':meta(visual[spatial]),'prefill_embeds':meta(arrays['prefill']),
             'full_deepstack':[meta(x) for x in streams],'injected_deepstack':[meta(x) for x in inject],
             'selected_deepstack':[meta(x) for x in inject],'intervention_metadata':{'seed':seed,'permutation':perm},
             'greedy_calls':1,'visual_calls':1,'main_merger_calls':1,'block_calls':[1,1],'ds_calls':[1,1,1],
             'forced':True,'cache_fingerprints':[[{'key':{'sha256':'0'*64,'dtype':'torch.bfloat16','shape':[1,8,n+i,128]},'value':{'sha256':'1'*64,'dtype':'torch.bfloat16','shape':[1,8,n+i,128]}} for _ in range(36)] for i in range(33)] if family=='E2' and arm in ('000','111') else []}
        return row,arrays
    def test_arms(self):
        for family,arm in [('E2',x) for x in ('000','111','100','010','001')]+[('E3','PERM')]:
            r,a=self.make(family,arm);self.assertTrue(validate_one(r,a))
    def test_corruption(self):
        r,a=self.make('E2','111');bad=copy.deepcopy(r);bad['decode_trace'][20]['layer_cache_lengths'][4]+=1
        with self.assertRaises(ValueError):validate_one(bad,a)
        b={k:v.copy() for k,v in a.items()};b['injected1'][0,0]^=1
        with self.assertRaises(ValueError):validate_one(r,b)
        r,a=self.make('E3','PERM');r['intervention_metadata']['permutation'][0]=0
        with self.assertRaises(ValueError):validate_one(r,a)

    def test_mean(self):
        r,a=self.make('E2','111');r['family']='E3';r['arm']='MEAN';r['cache_fingerprints']=[]
        for i in range(3):
            values=torch.arange(256,dtype=torch.float32).reshape(256,1)+i
            a['selected_before'+str(i)]=values.view(torch.uint8).numpy()
            a['ds'+str(i)][:256]=a['selected_before'+str(i)]
            mean=values.mean(0,keepdim=True).expand_as(values).contiguous().view(torch.uint8).numpy()
            a['injected'+str(i)]=mean
            r['full_deepstack'][i]['sha256']=digest(a['ds'+str(i)])
            r['injected_deepstack'][i]['sha256']=digest(mean)
            r['selected_deepstack'][i]['sha256']=digest(mean)
        self.assertTrue(validate_one(r,a))
        a['injected0'][0,0]^=1
        with self.assertRaises(ValueError):validate_one(r,a)

    def test_kv_content_identity_rejects_corruption(self):
        keys=('processor','ordered_ids','spatial_ids','retained_ids','main_visual','pruned_visual','full_deepstack',
              'full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','prefill_logits','output_ids')
        a={k:'same' for k in keys};a['cache_fingerprints']=[[{'key':'abc','value':'def'} for _ in range(36)] for _ in range(33)]
        b=copy.deepcopy(a);validate_kv_reference(a,b)
        b['cache_fingerprints'][15][17]['value']='tampered'
        with self.assertRaisesRegex(ValueError,'KV content'):validate_kv_reference(a,b)

    def test_e1_prefill_logits_metadata_required(self):
        r={'output_ids':[1],'answer':'a','eos_ids':[1],'eos':True,'truncated':False,
           'prefill_logits':{'dtype':'torch.bfloat16','shape':[1,8],'sha256':'a'*64}}
        e={k:r[k] for k in ('output_ids','answer','eos_ids','eos','truncated')};e['logits']=[r['prefill_logits'].copy()]
        validate_e1_endpoint(r,e)
        for key,value in (('dtype','torch.float32'),('shape',[1,7]),('sha256','b'*64)):
            bad=copy.deepcopy(e);bad['logits'][0][key]=value
            with self.assertRaisesRegex(ValueError,'logits'):validate_e1_endpoint(r,bad)

if __name__=='__main__':unittest.main()
