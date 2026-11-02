"""Synthetic CPU verifier tests; no model, GPU, labels, or scoring."""
import copy
import unittest
import numpy as np
import verifier as v


def fixture():
    ids=np.array([[1,4,-200,8,9]],np.int64)
    visual=(np.arange(576*4096,dtype=np.float32).reshape(1,576,4096)%1024).astype(np.float16)
    order=np.arange(127,-1,-1,dtype=np.int64)[None,:];mask=np.zeros((1,576),np.bool_);mask[0,:128]=True
    before=np.arange(2*4096,dtype=np.float32).reshape(2,4096).astype(np.float16)
    after=np.arange(2*4096,dtype=np.float32).reshape(2,4096).astype(np.float16)
    embed=np.concatenate((before,visual[0,:128],after),axis=0)[None,:,:]
    d=dict(actual_input_ids=ids,actual_input_fp16=np.zeros((1,3,336,336),np.float16),projected_fp16=visual,
       image_embeds=np.zeros((1,576,768),np.float32),visual_sim=np.zeros((1,576,576),np.float32),
       final_mask=mask,internal_ids=order,quotas=np.array([128],np.int64),prepared_embeddings_fp16=embed,
       text_global=np.ones((1,768),np.float32),text_dense=np.ones((1,77,768),np.float32),
       global_sim=np.ones((1,576,1),np.float32),dense_sim=np.ones((1,576,1,77),np.float32),importance=np.ones((1,576),np.float32),
       output_ids=np.array([[1,43,2]],np.int64),actual_generate_input_ids=ids.copy(),clip_input_ids=np.zeros((1,77),np.int64),clip_attention_mask=np.zeros((1,77),np.int64))
    return d

class Tests(unittest.TestCase):
    def setUp(self):self.a=fixture();self.b=copy.deepcopy(self.a)
    def test_valid_real_selection_and_prepared_embedding(self):
        r=v.check_selection(self.a['final_mask'],self.a['internal_ids'],self.a['quotas'])
        self.assertEqual(r['returned_retained_tokens'],128)
        _,r=v.check_embeddings(self.a['prepared_embeddings_fp16'],self.a['projected_fp16'],self.a['final_mask'],self.a['actual_input_ids'])
        self.assertEqual((r['marker_index'],r['n_visual']),(2,128));self.assertFalse(v.check_pair(self.a,self.b)['final_mask_changed'])
    def test_shapes_and_nonfinite_rejected(self):
        v.check_shapes(self.a,1)
        for key in ('image_embeds','importance','visual_sim'):
            x=copy.deepcopy(self.a);x[key].reshape(-1)[0]=float('nan')
            with self.assertRaises(AssertionError):v.check_shapes(x,1)
        x=copy.deepcopy(self.a);x['actual_generate_input_ids'][0,0]+=1
        with self.assertRaises(AssertionError):v.check_shapes(x,1)
    def test_wrong_selected_ids_quota_or_mask_rejected(self):
        for key in ('final_mask','internal_ids','quotas'):
            x=copy.deepcopy(self.a)
            if key=='final_mask':x[key][0,0]=False
            else:x[key].reshape(-1)[0]+=1
            with self.assertRaises(AssertionError):v.check_selection(x['final_mask'],x['internal_ids'],x['quotas'])
    def test_visual_must_match_actual_selected_spatial_rows(self):
        x=copy.deepcopy(self.a);x['prepared_embeddings_fp16'][0,2,0]+=1
        with self.assertRaises(AssertionError):v.check_embeddings(x['prepared_embeddings_fp16'],x['projected_fp16'],x['final_mask'],x['actual_input_ids'])
    def test_nonvisual_and_source_image_must_match(self):
        for key in ('actual_input_fp16','actual_input_ids','projected_fp16','image_embeds','visual_sim'):
            x=copy.deepcopy(self.b);x[key].reshape(-1)[0]+=1
            with self.assertRaises(AssertionError):v.check_pair(self.a,x)
        x=copy.deepcopy(self.b);x['prepared_embeddings_fp16'][0,0,0]+=1
        with self.assertRaises(AssertionError):v.check_pair(self.a,x)
    def test_guidance_sensitive_change_is_observation_not_failure(self):
        self.b['importance'][0,0]+=1;self.b['output_ids']=np.array([[1,12,2]],np.int64)
        result=v.check_pair(self.a,self.b);self.assertTrue(result['importance_changed']);self.assertTrue(result['output_ids_changed'])
    def test_actual_clip_segment_hash_and_padding(self):
        ids=np.zeros((2,77),np.int64);ids.reshape(-1)[:80]=np.arange(80)
        mask=np.zeros((2,77),np.int64);mask.reshape(-1)[:80]=1
        ref=dict(n_segments=2,n_tokens=80,input_ids_sha256=v.array_sha(ids),attention_mask_sha256=v.array_sha(mask))
        v.check_clip_inputs(ids,mask,ref)
        bad=ids.copy();bad[0,0]=9
        with self.assertRaises(AssertionError):v.check_clip_inputs(bad,mask,ref)
        bad=mask.copy();bad[1,5]=1
        with self.assertRaises(AssertionError):v.check_clip_inputs(ids,bad,ref)
    def test_real_manifest_historical_inputs_and_tampering(self):
        import common as c
        m=c.read(c.HERE/'inputs/manifest.json');self.assertEqual(len(v.check_manifest(c,m)),60)
        changed=copy.deepcopy(m);changed['rows'][0]['accepted_source']['text']+='tamper'
        with self.assertRaises(AssertionError):v.check_manifest(c,changed)
    def test_case_checks_archive_raw_and_output_identity(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);a=copy.deepcopy(self.a);a['clip_input_ids'][0,:3]=[2,3,4];a['clip_attention_mask'][0,:3]=1
            ref=dict(n_tokens=3,n_segments=1,text_sha256=v.text_sha('full'),input_ids=a['clip_input_ids'].tolist(),attention_mask=a['clip_attention_mask'].tolist(),input_ids_sha256=v.array_sha(a['clip_input_ids']),attention_mask_sha256=v.array_sha(a['clip_attention_mask']))
            item=dict(index=7,group_id=7,full_guidance='full',question_only_guidance='question',source_clip=ref,llm_input_ids=a['actual_input_ids'][0].tolist(),llm_n_tokens=5,llm_input_ids_sha256=v.array_sha(a['actual_input_ids']),image=dict(size=[10,10],pixel_float16_sha256=v.array_sha(a['actual_input_fp16'])),accepted_source=dict(text='A'))
            label='source_00';raw=dict(arm='source',position=0,index=7,group_id=7,native_returned=True,native_failure=None,output_ids=a['output_ids'].tolist(),response='A',returned_visual_token_num=128)
            ids=dict(arm='source',position=0,index=7,native_returned=True,output_ids=a['output_ids'].tolist())
            (out/(label+'.raw.json')).write_text(json.dumps(raw));(out/(label+'.output_ids.json')).write_text(json.dumps(ids))
            np.savez_compressed(out/(label+'.npz'),**a)
            def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            def read(path):return json.loads(Path(path).read_text())
            def desc(path):return dict(path=str(path),sha256=sha(path))
            rec=dict(status='captured_unvalidated',arm='source',position=0,index=7,group_id=7,calls=dict(encode=1,cross=1,greedy=1,prepare=1,clip=1),observer_errors=[],generate_scalar_kwargs=v.PARAMETERS,generate_keys=sorted(['images','image_sizes','texts',*v.PARAMETERS]),positional_args_count=1,guidance='full',encode_texts=['full'],image_sizes=[[10,10]],returned_visual_token_num=128,raw=desc(out/(label+'.raw.json')),output_ids_file=desc(out/(label+'.output_ids.json')),archive=desc(out/(label+'.npz')),response='A',arrays={k:dict(dtype=str(x.dtype),shape=list(x.shape),sha256=v.array_sha(x)) for k,x in a.items()})
            path=out/(label+'.json');path.write_text(json.dumps(rec))
            c=SimpleNamespace(OUT=out,sha=sha,read=read);author=dict(question_id=7,round_id=0,prompt='full',text='A');runtime=dict(eos_token_id=2,bos_token_id=1,vocab_size=100)
            v.check_case(c,'source',0,item,desc(path),author,runtime)
            (out/(label+'.output_ids.json')).write_text(json.dumps({**ids,'output_ids':[[1,44,2]]}))
            with self.assertRaises(AssertionError):v.check_case(c,'source',0,item,desc(path),author,runtime)
    def test_missing_eos_and_limit_are_reported(self):
        self.assertTrue(v.check_output(np.array([[1,3,2]],np.int64),100,1,[2])['EOS_at_end'])
        result=v.check_output(np.array([[1]+[3]*1024],np.int64),100,1,[2])
        self.assertTrue(result['at_max_tokens']);self.assertFalse(result['EOS_present'])
        with self.assertRaises(AssertionError):v.check_output(np.array([[1]],np.int64),100,1,[2])
    def test_bitwise_comparison_distinguishes_signed_zero(self):
        self.assertFalse(v.bits(np.array([0.],np.float32),np.array([-0.],np.float32)))

if __name__=='__main__':unittest.main(verbosity=2)
