"""Synthetic CPU acceptance and tamper checks; no GPU, model, GT or score."""
import copy
import unittest
import numpy as np
import verify_generation as v

class Tests(unittest.TestCase):
    def test_actual_size_type_capture_rejects_tampering(self):
        record={'image_sizes_argument_types':{'outer':'list','inner':['tuple']}}
        capture={'image_sizes_outer_type':'list','image_sizes_inner_types':['tuple'],'image_sizes_actual_repr':'[(426, 283)]'}
        item={'image':{'size':[426,283]}}
        v.check_size_capture(record,capture,item)
        for field,value in [('image_sizes_actual_repr','[(283, 426)]'),('image_sizes_inner_types',['list']),('image_sizes_outer_type','tuple')]:
            bad=copy.deepcopy(capture);bad[field]=value
            with self.assertRaises(AssertionError):v.check_size_capture(record,bad,item)
    def test_real_selection_order_and_mask(self):
        ids=list(range(127,-1,-1));mask=[i<128 for i in range(576)]
        self.assertEqual(v.check_selection(ids,[128],mask),list(range(128)))
        for field in ('ids','quota','mask'):
            x=list(ids);q=[128];m=list(mask)
            if field=='ids':x[0]=x[1]
            elif field=='quota':q=[127]
            else:m[0]=False
            with self.assertRaises(AssertionError):v.check_selection(x,q,m)
    def test_clip_actual_segments_match_cpu_fingerprint(self):
        ids=[[0]*77,[0]*77];mask=[[0]*77,[0]*77]
        for i in range(80):ids[i//77][i%77]=i+1;mask[i//77][i%77]=1
        ref=dict(n_tokens=80,n_segments=2,input_ids_sha256=v.int64_sha(sum(ids,[])),attention_mask_sha256=v.int64_sha(sum(mask,[])),input_ids=ids,attention_mask=mask)
        v.check_clip(ids,mask,ref)
        changed=copy.deepcopy(ids);changed[1][5]=2
        with self.assertRaises(AssertionError):v.check_clip(changed,mask,ref)
    def test_visual_and_nonvisual_metadata_contract(self):
        x=dict(dtype='float16',shape=[128,4096],sha256='a'*64);y=dict(dtype='float16',shape=[9,4096],sha256='b'*64)
        v.check_visual_relationship(x,copy.deepcopy(x),y,10)
        z=copy.deepcopy(x);z['sha256']='c'*64
        with self.assertRaises(AssertionError):v.check_visual_relationship(x,z,y,10)
        y['shape']=[8,4096]
        with self.assertRaises(AssertionError):v.check_visual_relationship(x,x,y,10)
    def test_output_eos_and_generation_limit(self):
        self.assertTrue(v.check_output([1,4,2],32000,1,[2])['EOS_at_end'])
        out=v.check_output([1]+[3]*1024,32000,1,[2]);self.assertTrue(out['at_max_tokens']);self.assertFalse(out['EOS_present'])
        with self.assertRaises(AssertionError):v.check_output([1],32000,1,[2])
    def test_reordering_is_exact_not_subset_or_modified(self):
        items=[dict(index=str(i)) for i in range(3)]
        original=[dict(position=i,index=str(i),response='a'+str(i)) for i in (2,0,1)]
        ordered=sorted(original,key=lambda x:x['position']);v.check_ordered(items,original,ordered)
        for key in ('position','index','response'):
            x=copy.deepcopy(ordered);x[1][key]='x'
            with self.assertRaises(AssertionError):v.check_ordered(items,original,x)
    def test_full_case_small_tensor_and_prefill_guards(self):
        import hashlib
        def meta(dtype,shape,digit):return dict(dtype=dtype,shape=shape,sha256=digit*64)
        ids=[[1,-200,5,6]];clip_ids=[[1,2,3,4,5]+[0]*72];clip_mask=[[1]*5+[0]*72]
        selected=[list(range(127,-1,-1))];mask=[[i<128 for i in range(576)]];quota=[128];out=[[1,9,2]]
        small=dict(actual_input_ids=ids,clip_input_ids=clip_ids,clip_attention_mask=clip_mask,internal_ids=selected,quotas=quota,final_mask=mask,output_ids=out)
        arrays={k:dict(dtype=('bool' if k=='final_mask' else 'int64'),shape=list(np.asarray(x).shape),sha256=v.array_sha(np.asarray(x,dtype=np.bool_ if k=='final_mask' else np.int64))) for k,x in small.items()}
        arrays['actual_generate_input_ids']=copy.deepcopy(arrays['actual_input_ids'])
        arrays['actual_input_fp16']=meta('float16',[1,3,336,336],'a');arrays['projected_fp16']=meta('float16',[1,576,4096],'b');arrays['prepared_embeddings_fp16']=meta('float16',[1,131,4096],'c')
        for key,shape in [('text_global',[1,768]),('text_dense',[1,77,768]),('image_embeds',[1,576,768]),('global_sim',[1,576,1]),('dense_sim',[1,576,1,77]),('importance',[1,576]),('visual_sim',[1,576,576])]:arrays[key]=meta('float32',shape,'d')
        clip=dict(text_sha256=hashlib.sha256(b'full').hexdigest(),n_tokens=5,n_segments=1,input_ids=clip_ids,attention_mask=clip_mask,input_ids_sha256=arrays['clip_input_ids']['sha256'],attention_mask_sha256=arrays['clip_attention_mask']['sha256'])
        item=dict(original_position=0,index=7,group_id=7,llm_input_ids=ids[0],llm_input_ids_sha256=arrays['actual_input_ids']['sha256'],image=dict(size=[10,10],pixel_float16_sha256='a'*64),full_guidance='full',question_only_guidance='short',source_clip=clip)
        d=dict(selected_visual=meta('float16',[128,4096],'e'),prepare_visual=meta('float16',[128,4096],'e'),prepare_nonvisual=meta('float16',[3,4096],'f'),marker_index=1)
        runtime=dict(vocab_size=100,bos_token_id=1,eos_token_id=2)
        record=dict(status='captured',arm='source',position=0,index=7,group_id=7,response='A',calls=dict(encode=1,cross=1,greedy=1,prepare=1,clip=1),observer_errors=[],generate_scalar_kwargs=v.PARAMETERS,generate_keys=sorted(['images','image_sizes','texts',*v.PARAMETERS]),positional_args_count=1,image_sizes=[[10,10]],image_sizes_argument_types=dict(outer='list',inner=['tuple']),returned_visual_token_num=128,guidance='full',encode_texts=['full'],arrays=arrays,small_arrays=small,derived=d,ending=v.check_output(out[0],100,1,[2]))
        raw=dict(arm='source',position=0,index=7,group_id=7,native_failure=None,native_returned=True,response='A',output_ids=out);idr={k:raw[k] for k in ('arm','position','index','group_id','native_returned','output_ids')}
        v.check_case(record,raw,idr,item,'source',runtime)
        for change in ('clip_attention_mask','final_mask','output_ids'):
            bad=copy.deepcopy(record);bad['small_arrays'][change][0][0]=0
            with self.assertRaises(AssertionError):v.check_case(bad,raw,idr,item,'source',runtime)
        bad=copy.deepcopy(record);bad['derived']['prepare_visual']['sha256']='0'*64
        with self.assertRaises(AssertionError):v.check_case(bad,raw,idr,item,'source',runtime)
    def test_pair_requires_shared_input_but_keeps_outcome_differences(self):
        def meta(shape,sha):return dict(dtype='float16',shape=shape,sha256=sha*64)
        arrays={k:dict(dtype='int64',shape=[1,4],sha256='c'*64) for k in ('actual_input_ids','actual_input_fp16','projected_fp16','image_embeds','visual_sim','importance')}
        record=dict(arrays=arrays,small_arrays=dict(actual_input_ids=[[1,-200,3,4]],final_mask=[[True]],internal_ids=[[1]],output_ids=[[1,2]]),derived=dict(selected_visual=meta([128,4096],'a'),prepare_visual=meta([128,4096],'a'),prepare_nonvisual=meta([3,4096],'b')),response='A')
        changed=copy.deepcopy(record);changed['response']='B';changed['small_arrays']['output_ids']=[[1,3,2]]
        result=v.check_pair(record,changed);self.assertTrue(result['response_changed'] and result['output_ids_changed'])
        changed['arrays']['visual_sim']['sha256']='d'*64
        with self.assertRaises(AssertionError):v.check_pair(record,changed)
    def test_small_tensor_value_and_type_guard(self):
        ids=[[1,2,3]];a=np.asarray(ids,dtype=np.int64)
        meta=dict(dtype='int64',shape=[1,3],sha256=v.array_sha(a))
        self.assertEqual(v.check_saved_small(ids,meta,'int64').tolist(),ids)
        with self.assertRaises(AssertionError):v.check_saved_small([[1,2,4]],meta,'int64')
        with self.assertRaises(AssertionError):v.check_saved_small([[1,2,True]],meta,'int64')
    def test_anchor_rechecks_real_bytes_and_metadata(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.npz';b=Path(tmp)/'b.npz';arr=np.array([0.,1.],dtype=np.float32)
            meta={'x':dict(dtype='float32',shape=[2],sha256=v.array_sha(arr))}
            np.savez(a,x=arr);np.savez(b,x=arr.copy());v.check_full_anchor(a,meta,b,meta)
            altered=arr.copy();altered[0]=-0.;np.savez(b,x=altered)
            with self.assertRaises(AssertionError):v.check_full_anchor(a,meta,b,meta)
    def test_real_completed_panel_archive_is_read_and_tamper_rejected(self):
        import json
        from pathlib import Path
        old=Path(__file__).resolve().parent.parent/'question_only_guidance_panel'/'execution'
        record=json.loads((old/'source_00.json').read_text());path=old/'source_00.npz'
        v.check_full_anchor(path,record['arrays'],path,record['arrays'])
        bad=copy.deepcopy(record['arrays']);bad['final_mask']['sha256']='0'*64
        with self.assertRaises(AssertionError):v.check_full_anchor(path,bad,path,record['arrays'])
    def test_completed_panel_real_case_adapter(self):
        import json
        from pathlib import Path
        panel=Path(__file__).resolve().parent.parent/'question_only_guidance_panel'
        item=json.loads((panel/'inputs/manifest.json').read_text())['rows'][0]
        old=json.loads((panel/'execution/source_00.json').read_text());raw_old=json.loads((panel/'execution/source_00.raw.json').read_text())
        runtime=json.loads((panel/'execution/runtime.json').read_text())
        with np.load(panel/'execution/source_00.npz',allow_pickle=False) as z:
            a={key:z[key] for key in z.files}
        pos=item['original_position'];mask=a['final_mask'][0];marker=a['actual_input_ids'][0].tolist().index(-200)
        visual=a['projected_fp16'][0,mask];prepared=a['prepared_embeddings_fp16'][0,marker:marker+128]
        nonvisual=np.concatenate((a['prepared_embeddings_fp16'][0,:marker],a['prepared_embeddings_fp16'][0,marker+128:]))
        def meta(x):return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=v.array_sha(x))
        d=dict(selected_visual=meta(visual),prepare_visual=meta(prepared),prepare_nonvisual=meta(nonvisual),marker_index=marker)
        small={k:a[k].tolist() for k in v.SMALL};out=small['output_ids'];ending=v.check_output(out[0],runtime['vocab_size'],runtime['bos_token_id'],[runtime['eos_token_id']])
        # Old JSON did not retain Python container types; this type is synthetic.
        rec={**old,'status':'captured','position':pos,'arrays':old['arrays'],'small_arrays':small,'derived':d,'ending':ending,'image_sizes_argument_types':dict(outer='list',inner=['tuple'])}
        raw={**raw_old,'position':pos};ids={k:raw[k] for k in ('arm','position','index','group_id','native_returned','output_ids')}
        v.check_case(rec,raw,ids,item,'source',runtime)
        rec['derived']['prepare_visual']['sha256']='0'*64
        with self.assertRaises(AssertionError):v.check_case(rec,raw,ids,item,'source',runtime)
    def test_bitwise_comparison_distinguishes_signed_zero(self):
        self.assertFalse(v.array_equal(np.array([0.],np.float32),np.array([-0.],np.float32)))

if __name__=='__main__':unittest.main(verbosity=2)
