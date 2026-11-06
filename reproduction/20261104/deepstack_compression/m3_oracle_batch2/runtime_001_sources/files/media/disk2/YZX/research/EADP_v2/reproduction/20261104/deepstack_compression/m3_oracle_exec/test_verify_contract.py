import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import search,verify_contract
class VerifyTests(unittest.TestCase):
    def test_replay_and_tamper_rejection(self):
        row={'qid':'x'};order=list(range(256));values=[[float(i) for i in range(1024)] for _ in range(4)];meta={'sha256':'a'*64};pos=[[[i for i in range(1026)]] for _ in range(3)]
        e1={'ordered_ids':order,'visual_sha256':meta,'deepstack_full_sha256':[meta]*3,'logits':[meta],
            'eos_ids':[2],'generation_scalars':{'max_new_tokens':2048},'processor':{'input_ids':meta},
            'image_start':1,'original_length':1026,'original_position_ids':pos,'full_rope_delta':[[-2]]}
        binding={'qid':'x','arm':'P3-D1','verified':True,'record_sha256':'a'*64,'acceptance_sha256':'b'*64,'inputs_sha256':'c'*64}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'q';j=search.Journal(path)
            j.append('e1_baseline_first_logits',actual=meta,expected=meta,identical=True);j.append('e1_baseline_verified',**binding)
            j.append('reference_access_started');j.append('target_encoded');j.append('full_gradient_completed',mean_nll=1.,gradient_streams=4,utility=values,
                evidence={'features':[meta]*4,'gradients':[{'shape':[1024,4096],'sha256':'b'*64}]*4,'gradient_nonzero_counts':[1]*4})
            def generate(ids,callback):
                raw={'token_ids':[2],'eos_ids':[2],'generation_scalars':e1['generation_scalars']};callback(raw)
                retained=[0]+[1+i for i in ids]+[1025]
                evidence={'processor':e1['processor'],'full_streams':[meta]*4,'selected_ds':[meta]*3,'spatial_ids':ids,'retained_ids':retained,
                    'position_ids':[[[a[0][i] for i in retained]] for a in pos],'rope_delta':[[766]],'ds_mask':[[False]+[True]*256+[False]],
                    'decode_trace':[{'position_ids':None,'cache_position':list(range(258)),'rope_delta':[[766]],'layer_cache_lengths':[],
                                     'layer_cache_lengths_after':[258]*36}]}
                return {**raw,'eos':True,'truncated':False,'answer':'','actual_selection_verified':True,'evidence':evidence}
            search.execute('x',order,values,lambda _:1.,generate,j,binding,lambda:None);j.close()
            events=[json.loads(x) for x in path.read_text().splitlines()]
        result=verify_contract.validate(row,e1,events,lambda ids:'');self.assertEqual(result['new_generations'],6)
        bad=json.loads(json.dumps(events));next(x for x in bad if x['event']=='e1_baseline_first_logits')['identical']=False
        with self.assertRaises(ValueError):verify_contract.validate(row,e1,bad,lambda ids:'')
        bad=json.loads(json.dumps(events));next(x for x in bad if x['event']=='generation_completed')['output']['evidence']['ds_mask'][0][0]=True
        with self.assertRaises(ValueError):verify_contract.validate(row,e1,bad,lambda ids:'')
        with self.assertRaises(ValueError):verify_contract.validate(row,e1,events,lambda ids:'different')
if __name__=='__main__':unittest.main()
