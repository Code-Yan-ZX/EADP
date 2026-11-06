import copy
import unittest
from compare import compare_one, summarize, REQUIRED_CHECKS

def example(task='DocVQA_VAL', qid='0'):
    anchor={'task':task, 'qid':qid, 'group_id':task+qid, 'image_sha256':'i', 'message_sha256':'m'}
    rec={**anchor, 'processor_identity':{'p':1}, 'pixelprune_config':{'threshold':'0.2'},
         'generation_kwargs':{'do_sample':False}, 'eos_token_ids':[2], 'selected_indices':[0,3],
         'selected_count':2, 'sequence_length':2, 'generated_count':2, 'ends_eos':True,
         'decoded':'answer', 'status':'completed', 'diagnostic_checks':{k:True for k in REQUIRED_CHECKS}}
    rec['wavefront_selector_calls']=1;rec['diagnostic_checks']['wavefront_called_once']=True
    gen={'prompt_length':2,'full_sequence_ids':[11,12,17,2],'generated_token_ids':[17,2]}
    return anchor, rec, gen

class ContractTests(unittest.TestCase):
    def test_equal_and_different_outputs_are_preserved(self):
        a,b,g=example(); c=copy.deepcopy(b); h=copy.deepcopy(g)
        self.assertTrue(compare_one(a,b,g,c,h)['output_tokens_equal'])
        c['selected_indices']=[0,4]; h['generated_token_ids']=[18,2]; h['full_sequence_ids']=[11,12,18,2]
        r=compare_one(a,b,g,c,h)
        self.assertFalse(r['selection_order_equal']); self.assertFalse(r['output_tokens_equal']); self.assertTrue(r['decoded_equal'])

    def test_identity_and_recipe_change_rejected(self):
        a,b,g=example()
        for field in ('task','qid','group_id','image_sha256','message_sha256','processor_identity','pixelprune_config','generation_kwargs','eos_token_ids'):
            c=copy.deepcopy(b); c[field]='bad'
            with self.subTest(field=field), self.assertRaises(ValueError):compare_one(a,b,g,c,g)

    def test_incomplete_or_unpreserved_generation_rejected(self):
        a,b,g=example(); c=copy.deepcopy(b); c['diagnostic_checks']['positions_and_kv']=False
        with self.assertRaises(ValueError):compare_one(a,b,g,c,g)
        h=copy.deepcopy(g); h['full_sequence_ids'][-1]=19
        with self.assertRaises(ValueError):compare_one(a,b,g,b,h)
        c=copy.deepcopy(b);c['wavefront_selector_calls']=0
        with self.assertRaises(ValueError):compare_one(a,b,g,c,g)

    def test_all_384_in_order_no_filtering(self):
        anchors=[]; results=[]
        for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
            for i in range(128):
                a,b,g=example(task,str(i)); c=copy.deepcopy(b)
                if i==0: c['selected_indices']=[0,4]
                anchors.append(a); results.append(compare_one(a,b,g,c,g))
        result=summarize(anchors,results)
        self.assertEqual(result['questions'],384);self.assertEqual(result['equal_counts']['selection_order_equal'],381)
        with self.assertRaises(ValueError):summarize(anchors,results[:-1])
        with self.assertRaises(ValueError):summarize(anchors,list(reversed(results)))
        bad=copy.deepcopy(results);bad[0]['decoded_equal']=None
        with self.assertRaises(ValueError):summarize(anchors,bad)

if __name__=='__main__':unittest.main()
