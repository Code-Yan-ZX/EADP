import hashlib,importlib.util,json
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e1_admission
spec=importlib.util.spec_from_file_location('e1_baseline_test',Path(__file__).resolve().parent/'e1_baseline.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
class E1BaselineTests(unittest.TestCase):
    def test_unique_sample_and_full_accepted_hashes(self):
        rows=[];ins=[]
        for i in range(384):
            ins.append({'task':'fake','qid':str(i),'group_id':str(i),'image_path':'x','image_sha256':'a','message_sha256':'b'})
            for arm in ('P1-D0','P1-D1','P3-D0','P3-D1'):
                rows.append({'task':'fake','qid':str(i),'group_id':str(i),'arm':arm,'forced':False})
        journal=('\n'.join(json.dumps(x) for x in rows)+'\n').encode();inputs=('\n'.join(json.dumps(x) for x in ins)+'\n').encode()
        h=lambda x:hashlib.sha256(x).hexdigest()
        proof={'status':'passed','stage':'full','natural_questions':384,'natural_records':1536,'tokenizer_decode_checked':1536,'GT_opened':False,'scoring_performed':False,
               'files':{'journal':{'sha256':h(journal)},'inputs':{'sha256':h(inputs)}}}
        receipt=e1_admission.Receipt(h(json.dumps(proof).encode()),h(journal),h(inputs),'d'*64,'e'*64,'f'*64)
        line,row,p=e.select(json.dumps(proof).encode(),journal,inputs,ins[0],receipt);self.assertEqual(json.loads(line)['arm'],'P3-D1');self.assertEqual(row,ins[0]);self.assertEqual(p['journal_sha256'],h(journal))
        with self.assertRaisesRegex(ValueError,'hash mismatch'):e.select(json.dumps(proof).encode(),journal+b' ',inputs,ins[0],receipt)
        bad=dict(ins[0],message_sha256='different')
        with self.assertRaisesRegex(ValueError,'input mismatch'):e.select(json.dumps(proof).encode(),journal,inputs,bad,receipt)
        rows[7]=dict(rows[3]);journal=('\n'.join(json.dumps(x) for x in rows)+'\n').encode();proof['files']['journal']['sha256']=h(journal)
        receipt=e1_admission.Receipt(h(json.dumps(proof).encode()),h(journal),h(inputs),'d'*64,'e'*64,'f'*64)
        with self.assertRaisesRegex(ValueError,'duplicate'):e.select(json.dumps(proof).encode(),journal,inputs,ins[0],receipt)
if __name__=='__main__':unittest.main()
