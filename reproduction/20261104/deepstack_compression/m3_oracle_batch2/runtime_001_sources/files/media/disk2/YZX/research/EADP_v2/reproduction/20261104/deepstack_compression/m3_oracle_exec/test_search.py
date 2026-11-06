import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parent/'search.py';spec=importlib.util.spec_from_file_location('search_under_test',path)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def binding():return {'qid':'synthetic','arm':'P3-D1','verified':True,'record_sha256':'a'*64,'acceptance_sha256':'b'*64,'inputs_sha256':'c'*64}

class SearchTests(unittest.TestCase):
    def test_equal_budget_dedup_ties_and_six_new_exposures(self):
        order=list(range(256));values=[[float(i) for i in range(1024)] for _ in range(4)]
        calls=[];generated=[]
        with tempfile.TemporaryDirectory() as d:
            j=s.Journal(Path(d)/'journal.jsonl')
            def evaluate(ids):calls.append(ids);return 1.0
            def generate(ids,callback):
                generated.append(ids);value={'token_ids':[],'eos':True,'truncated':False};callback(value);return value
            result=s.execute('synthetic',order,values,evaluate,generate,j,binding(),lambda:None);j.close()
            rows=[json.loads(x) for x in (Path(d)/'journal.jsonl').read_text().splitlines()]
        self.assertEqual(len(result['winners']),6);self.assertEqual(len(generated),6)
        self.assertTrue(all(w['proposal_index']==-1 and w['n_proposals']==8 for w in result['winners']))
        self.assertEqual(len(calls),len({tuple(x) for x in calls}))
        self.assertEqual(sum(r['event']=='generation_started' for r in rows),6)
        self.assertEqual(sum(r['event']=='generation_returned' for r in rows),6)
        self.assertEqual(rows[-1]['event'],'question_completed')
    def test_failure_exposure_is_durable_and_no_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'journal';j=s.Journal(path)
            with self.assertRaisesRegex(ValueError,'endpoint'):
                s.execute('synthetic',list(range(256)),[[float(i) for i in range(1024)]]*4,
                          lambda _:0.,lambda _,cb:(cb({'token_ids':[7]}),{'token_ids':[7],'eos':False,'truncated':False})[1],j,binding(),lambda:None)
            j.close();rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'],'generation_returned');self.assertEqual(rows[-1]['output']['token_ids'],[7])
            with self.assertRaises(FileExistsError):s.Journal(path)
    def test_nonfinite_evaluation_stops_before_generation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'journal';j=s.Journal(path)
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                s.execute('synthetic',list(range(256)),[[float(i) for i in range(1024)]]*4,
                          lambda _:float('nan'),lambda *_:self.fail('unexpected generation'),j,binding(),lambda:None)
            j.close();rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'],'evaluation_started')

if __name__=='__main__':unittest.main()
