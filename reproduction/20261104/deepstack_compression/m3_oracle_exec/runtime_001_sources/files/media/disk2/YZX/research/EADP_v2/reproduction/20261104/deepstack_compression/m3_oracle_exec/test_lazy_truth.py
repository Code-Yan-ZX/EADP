import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('lazy_truth_test',Path(__file__).resolve().parent/'lazy_truth.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class LazyTests(unittest.TestCase):
    def test_only_frozen_target_is_parsed_and_source_changes_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.tsv';p.write_text('index\tanswer\n1\t[\'first\', \'other\']\n2\t[\'second\']\n')
            digest=hashlib.sha256(p.read_bytes()).hexdigest();sample=[{'task':'DocVQA_VAL','qid':'1'}]
            loader=m.LazyTruth({'DocVQA_VAL':{'path':str(p),'sha256':digest}},sample)
            self.assertEqual(loader.index,{})
            with self.assertRaisesRegex(ValueError,'frozen sample'):loader({'task':'DocVQA_VAL','qid':'2'})
            self.assertEqual(loader({'task':'DocVQA_VAL','qid':'1'}),{'answers':['first','other']})
            p.write_text('index\tanswer\n1\tchanged\n')
            with self.assertRaisesRegex(ValueError,'changed'):loader({'task':'DocVQA_VAL','qid':'1'})
if __name__=='__main__':unittest.main()
