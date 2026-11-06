import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
spec=importlib.util.spec_from_file_location('m3_one_question_test',Path(__file__).resolve().parent/'one_question.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
class QuestionTests(unittest.TestCase):
    def test_invalid_baseline_fails_before_reference_read(self):
        events=[];j=SimpleNamespace(append=lambda event,**kwargs:events.append((event,kwargs)))
        def forbidden(_):self.fail('reference opened before baseline')
        row={'task':'fake','qid':'1','group_id':'g','split':'DEV'};full={'streams':[1,2,3,4],'input_ids':torch.zeros((1,1))}
        with patch.object(o.baseline,'validate',side_effect=ValueError('P3 changed')):
            with self.assertRaisesRegex(ValueError,'P3 changed'):
                o.run(row,forbidden,None,full,None,None,None,None,SimpleNamespace(model=None),None,None,{},[],j,torch,b'{}',{},lambda:None)
        self.assertEqual(events[0][0],'e1_baseline_rejected');self.assertEqual(events[0][1]['message'],'P3 changed')
if __name__=='__main__':unittest.main()
