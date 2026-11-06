import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('oracle_run_control',Path(__file__).resolve().parent/'run_control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class ControlTests(unittest.TestCase):
    def test_wall_and_question_deadline_fail_closed(self):
        v={'wall':10,'mono':20};d=c.Deadline(100,30,wall=lambda:v['wall'],clock=lambda:v['mono']);d.check()
        v['mono']=50
        with self.assertRaises(TimeoutError):d.check()
        v['mono']=20;v['wall']=100
        with self.assertRaises(TimeoutError):d.check()
if __name__=='__main__':unittest.main()
