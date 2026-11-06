import importlib.util
import io
from pathlib import Path
import unittest

import pandas as pd


class PromptTest(unittest.TestCase):
    def test_pandas_na_option_is_not_given_to_ai2d_prompt(self):
        path=Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py')
        spec=importlib.util.spec_from_file_location('author_prompt_test',path)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        obj=mod.Qwen3VLPromptMixin(use_custom_prompt=True)
        obj.set_dump_image(lambda row:['/tmp/fake-image'])
        data=pd.read_csv(io.StringIO('question\tA\tB\tC\tD\nquestion?\ta\tb\tc\tNA\n'),sep='\t')
        message=obj._build_mcq_prompt(data.iloc[0].to_dict(),'AI2D_TEST')
        self.assertNotIn('D. NA',message[-1]['value'])
        self.assertIn('C. c',message[-1]['value'])

if __name__=='__main__': unittest.main()
