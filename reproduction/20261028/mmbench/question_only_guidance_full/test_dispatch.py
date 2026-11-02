"""Execute author eval entry with synthetic CPU stubs; no model weights or CUDA."""
import ast, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import torch
import common as c

class Bar:
    def __init__(self,it,total):self.it=it
    def __iter__(self):return iter(self.it)
    def set_postfix(self,**kwargs):pass

class Tests(unittest.TestCase):
    def test_actual_entry_row_dispatch_and_only_guidance_changes(self):
        self.assertFalse(torch.cuda.is_initialized())
        sys.path.insert(0,str(c.SOURCE.parents[2]));sys.path.insert(0,str(c.HERE.parent/'question_only_guidance_panel_design'))
        import source_boundary as b
        code=ast.parse(c.SOURCE.read_bytes(),filename=str(c.SOURCE));seen=[];prompts=[];loads=[]
        frame=pd.DataFrame([dict(index=101,question='What is it?',hint='a hint',A='red',B='blue',C=None,D=None,image='unused'),dict(index=102,question='Second?',hint=None,A='yes',B='no',C=None,D=None,image='unused')])
        class Model:
            config=SimpleNamespace(mm_use_im_start_end=False)
            model=SimpleNamespace()
            def generate(self,ids,**kwargs):seen.append((ids.clone(),kwargs));return torch.tensor([[1,2]]),128
        model=Model()
        class Tokenizer:
            def batch_decode(self,*args,**kwargs):return ['A']
        tokenizer=Tokenizer()
        def load(*args,**kwargs):loads.append((args,kwargs));return tokenizer,model,None,2048
        def tokenizer_image(prompt,*args,**kwargs):prompts.append(prompt);return torch.tensor([1,-200,9])
        class Image: size=(10,11)
        modules={}
        for arm,tree in [('source',code),('question_only',b.transform(code))]:
            compiled=compile(tree,str(c.SOURCE),'exec',dont_inherit=True,optimize=0);ns={'__name__':'stub_'+arm,'__file__':str(c.SOURCE)};exec(compiled,ns)
            ns.update(load_pretrained_model=load,disable_torch_init=lambda:None,get_model_name_from_path=lambda p:'llava-v1.5-7b',tokenizer_image_token=tokenizer_image,load_image_from_base64=lambda s:Image(),process_images=lambda *a:torch.zeros((1,3,2,2)),tqdm=Bar)
            modules[arm]=ns
        with tempfile.TemporaryDirectory() as td,patch.object(torch.Tensor,'cuda',lambda self:self):
            for pos in range(2):
                for arm in ('source','question_only'):
                    modules[arm]['pd']=SimpleNamespace(read_table=lambda path,pos=pos:frame.iloc[[pos]].copy())
                    target=Path(td)/('%s_%s.jsonl'%(arm,pos))
                    args=SimpleNamespace(model_path='dummy',model_base=None,question_file='sanitized.tsv',answers_file=str(target),conv_mode='vicuna_v1',num_chunks=1,chunk_idx=0,all_rounds=False,single_pred_prompt=True,lang='en',temperature=0.,top_p=None,num_beams=1,visual_token_num=128,beta=1.,alpha=.5)
                    modules[arm]['eval_model'](args);r=json.loads(target.read_text());self.assertEqual(r['question_id'],101+pos);self.assertEqual(r['text'],'A')
        self.assertEqual(len(loads),4)
        for i in range(2):
            self.assertEqual(prompts[2*i],prompts[2*i+1])
            self.assertEqual(seen[2*i][1]['texts'], 'a hint\nWhat is it?\nA. red\nB. blue' if i==0 else 'Second?\nA. yes\nB. no')
            self.assertEqual(seen[2*i+1][1]['texts'],frame.iloc[i]['question'])
            for key in ('do_sample','temperature','top_p','num_beams','max_new_tokens','use_cache','image_sizes'):
                self.assertEqual(seen[2*i][1][key],seen[2*i+1][1][key])
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
