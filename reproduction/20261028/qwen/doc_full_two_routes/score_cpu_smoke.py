"""Two synthetic rows through actual author evaluator + independent ANLS."""
import importlib.util,json,os,sys,tempfile
from pathlib import Path
import common as c
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
old=c.read(c.OLD/'run_plan.json');sys.path.insert(0,old['author_root']);sys.path.insert(0,str(Path(old['author_root'])/'VLMEvalKit'))
import pandas as pd
from vlmeval.dataset.image_vqa import ImageVQADataset
spec=importlib.util.spec_from_file_location('audit_qwen',c.Q/'independent_audit/audit_qwen.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
import torch
assert not torch.cuda.is_initialized()
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'synthetic.xlsx'
 pd.DataFrame([dict(index=1,question='synthetic',answer="['foo']",prediction='foo',split='val'),dict(index=2,question='synthetic',answer="['abc']",prediction='abd',split='val')]).to_excel(path,index=False)
 ds=ImageVQADataset.__new__(ImageVQADataset);ds.dataset_name='DocVQA_VAL';aggregate=ds.evaluate(str(path))
 actual=float(aggregate['Overall'].iloc[0]);expected=100*(audit.independent_anls('foo',['foo'])+audit.independent_anls('abd',['abc']))/2
 assert abs(actual-expected)<1e-10
 assert len(pd.read_excel(Path(tmp)/'synthetic_results.xlsx'))==2
assert not torch.cuda.is_initialized()
print(json.dumps(dict(success=True,synthetic_rows=2,official=actual,independent=expected,real_GT_read=False,model_loaded=False,GPU_requested=False,CUDA_initialized=False)))
