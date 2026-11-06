"""Freeze input-only identities without reading any output or labels."""
import hashlib,json,os
from pathlib import Path
from design import panel
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/pixelprune_t020_dev384/plan001.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=json.loads(P.read_text());anchors=panel(prior['anchors']);out=D/'design001.json'
value={'status':'passed','scope':'fixed 384 DEV input identity inventory; no score or prediction value','anchors':anchors,'counts':{'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128},'resolution':512,'visual_tokens_expected':256,'parent_plan_sha256':sha(P),'design_source_sha256':sha(D/'design.py'),'freeze_source_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'GPU_used':False,'errors':[]}
with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
