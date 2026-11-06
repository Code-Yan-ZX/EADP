"""Freeze all residual identities; reads input messages but never uses or saves them."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 f=json.loads((D/'plan_freeze001.json').read_text());src=f['source_sha256']
 if f.get('status')!='frozen' or any(sha(k)!=v for k,v in src.items()):raise ValueError('source/plan changed')
 p=D/'selection.py';name='_multistream352_selection'
 if name in sys.modules or sha(p)!=src[str(p)]:raise ValueError('unapproved module')
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m)
 rows=[json.loads(x) for x in (B/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines() if x.strip()];pilot=json.loads((B/'e5/selector_real32/sample_manifest_001.json').read_text())['pilot'];result=m.select(rows,pilot)
 if any(sha(k)!=v for k,v in src.items()):raise ValueError('source drift')
 value={'status':'frozen','scope':'posthoc metadata-only selection, no sample feature or score extraction; pilot is included in residual cohort','plan_freeze_sha256':sha(D/'plan_freeze001.json'),'selection_source_sha256':sha(p),'runner_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'feature_values_opened':False,'score_values_opened':False,'GPU_used':False,'input_messages_parsed_for_metadata_only':True,'errors':[],**result}
 with (D/'selection001.json').open('x') as out:json.dump(value,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
if __name__=='__main__':main()
