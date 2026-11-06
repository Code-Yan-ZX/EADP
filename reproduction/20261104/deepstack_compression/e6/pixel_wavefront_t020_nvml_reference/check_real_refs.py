"""Read only previously accepted prediction identities; never load model/GT."""
import hashlib,json,os
from pathlib import Path
import reference
D=Path(__file__).resolve().parent
refs,proof=reference.load_refs()
inventory=[]
for (task,qid,arm),r in sorted(refs.items()):
 ids=r['selected_indices'];assert len(ids)==r['selected_count'] and all(type(x) is int for x in ids)
 raw=json.dumps(ids,separators=(',',':')).encode();inventory.append({'task':task,'qid':qid,'arm':arm,'selected_count':len(ids),'selected_indices_sha256':hashlib.sha256(raw).hexdigest(),'generated_count':len(r['tokens']),'record_sha256':r['record_sha256']})
value={'status':'passed','references':len(inventory),'inventory':inventory,'source_sha256':hashlib.sha256((D/'reference.py').read_bytes()).hexdigest(),'prior':proof,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted same-arm reference identity only','GT_opened':False,'GPU_used':False,'model_loaded':False,'limitations':['t0 complete ordered indices checked in prior record; high-dimensional intermediate tensors not fully archived']}
with (D/'real_refs001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
