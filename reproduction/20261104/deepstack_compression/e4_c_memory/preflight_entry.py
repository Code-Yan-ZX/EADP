"""Worker side of reviewed, retained, no-GPU memory CPU preflight."""
import argparse,json,os
from pathlib import Path
from admission import admit
from worker_preflight import prepare_memory

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 approval=admit();launch=json.loads((out/'launch.json').read_text())
 if launch.get('approval_bundle')!=approval:raise ValueError('memory preflight launch approval mismatch')
 result=prepare_memory('smoke')
 if result.get('cuda_initialized') is not False or result.get('model_loaded') is not False:raise ValueError('memory preflight touched GPU/model')
 result.update(approval_bundle=approval,scope='retained container CPU preflight; no GPU, model, predictions, GT or score')
 with (out/'worker_preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
