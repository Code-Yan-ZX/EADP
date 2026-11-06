"""Real model-before-load worker; never authorizes or initializes GPU."""
import argparse,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import admission
if Path(admission.__file__).resolve()!=D/'admission.py':raise ValueError('foreign admission')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve():raise ValueError('unplanned output')
 approval=admission.require_preflight()
 import runtime_support
 if Path(runtime_support.__file__).resolve()!=D/'runtime_support.py':raise ValueError('foreign runtime support')
 result=runtime_support.preflight(admission.read(D/'preflight_plan002.json'));result.update(approval=approval,scope='three-anchor real processor and four-weight CPU preflight only')
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('passed')
if __name__=='__main__':main()
