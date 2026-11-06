"""Save exact reviewed source bytes only after complete clean run."""
import json
import os
import shutil
from pathlib import Path
import admission
import finish_guard

D=Path(__file__).resolve().parent
def main():
    plan=admission.check_plan();finish_guard.validate();directory=D/'snapshot001';directory.mkdir(exist_ok=False)
    files={}
    for number,(path,digest) in enumerate(sorted(plan['source_sha256'].items())):
        target=directory/f'{number:04d}.bin'
        with open(path,'rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
        if admission.sha(target)!=digest:raise ValueError('snapshot changed')
        files[path]={'path':str(target),'sha256':digest}
    with (directory/'manifest.json').open('x') as f:
        json.dump({'status':'passed','plan_sha256':admission.sha(D/'plan001.json'),'files':files},f,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())

if __name__=='__main__':main()
