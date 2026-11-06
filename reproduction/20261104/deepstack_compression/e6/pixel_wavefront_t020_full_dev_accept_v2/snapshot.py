"""Save exact reviewed source bytes only after complete clean run."""
import json
import os
import shutil
from pathlib import Path
import sys
ORIGINAL=Path(__file__).resolve().parent.parent/'pixel_wavefront_t020_full_dev'
sys.path.insert(1,str(ORIGINAL))
import admission
import finish_guard
import acceptance

V=Path(__file__).resolve().parent;D=ORIGINAL
def main():
    acceptance.check();acceptance.approval();plan=admission.check_plan();finish_guard.validate();directory=V/'snapshot002';directory.mkdir(exist_ok=False)
    files={}
    for number,(path,digest) in enumerate(sorted(plan['source_sha256'].items())):
        target=directory/f'{number:04d}.bin'
        with open(path,'rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
        if admission.sha(target)!=digest:raise ValueError('snapshot changed')
        files[path]={'path':str(target),'sha256':digest}
    with (D/'plan002.json').open('rb') as src,(directory/'original_plan.json').open('xb') as dst:
        shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
    if admission.sha(directory/'original_plan.json')!=admission.sha(D/'plan002.json'):raise ValueError('run plan snapshot mismatch')
    with (directory/'manifest.json').open('x') as f:
        json.dump({'status':'passed','plan_sha256':admission.sha(D/'plan002.json'),'original_plan_path':str(directory/'original_plan.json'),'files':files},f,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())

if __name__=='__main__':main()
