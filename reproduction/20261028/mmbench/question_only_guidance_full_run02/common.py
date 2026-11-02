"""CPU-only utilities and identity guards for the fixed MMB panel."""
import hashlib, json, os
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLAN=HERE/'plan.json'
OUT=HERE/'execution'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
SOURCE=HERE.parents[3]/'_reproduction/author_eadp/LLaVA/llava/eval/model_vqa_mmbench.py'
MODEL=Path('/media/disk2/YZX/doct/FastV/llava-v1.5-7b')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save_new(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def identity(pid):
    try:
        p=Path('/proc')/str(pid)
        return dict(pid=pid,start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None

def check(plan):
    assert plan['run_id']=='MMB_full_source_vs_question_only_run02'
    assert plan['prepared_only'] is True and plan['n_rows']==4377 and plan['n_groups']==1176
    for item in plan['files']:
        assert sha(item['path'])==item['sha256'],item['path']
    for path,expected in plan['model_stat'].items():
        p=Path(path);st=p.stat()
        actual=dict(bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
        # The same host file may resolve through a different bind-mount path
        # in the device-free verifier. Compare actual inode, not path spelling.
        assert actual=={k:v for k,v in expected.items() if k not in ('prior_verified_sha256','resolved')},path
        assert os.path.samefile(p,expected['resolved']),path
    assert sha(HERE/'inputs/panel.tsv')==plan['panel_sha256']
    assert sha(HERE/'inputs/manifest.json')==plan['manifest_sha256']
    assert 'torch' not in __import__('sys').modules
