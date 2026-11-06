"""E1 baseline provenance checks; no dataset reference answers are opened.

This is an upstream evidence gate, not authorization to run M3. The future
outer run admission, locked resource plan and independent review are required.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PROJECT=Path('/media/disk2/YZX/research/EADP_v2')
E1=ROOT/'e1';REVIEW=ROOT/'e1_independent_review';WEIGHTS=PROJECT/'reproduction/20261028/qwen/qwen_weight_identity.json'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())

def supervisor(started,terminal,script,out):
    s=read(started);t=read(terminal);cwd=Path(s.get('cwd','/'));cmd=s.get('command',[])
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
        not isinstance(s.get('worker_pid'),int) or s['worker_pid']<=0 or t.get('worker_pid')!=s['worker_pid'] or
        [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or
        cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):
        raise ValueError('E1 supervisor identity/exit mismatch')

class Receipt:
    def __init__(self,proof_sha,journal_sha,inputs_sha,review_sha,weights_sha,plan_sha):
        self.proof_sha=proof_sha;self.journal_sha=journal_sha;self.inputs_sha=inputs_sha
        self.review_sha=review_sha;self.weights_sha=weights_sha;self.plan_sha=plan_sha

def validate(*,metadata_only=False):
    proofpath=E1/'e1full001_completion.json';reviewpath=REVIEW/'e1full001_review.json'
    cs=E1/'monitorverifye1full001/started.json';ct=E1/'monitorverifye1full001/terminal.json'
    rs=REVIEW/'monitor_e1full001/started.json';rt=REVIEW/'monitor_e1full001/terminal.json'
    supervisor(cs,ct,ROOT/'e1_analysis/completion.py',proofpath)
    supervisor(rs,rt,REVIEW/'review_full.py',reviewpath)
    p=read(proofpath);r=read(reviewpath)
    if (p.get('status')!='passed' or p.get('stage')!='full' or p.get('natural_questions')!=384 or
        p.get('natural_records')!=1536 or p.get('tokenizer_decode_checked')!=1536 or p.get('GT_opened') is not False or
        p.get('scoring_performed') is not False or p.get('verifier_sha256')!=sha(ROOT/'e1_analysis/completion.py') or
        r.get('status')!='passed' or r.get('n_verified')!=384 or r.get('errors')!=[] or
        r.get('completion_proof_sha256')!=sha(proofpath) or r.get('completion_started_sha256')!=sha(cs) or
        r.get('completion_terminal_sha256')!=sha(ct) or r.get('reviewer_sha256')!=sha(REVIEW/'review_full.py')):
        raise ValueError('E1 formal/independent proof mismatch')
    files=p['files'];required={'journal','inputs','frozen','plan','launch','terminal','container_exit','runner','position_adapter'}
    if not required.issubset(files):raise ValueError('E1 source inventory incomplete')
    for name,item in files.items():
        # Preflight must not open prediction-bearing journals; real admission
        # hashes all files before reading any prediction or reference.
        if metadata_only and name in ('journal','generated_arms','quartets'):continue
        if sha(item['path'])!=item['sha256']:raise ValueError('E1 accepted artifact changed: '+item['path'])
    frozen=read(files['frozen']['path'])
    if (frozen.get('weights_identity_sha256')!=sha(WEIGHTS) or frozen.get('plan_sha256')!=files['plan']['sha256'] or
        frozen.get('inputs_sha256')!=files['inputs']['sha256'] or frozen.get('gt_read') is not False):
        raise ValueError('E1 frozen weights/recipe/input mismatch')
    weights=read(WEIGHTS);entries=weights.get('files',[])
    if len(entries)!=4 or {Path(x['path']).name for x in entries}!={f'model-{i:05d}-of-00004.safetensors' for i in range(1,5)}:
        raise ValueError('weight inventory mismatch')
    for x in entries:
        pth=Path(x['path'])
        if pth.stat().st_size!=x['bytes'] or sha(pth)!=x['expected_lfs_sha256'] or x['sha256']!=x['expected_lfs_sha256']:
            raise ValueError('actual model payload mismatch: '+str(pth))
    return Receipt(sha(proofpath),files['journal']['sha256'],files['inputs']['sha256'],sha(reviewpath),sha(WEIGHTS),files['plan']['sha256'])
