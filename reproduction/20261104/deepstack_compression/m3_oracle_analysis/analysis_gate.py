"""Require source-bound CPU tests and separate review before any scoring."""
import json
from pathlib import Path
import score_gate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
APPROVAL=HERE/'analysis_approval.json'

def read(path):return json.loads(Path(path).read_text())
def validate():
    if not APPROVAL.is_file():raise ValueError('oracle scoring locked: independent analysis approval absent')
    a=read(APPROVAL);required={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}
    if set(a)!=required:raise ValueError('incomplete analysis approval')
    cpu=Path(a['cpu_proof']);score_gate.runtime_gate.supervised(a['cpu_started'],a['cpu_terminal'],HERE/'cpu_validation.py',cpu.parent);proof=read(cpu)
    if (proof.get('status')!='passed' or proof.get('GT_opened') is not False or proof.get('predictions_opened') is not False or
        proof.get('scoring_performed') is not False or proof.get('gpu_used') is not False or proof.get('source_sha256_before')!=proof.get('source_sha256_after')):
        raise ValueError('analysis CPU proof invalid')
    sources=proof['source_sha256_after'];sha=score_gate.runtime_gate.sha
    if any(sha(path)!=digest for path,digest in sources.items()) or any(sources.get(str(p.resolve()))!=sha(p) for p in list(HERE.glob('*.py'))+[HERE/'plan.json']):
        raise ValueError('analysis source changed or not covered')
    review_path=Path(a['review_proof']);reviewer=ROOT/'e1_independent_review/review_m3oracle_analysis.py'
    if review_path.resolve()!=(HERE/'independent_review.json').resolve():raise ValueError('unexpected analysis review')
    score_gate.runtime_gate.supervised(a['review_started'],a['review_terminal'],reviewer,review_path);review=read(review_path)
    if (review.get('status')!='passed' or review.get('independent_review') is not True or review.get('GT_opened') is not False or
        review.get('predictions_opened') is not False or review.get('gpu_used') is not False or review.get('reviewer_sha256')!=sha(reviewer) or
        review.get('cpu_proof_sha256')!=sha(cpu) or review.get('cpu_started_sha256')!=sha(a['cpu_started']) or
        review.get('cpu_terminal_sha256')!=sha(a['cpu_terminal']) or review.get('source_sha256')!=sources):
        raise ValueError('analysis independent review invalid')
    return {'approval_sha256':sha(APPROVAL),'cpu_sha256':sha(cpu),'review_sha256':sha(review_path)}
