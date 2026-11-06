"""Freeze the existing M3 diagnostic contracts before any reference is read."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    cpu=HERE/'cpu_validation001.json';terminal=HERE/'monitor_cpu_validation001/terminal.json'
    sample=HERE/'sample_001.json';review=HERE/'sample_001_independent_review.json'
    target_terminal=HERE/'monitor_target001/terminal.json'
    c=json.loads(cpu.read_text()); t=json.loads(terminal.read_text()); s=json.loads(sample.read_text()); v=json.loads(review.read_text())
    if c.get('status')!='passed' or c['source_sha256_before']!=c['source_sha256_after'] or any(sha(p)!=h for p,h in c['source_sha256_after'].items()):
        raise ValueError('CPU source contract changed')
    for p in (terminal,target_terminal,HERE/'monitor_review_sample001/terminal.json'):
        state=json.loads(p.read_text())
        if state.get('reason')!='completed' or state.get('worker_returncode')!=0 or state.get('remaining_process_group')!=[]:
            raise ValueError('a contract/review did not complete')
    if (s.get('GT_opened') is not False or s.get('predictions_opened') is not False or v.get('status')!='passed'
            or v.get('sample_sha256')!=sha(sample) or v.get('GT_opened') is not False or v.get('predictions_opened') is not False):
        raise ValueError('sample provenance')
    paths=[cpu,terminal,sample,review,target_terminal,HERE/'target_contract.py',HERE/'test_target_contract.py',Path(__file__)]
    protocol={'stage':'prospective_cpu_protocol_only','created_unix':time.time(),'selection_after_e4_dev_scoring':True,
              'source_sha256':{str(p):sha(p) for p in paths},'cpu_source_sha256':c['source_sha256_after'],
              'questions_by_task':s['counts'],'GT_opened':False,'predictions_opened':False,
              'reference':'DocVQA/TextVQA first reference verbatim; AI2D correct option string; no alternate by loss',
              'target_tokenization':'tokenizer.encode(answer, add_special_tokens=False), no added space/EOS; first 128 token IDs; report original length and truncation; empty target fails',
              'teacher':'frozen full Qwen3-VL weights and P3, batch 1, 1024 full visual tokens for one gradient; no cache; mean token cross entropy',
              'utility':'four streams mean(abs(float32(feature)*float32(gradient))) across hidden dimensions',
              'rank':'0-1 percentile across all 1024 positions; lower spatial ID ranks higher on an exact utility tie',
              'base_selection':'actual EADP greedy K256 order; always remove its final r IDs before replacement',
              'proposals':'r=8/16; MAIN/MULTI/RANDOM each pool64, proposal0 top-r, 7 deterministic unique random subsets; max128 retries and equal source count',
              'random_seed':'first 8 bytes big-endian SHA256 of UTF-8 pipe-separated 20261104,qid,source,r,j; random-pool seed uses 20261104,qid,RANDOM,POOL; Python random.Random',
              'selection_by_teacher':'minimum reference mean NLL; tie do-nothing then proposal index; same per-source search budget, shared dedup cache',
              'natural_generations':'one per winning source/r plus fixed EADP baseline; complete panel before scoring',
              'bootstrap':'10000 image-group resamples; MULTI minus MAIN and MULTI minus RANDOM; 95% intervals, 1 percentage point minimum relevant gain',
              'limits':['Answer-dependent oracle is not deployable or an upper bound.', 'No sample question is used for student confirmation.',
                        'Real-model smoke, resources, exact source freeze, gate and independent review are still required.']}
    with (HERE/'protocol_001.json').open('x') as f:json.dump(protocol,f,indent=2);f.write('\n')
    print(json.dumps({'status':'protocol_frozen','sha256':sha(HERE/'protocol_001.json')}))
if __name__=='__main__':sys.exit(main())
