"""Bounded M3 selection and exposure protocol. Caller supplies verified inputs.

No model, labels, or data are opened here. Every uncached evaluation and every
natural-generation attempt is journaled durably. A partial question is never
resumed automatically: a failed call can have exposed an unrecorded output.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROPOSALS=HERE.parent/'m3_oracle/proposals.py'
spec=importlib.util.spec_from_file_location('m3_oracle_exec_proposals',PROPOSALS)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class Journal:
    def __init__(self,path):
        self.path=Path(path)
        # Exclusive creation makes a second attempt explicit and auditable.
        self.f=self.path.open('x',encoding='utf-8')
    def append(self,event,**fields):
        self.f.write(json.dumps({'event':event,**fields},separators=(',',':'),allow_nan=False)+'\n')
        self.f.flush();os.fsync(self.f.fileno())
    def close(self):self.f.close()

def execute(qid,greedy_order,utilities,evaluate,generate,journal,baseline_binding,checkpoint):
    """Reference one accepted E1 baseline and generate 6 source/r outputs; no scoring.

    evaluate(sorted 256 IDs) returns finite mean reference NLL. generate(sorted
    IDs, on_generated) must call on_generated immediately after model.generate
    returns with raw token_ids, then return token_ids/eos/truncated. Both are caller-owned
    real-model operations, and must separately validate model/input identity.
    """
    if (not isinstance(baseline_binding,dict) or baseline_binding.get('qid')!=str(qid) or
        baseline_binding.get('arm')!='P3-D1' or baseline_binding.get('verified') is not True or
        any(not isinstance(baseline_binding.get(k),str) or len(baseline_binding[k])!=64 for k in ('record_sha256','acceptance_sha256','inputs_sha256'))):
        raise ValueError('verified same-question E1 baseline binding required')
    proposal=p.proposals(str(qid),greedy_order,utilities)
    journal.append('question_started',qid=str(qid),baseline=baseline_binding,python_version=sys.version,
                   proposals_source_sha256=hashlib.sha256(PROPOSALS.read_bytes()).hexdigest(),proposal=proposal)
    cache={};evaluations=0
    def loss(ids):
        nonlocal evaluations
        key=tuple(ids)
        if key not in cache:
            checkpoint()
            journal.append('evaluation_started',spatial_ids=list(key))
            value=float(evaluate(list(key)))
            if not math.isfinite(value):raise ValueError('nonfinite teacher loss')
            cache[key]=value;evaluations+=1
            journal.append('evaluation_completed',spatial_ids=list(key),mean_nll=value)
        return cache[key]
    baseline=sorted(greedy_order);baseline_loss=loss(baseline);winners=[]
    for r in (8,16):
        block=proposal['by_r'][str(r)];count=block['n_proposals_per_source']
        if not 0<count<=8 or any(len(block['sources'][s])!=count for s in p.SOURCES):
            raise ValueError('unfair proposal counts')
        for source in p.SOURCES:
            best=(baseline_loss,-1,baseline)
            for i,added in enumerate(block['sources'][source]):
                ids=p.replacement(greedy_order,added,r);candidate=(loss(ids),i,ids)
                # Baseline wins exact ties, then smaller proposal index.
                if candidate[:2]<best[:2]:best=candidate
            winners.append({'r':r,'source':source,'proposal_index':best[1],
                            'spatial_ids':best[2],'mean_nll':best[0],'n_proposals':count})
    journal.append('search_completed',baseline_nll=baseline_loss,unique_evaluations=evaluations,winners=winners)
    arms=winners
    outputs=[]
    for arm in arms:
        identity={'source':arm['source'],'r':arm['r'],'spatial_ids':arm['spatial_ids']}
        checkpoint()
        journal.append('generation_started',**identity)
        raw=[]
        def on_generated(output):
            if raw:raise ValueError('multiple generation returns')
            journal.append('generation_returned',**identity,output=output);raw.append(output)
        value=generate(arm['spatial_ids'],on_generated)
        if not raw or not isinstance(raw[0],dict) or not isinstance(value,dict) or raw[0].get('token_ids')!=value.get('token_ids'):
            raise ValueError('actual generation output not journaled')
        if (not isinstance(value,dict) or not isinstance(value.get('token_ids'),list) or
            any(type(i)!=int or i<0 for i in value['token_ids']) or
            type(value.get('eos')) is not bool or type(value.get('truncated')) is not bool or
            not (value['eos'] or value['truncated'])):raise ValueError('invalid natural generation endpoint')
        journal.append('generation_completed',**identity,output=value)
        outputs.append({**identity,'output':value})
    journal.append('question_completed',qid=str(qid),generation_count=len(outputs))
    return {'proposal':proposal,'baseline_nll':baseline_loss,'winners':winners,
            'unique_evaluations':evaluations,'outputs':outputs}
