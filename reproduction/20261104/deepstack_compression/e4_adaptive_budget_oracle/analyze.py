"""Exact in-sample answer-informed K allocation; not an inference policy."""
import argparse, hashlib, json
from collections import Counter
from decimal import Decimal
from pathlib import Path

B=Path(__file__).resolve().parent.parent
D=B/'e4_adaptive_budget_oracle'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def optimize(scores,capacity):
    # states[cost] = (exact score sum, allocation tuple); previous rows sorted.
    states={0:(Decimal(0),())}
    for row in scores:
        nxt={}
        for used,(value,allocation) in sorted(states.items()):
            for k in (128,256,512):
                cost=used+k//128
                if cost>capacity:continue
                v=value+row[k]
                if cost not in nxt or v>nxt[cost][0]:nxt[cost]=(v,allocation+(k,))
        states=nxt
    if not states:raise ValueError('no feasible allocation')
    best=max(x[0] for x in states.values())
    cost=min(c for c,x in states.items() if x[0]==best)
    return best,cost,states[cost][1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads((D/'plan001.json').read_text());source=B/plan['source_scores']
    if sha(source)!=plan['source_scores_sha256']:raise ValueError('source hash mismatch')
    first=json.loads((B/plan['source_reviews'][0]).read_text());second=json.loads((B/plan['source_reviews'][1]).read_text())
    if (first.get('status')!='passed' or first.get('score_journal_sha256')!=sha(source) or first.get('questions_verified')!=384
        or second.get('status')!='passed' or second.get('score_sha256')!=sha(source) or second.get('questions')!=384):raise ValueError('missing score verification')
    rows=[json.loads(line,parse_float=Decimal) for line in source.read_text().splitlines()]
    if len(rows)!=384 or len({(r['task'],str(r['qid'])) for r in rows})!=384 or len({r['group_id'] for r in rows})!=384:raise ValueError('DEV inventory')
    a.out.mkdir(exist_ok=False);reports={};assignments=[]
    for task in plan['tasks']:
        part=sorted((r for r in rows if r['task']==task),key=lambda r:str(r['qid']).encode())
        if len(part)!=128:raise ValueError('incomplete task')
        scores=[]
        for r in part:
            d={}
            for k in plan['arms']:
                arm=r['arms'][f'K{k}-D1'];s=Decimal(arm['score']);ind=Decimal(arm['independent_score'])
                if not (s.is_finite() and 0<=s<=1 and abs(s-ind)<=Decimal('1e-12')):raise ValueError('unverified score')
                d[k]=s
            scores.append(d)
        total,cost,ks=optimize(scores,256);base=sum(r[256] for r in scores)
        if cost>256 or total<base:raise ValueError('invalid optimum')
        counts=Counter(ks)
        reports[task]={'questions':128,'oracle_mean_pct':float(total*100/128),'fixed_k256_mean_pct':float(base*100/128),'oracle_minus_fixed_pp':float((total-base)*100/128),'selected_counts':{str(k):counts[k] for k in plan['arms']},'total_visual_tokens':cost*128,'mean_visual_tokens':cost,'questions_with_different_k':sum(k!=256 for k in ks)}
        assignments.extend({'task':task,'qid':str(r['qid']),'group_id':r['group_id'],'selected_k':k,'oracle_score':float(score[k]),'fixed_k256_score':float(score[256])} for r,score,k in zip(part,scores,ks))
    report={'status':'completed','plan_sha256':sha(D/'plan001.json'),'source_sha256':sha(source),'tasks':reports,'GT_opened':False,'new_predictions_opened':False,'limitations':plan['limitations']}
    with (a.out/'assignments.jsonl').open('x') as f:
        for row in assignments:f.write(json.dumps(row,sort_keys=True)+'\n')
    with (a.out/'analysis.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('completed')

if __name__=='__main__':main()
