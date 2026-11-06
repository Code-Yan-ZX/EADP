"""Fixed residual cohort; never uses scores or feature values."""
from collections import Counter
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def build(manifest):
 rows=manifest['remaining'];pilot=manifest['pilot'];key=lambda r:(r['task'],str(r['qid']))
 if len(rows)!=352 or len(pilot)!=9 or len({key(r) for r in rows})!=352 or len({key(r) for r in pilot})!=9:raise ValueError('upstream denominator')
 bykey={key(r):r for r in rows}
 if any(bykey.get(key(r))!=r for r in pilot):raise ValueError('pilot identity not in fixed cohort')
 ids={key(r) for r in pilot};result=[r for r in rows if key(r) not in ids]
 if len(result)!=343 or Counter(r['task'] for r in result)!={'DocVQA_VAL':114,'TextVQA_VAL':114,'AI2D_TEST':115}:raise ValueError('remaining denominator')
 return result
