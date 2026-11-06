"""Static host allocation and bounds contract; no GPU execution entry."""
import math,re
MODES=('C-ON',)

def validate(plan,allocation,now,budget):
    if (plan.get('stage')!='smoke' or plan.get('mode') not in MODES or plan.get('jobs')!=24
        or type(plan.get('worker_timeout_seconds')) is not int or not 60<=plan['worker_timeout_seconds']<=1800
        or type(plan.get('sampler_timeout_seconds')) is not int or not plan['worker_timeout_seconds']<plan['sampler_timeout_seconds']<=2100
        or type(plan.get('outer_timeout_seconds')) is not int or not plan['sampler_timeout_seconds']+120<=plan['outer_timeout_seconds']<=43200
        or type(plan.get('grace_seconds')) is not int or plan['grace_seconds']<120
        or type(plan.get('sample_interval_ns')) is not int or not 100_000_000<=plan['sample_interval_ns']<=1_000_000_000
        or type(plan.get('max_sample_gap_ns')) is not int or not plan['sample_interval_ns']<plan['max_sample_gap_ns']<=5_000_000_000):
        raise ValueError('invalid worker/sampler/budget bounds')
    if (not isinstance(allocation,dict) or allocation.get('authorization')!='root_scheduled_C_ON_memory_smoke'
        or allocation.get('stage')!='smoke' or allocation.get('mode')!='C-ON'
        or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',str(allocation.get('gpu_uuid','')))
        or not isinstance(budget,dict) or type(budget.get('budget_start_unix')) not in (float,int)
        or type(budget.get('budget_end_unix')) not in (float,int)
        or not math.isfinite(budget['budget_start_unix']) or not math.isfinite(budget['budget_end_unix'])
        or allocation.get('budget_end_unix')!=budget['budget_end_unix']
        or type(now) not in (float,int) or not math.isfinite(now)
        or now<budget['budget_start_unix'] or now+plan['outer_timeout_seconds']+plan['grace_seconds']>budget['budget_end_unix']
        or not re.fullmatch(r'[0-9a-f]{64}',str(allocation.get('prelaunch_review_sha256','')))
        or not re.fullmatch(r'[0-9a-f]{32}',str(allocation.get('nonce','')))):
        raise ValueError('invalid reviewed allocation or insufficient budget')
    return {'status':'validated_fields_only','stage':'smoke','mode':'C-ON',
            'scope':'no GPU authority; caller must independently verify proof and allocation'}

if __name__=='__main__':raise SystemExit('locked: no GPU launch authority')
