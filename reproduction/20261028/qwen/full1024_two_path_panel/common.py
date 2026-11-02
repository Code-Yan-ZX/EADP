"""Fixed16 two-path diagnostic metadata, CPU guard, and comparison helpers."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'gpu_formula_facility_panel_run02'
PLAN = HERE/'plan.json'
RUN_ID = 'Qwen_Doc_fixed16_plain_vs_EADP1024_paths'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

def save(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def plan_checks(p, namespace='host'):
    assert p['run_id']==RUN_ID and p['status']=='prepared_not_started'
    for path,digest in p['files'].items(): assert sha(path)==digest,path
    sys.path.insert(0,str(PREV))
    import diagnose
    original=read(PREV/'run_plan.json')
    diagnose.validate(original,namespace)
    assert p['panel']==original['panel'] and p['base_recipe']==original['recipe']
    assert len(p['panel'])==16 and p['visual_token_num']==1024
    assert p['registered_model'] is False and p['benchmark_result'] is False

def compare_metadata(a,b):
    """Exact per-array identity; missing keys are not called equal."""
    return {k:(a[k]==b[k] if k in a and k in b else None) for k in sorted(set(a)|set(b))}
