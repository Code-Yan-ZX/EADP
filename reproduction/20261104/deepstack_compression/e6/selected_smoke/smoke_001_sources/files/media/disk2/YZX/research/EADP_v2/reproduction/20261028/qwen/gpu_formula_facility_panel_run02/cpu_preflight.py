"""Inside CPU container: run every frozen guard, with no tensor/model import."""
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose import PLAN, validate, stat, sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not any(Path('/dev', name).exists() for name in ['nvidia0', 'nvidiactl', 'nvidia-uvm'])
    plan = json.loads(PLAN.read_text()); validate(plan, namespace='container')
    assert 'torch' not in sys.modules and not any(n.startswith('flash_attn') for n in sys.modules)
    print(json.dumps(dict(status='all_frozen_guards_passed_in_CPU_container', run_plan_sha256=sha(PLAN),
        source_guards=sum(len(plan[key]) for key in ['source_sha256', 'implementation_sha256', 'reference_sha256']),
        model_stats={path:stat(path) for path in plan['model_stat_guards']}, image_guards=len(plan['panel']),
        torch_imported=False, model_loaded=False, GPU_devices_present=False, CUDA_VISIBLE_DEVICES='',
        model_namespace='Explicit destination-root mapping; exact device/inode/bytes/mtime/ctime preserved')))


if __name__ == '__main__': main()
