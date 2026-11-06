"""Candidate fixed worker callbacks; no CLI or standalone GPU authority."""
import importlib
import importlib.util
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
LATENCY=BASE/'e4_c_latency';REAL=BASE/'e4_c_memory_real';RUNTIME=BASE/'e4_c_memory_runtime';MEMORY=BASE/'e4_c_memory'
PINNED={name:LATENCY/(name+'.py') for name in ('latency_worker','latency_gate','real_worker','schedule',
        'input_preflight','identity_gate','contract','runner','timer','stage_runner','light_adapter','memory_probe')}
PINNED.update({name:RUNTIME/(name+'.py') for name in ('worker_candidate','worker_barrier','bridge')})
PINNED['skip_deepstack']=BASE/'e4/skip_deepstack.py'
PINNED['plan']=MEMORY/'plan.py'

def guard_modules():
    for name,path in PINNED.items():
        module=sys.modules.get(name)
        if module is not None and Path(getattr(module,'__file__','/')).resolve()!=path.resolve():
            raise ValueError('preloaded module source mismatch: '+name)

def fixed_import(name):
    if name not in PINNED:raise ValueError('module is not pinned')
    guard_modules();sys.path.insert(0,str(PINNED[name].parent))
    module=importlib.import_module(name);guard_modules()
    if Path(getattr(module,'__file__','/')).resolve()!=PINNED[name].resolve():raise ValueError('module source mismatch: '+name)
    return module

def load_named(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ValueError('cannot load fixed source')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=path.resolve():raise ValueError('source module path mismatch')
    return module

def prepare(approval):
    guard_modules()
    latency_gate=fixed_import('latency_gate');latency_worker=fixed_import('latency_worker')
    real=fixed_import('real_worker');schedule=fixed_import('schedule')
    real_worker=load_named('eadp_smoke_real_worker',REAL/'worker.py')
    return real_worker.prepare_cpu(stage='smoke',mode='C-ON',approved_sources=approval['source_sha256'],
              require_identity=latency_gate.require_light_identity,prepare=latency_worker.prepare,
              load_runtime=latency_worker.load_runtime,build=schedule.build,sha=real.sha)

def continue_run(context,directory,nonce,approval):
    guard_modules()
    candidate=fixed_import('worker_candidate');real=fixed_import('real_worker')
    runner=fixed_import('runner');probe=fixed_import('memory_probe')
    fixed_import('skip_deepstack');fixed_import('light_adapter')
    plan_path=MEMORY/'plan.py';existing=sys.modules.get('plan')
    if existing is not None and Path(getattr(existing,'__file__','/')).resolve()!=plan_path.resolve():
        raise ValueError('unexpected preloaded memory plan module')
    if existing is None:sys.modules['plan']=load_named('eadp_smoke_memory_plan',plan_path)
    memory_worker=load_named('eadp_smoke_memory_jobs',MEMORY/'worker.py')
    guard_modules()
    return candidate.run_after_preflight(stage='smoke',mode='C-ON',nonce=nonce,directory=directory,
        prepared=context['prepared'],expected_schedule=context['expected_schedule'],identity=context['identity'],
        torch=context['torch'],native=context['native'],wrapper_class=context['wrapper_class'],
        preflight_one=real.preflight_one,execute_one=runner.execute_one,memory_measure=probe.memory_measure,
        run_memory_jobs=memory_worker.run_memory_jobs,durable_journal=runner.durable_journal,sha=real.sha)

if __name__=='__main__':raise SystemExit('locked: fixed callbacks alone do not authorize GPU')
