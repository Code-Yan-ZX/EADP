"""CPU-only recovery hard gates; no run() or CUDA model operation."""
import copy
import importlib.util
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    recovery=module(OUT/'resume_next_full_controls_after_json_gate.py','tested_recovery')
    controller=recovery.load(recovery.CONTROLLER,'tested_frozen_controller')
    original=controller.read(controller.STATE);checks=[]
    recovery.unchanged_failure_state(controller,original);checks.append('real failed registration matches exact recoverable gate')
    def rejects(name,state,alive_pid=None):
        previous=controller.process
        if alive_pid is not None:controller.process=lambda pid:dict(alive=pid==alive_pid)
        try:
            try:recovery.unchanged_failure_state(controller,state)
            except ValueError:checks.append(name)
            else:raise AssertionError('Gate accepted '+name)
        finally:controller.process=previous
    rejects('live prior supervisor refused',original,original['supervisor_pid'])
    rejects('live prior GPU child refused',original,original['jobs']['EADP_beta2']['pid'])
    fixture=copy.deepcopy(original);fixture['registered_methods']=fixture['registered_methods'][:2];rejects('registered scope changes refused',fixture)
    fixture=copy.deepcopy(original);fixture['started_utc']=None;rejects('missing original start refused',fixture)
    fixture=copy.deepcopy(original);fixture['child_pids']=[1];rejects('nonempty prior child reservation refused',fixture)
    fixture=copy.deepcopy(original);fixture['jobs']['EADP_beta2']['returncode']=1;rejects('generation nonzero exit refused',fixture)
    fixture=copy.deepcopy(original);fixture['jobs']['AZ_beta2']['error']='source mismatch';rejects('different failure cause refused',fixture)
    fixture=copy.deepcopy(original);fixture['jobs']['EADP_beta1']['status']='complete';rejects('already executed beta1 refused',fixture)
    def gate_case(name,gpu,active,progress,dependency,expected):
        if controller.gate(gpu,active,progress,dependency)['ready'] is not expected:raise AssertionError(name)
        checks.append(name)
    gate_case('empty GPU and original dependencies ready accepted',dict(entries=[],free_mib=44257),set(),{},True,True)
    gate_case('foreign unknown zero context refused',dict(entries=[dict(pid=999,memory_mib=0,alive=False,command='')],free_mib=44257),set(),{},True,False)
    gate_case('insufficient free memory refused',dict(entries=[],free_mib=22499),set(),{},True,False)
    gate_case('main panel incomplete refused',dict(entries=[],free_mib=44257),set(),{},False,False)
    gate_case('loading reservation retained without actual own CUDA context',dict(entries=[],free_mib=44257),{99},{99:True},True,False)
    gate_case('only real zero-memory Ollama exception accepted',dict(entries=[dict(pid=999,memory_mib=0,alive=True,command='/usr/local/bin/ollama runner model')],free_mib=44257),set(),{},True,True)
    paths=controller.read(controller.PLAN)['jobs']['EADP_beta1']['output_paths']
    if any(Path(paths[k]).exists() for k in ('prediction','runtime','native_protocol','state','score','finished','log')):raise AssertionError('beta1 output already exists')
    checks.append('real original beta1 generation artifacts all absent')
    result=dict(success=True,n=len(checks),checks=checks,CPU_only=True,GPU_started=False,
        original_state_sha256=controller.sha(controller.STATE),original_controller_sha256=controller.sha(recovery.CONTROLLER),
        recovery_source_sha256=controller.sha(recovery.__file__),overlay_sha256=controller.sha(recovery.OVERLAY))
    target=OUT/'resume_next_full_controls_cpu_validation.json'
    if target.exists():raise FileExistsError('Preserve existing CPU validation')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
