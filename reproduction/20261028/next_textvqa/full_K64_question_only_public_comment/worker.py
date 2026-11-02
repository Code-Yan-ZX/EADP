"""Frozen K64 native framework, isolated paths, one guidance AST intervention."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import guidance_patch

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
NEXT=OUT.parent
PLAN=OUT/'plan.json'
RUN=OUT/'execution'
BASE=NEXT/'full_default_K64_public/worker.py'
BASE_SHA='bc453330fccf089bfbdceb0b4b02796fab084233bf78bd04f9366a9d22ab5f61'
RUN_ID='NeXT_full_K64_question_only_public_comment'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def adapted_module():
    assert sha(BASE)==BASE_SHA
    spec=importlib.util.spec_from_file_location('frozen_completed_K64_worker_adapter',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT,module.PLAN,module.RUN=OUT,PLAN,RUN
    # The process actually starts this new worker; the reused adapter remains
    # separately frozen in the proof rather than masquerading as its source.
    module.__file__=str(Path(__file__).resolve())
    full,original_proof=module.adapted_module()
    full.run.__globals__['runpy']=types.SimpleNamespace(run_path=guidance_patch.run_path)
    result=dict(original_proof,base_K64_adapter_path=str(BASE),base_K64_adapter_sha256=BASE_SHA,
        guidance_entry_transform=guidance_patch.proof(),guidance_input_changed=True,
        pruning_math_changed=False,full_LLM_OCR_prompt_changed=False,
        interpretation='Inactive public-source comment input hypothesis; not known paper argv')
    return full,result


if __name__=='__main__':
    assert len(sys.argv)==2
    note=json.loads(Path(sys.argv[1]).read_text());claim=json.loads((OUT/'launch.claim.json').read_text())
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==RUN_ID
    assert note['plan_sha256']==sha(PLAN)==claim['plan_sha256'] and sha(sys.argv[1])==claim['allocation_note_sha256']
    assert os.getppid()==claim['controller']['pid']
    from run import identity,gpu_pids,validate_allocation
    assert identity(os.getppid())==claim['controller']
    validate_allocation(note,sha(PLAN))
    full,_=adapted_module();full.run(True)
