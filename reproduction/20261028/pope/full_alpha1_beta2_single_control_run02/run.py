"""Default CPU-only check; one full8910 run needs separate root allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import common as c


def identity(pid):
    try:tick=int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None
    return dict(pid=pid,start_ticks=tick,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def gpu_pids():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return sorted({int(x.strip()) for x in raw.splitlines() if x.strip()})


def allocation(note,plan):
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==plan['run_id']
    assert note['plan_sha256']==c.sha(c.PLAN) and note['GPU_slot'] in ['A','B'] and note['GPU_device']==0
    assert note['single_attempt'] is True and note['automatic_retry'] is False
    assert note['CPU_acceptance_sha256']==c.sha(c.HERE/'cpu_acceptance.json')
    cpu=c.read(c.HERE/'cpu_acceptance.json');assert cpu['success'] and cpu['plan_sha256']==c.sha(c.PLAN) and cpu['GPU_started'] is False
    assert note['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert note['maximum_concurrent_models_including_loading']==2
    others=note['allowed_other_gpu_processes'];assert len(others)<2
    assert gpu_pids()==sorted(x['pid'] for x in others) and all(identity(x['pid'])==x for x in others)


def run(note_path):
    plan=c.read(c.PLAN);c.check(plan);note=c.read(note_path);allocation(note,plan)
    assert c.sha(c.PLAN)==c.read(c.HERE/'READY.json')['plan_sha256']
    assert not c.OUT.exists() and not (c.HERE/'state.json').exists()
    controller=identity(os.getpid())
    with (c.HERE/'launch.claim.json').open('x') as stream:
        json.dump(dict(controller=controller,plan_sha256=c.sha(c.PLAN),
            allocation_note_sha256=c.sha(note_path),automatic_retry=False,launch_attempt=1),stream,indent=2)
        stream.write('\n')
    state=dict(status='starting',controller=controller,started_utc=c.UTC(),plan_sha256=c.sha(c.PLAN),
        allocation_note_sha256=c.sha(note_path),automatic_retry=False)
    def state_save():
        target=c.HERE/'state.json';temp=c.HERE/'.state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(target)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for key in list(env):
        if key.startswith(('USE_LLAVA_ARCH_','ANCHORZIP_','EADP_','ENTROPY_')):env.pop(key)
    try:
        with (c.HERE/'generation.log').open('x') as log:
            process=subprocess.Popen([c.PYTHON,'-B',str(c.HERE/'worker.py'),str(note_path.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='generating',native_worker=identity(process.pid));state_save();assert state['native_worker']
            code=process.wait()
        state.update(status='native_exited',native_exit_code=code);state_save();assert code==0
        assert identity(process.pid)!=state['native_worker'] and process.pid not in gpu_pids()
        with (c.HERE/'verification.log').open('x') as log:
            verify=subprocess.run([c.PYTHON,'-B',str(c.HERE/'verify_terminal.py')],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        state['CPU_verifier_exit_code']=verify.returncode;state_save();assert verify.returncode==0
        proof=c.read(c.OUT/'independent_verification.json')
        assert proof['success'] and proof['n_predictions']==8910 and proof['actual_guidance_checks']==proof['input_ID_checks']==proof['output_redecode_checks']==proof['budget_mask_prefix_checks']==8910 and proof['panel_beta2_anchor_checks']==16
        c.check(plan)
        assert c.sha(note_path)==state['allocation_note_sha256']
        c.save(c.OUT/'release.json',dict(success=True,n_predictions=8910,native_worker=state['native_worker'],controller=controller,
            actual_native_exited=True,native_exit_code=0,plan_sha256=c.sha(c.PLAN),allocation_note_sha256=c.sha(note_path),
            independent_verification_sha256=c.sha(c.OUT/'independent_verification.json'),
            predictions_sha256=c.sha(c.OUT/'predictions.jsonl'),runtime_observations_sha256=c.sha(c.OUT/'runtime_observations.jsonl'),
            official_macro_f1_percent=proof['official_macro_f1_percent'],independent_macro_f1_percent=proof['independent_macro_f1_percent'],
            controller_actual_exit_requires_root_proof=True,
            automatic_retry=False,all_generation_completed_before_scoring=True,score_direction_used_for_acceptance=False))
        state.update(status='complete',finished_utc=c.UTC(),release_sha256=c.sha(c.OUT/'release.json'));state_save()
        print(json.dumps(dict(status='complete',n_predictions=8910,score_direction_used_for_acceptance=False)))
    except BaseException as error:
        state.update(status='terminal_failure_no_retry',error=repr(error));state_save();c.save(c.HERE/'failure.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['check','run'],nargs='?',default='check')
    parser.add_argument('--allocation-note',type=Path);args=parser.parse_args()
    if args.action=='check':
        c.check(c.read(c.PLAN));print(json.dumps(dict(CPU_guards=True,model_loaded=False,GPU_started=False,torch_imported='torch' in sys.modules)))
    else:
        assert args.allocation_note is not None;run(args.allocation_note.resolve())
