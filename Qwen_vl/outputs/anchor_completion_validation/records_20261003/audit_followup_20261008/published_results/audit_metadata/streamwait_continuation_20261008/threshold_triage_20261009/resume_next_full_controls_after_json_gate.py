"""Resume only the registered beta1 control after a proved JSON-key gate error.

Default/--prepare validates completed evidence and writes a separate recovery
plan, with no GPU launch or original-artifact changes. Explicit --run requires
the byte-exact failure archive and exclusive original supervisor lock. It
reuses both complete beta2 predictions and their unchanged official scores.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
AUDIT=OUT.parent
DIAG=AUDIT/'next_gap_diagnosis_20261008'
CONTROLLER=DIAG/'schedule_next_full_controls.py'
OVERLAY=OUT/'json_count_distribution_overlay.py'
RECOVERY_PLAN=OUT/'resume_next_full_controls_after_json_gate.plan.json'
ARCHIVE_MANIFEST=OUT/'json_gate_failure_archive.json'
DEFAULT_ARCHIVE=OUT/'json_gate_failure_originals'
PYTHON=Path('/home/dell/miniconda3/envs/llava_pruner/bin/python')
ARMS=['EADP_beta2','AZ_beta2','EADP_beta1']
EXPECTED_FAILURE='Independent full official rescore differs'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def fixed_runtime():
    if sys.version_info[:2]!=(3,10) or Path(sys.executable).resolve()!=PYTHON.resolve():
        raise ValueError('Recovery requires the original llava_pruner Python 3.10; no numerical tolerance is used')

def unchanged_failure_state(controller,state):
    if (state.get('status')!='failed' or state.get('child_pids')!=[] or not state.get('started_utc')
        or state.get('selected_arms')!=ARMS or state.get('registered_methods')!=ARMS
        or state.get('publication_expected_groups')!=controller.expected_groups(ARMS)):
        raise ValueError('Original failed registration or publication scope differs')
    if controller.process(state.get('supervisor_pid'))['alive']:
        raise ValueError('Original full-control supervisor is still alive')
    for arm in ARMS:
        job=state.get('jobs',{}).get(arm,{})
        if controller.process(job.get('pid'))['alive']:
            raise ValueError('An original control child is still alive: '+arm)
    for arm in ARMS[:2]:
        job=state['jobs'][arm]
        if job.get('returncode')!=0 or job.get('status')!='failed' or job.get('error')!=EXPECTED_FAILURE:
            raise ValueError('Only the proven JSON-key false rejection can be resumed')
    if state['jobs']['EADP_beta1'].get('status')!='skipped_dependency':
        raise ValueError('Beta1 was not an unstarted dependency skip')

def preflight(controller,overlay):
    fixed_runtime();plan=controller.read(controller.PLAN);state=controller.read(controller.STATE)
    controller.verify(plan)
    if state.get('plan_sha256')!=controller.sha(controller.PLAN):raise ValueError('Original plan SHA differs')
    unchanged_failure_state(controller,state)
    marker_path=controller.STABILITY/'next_text_full_streamwait.controller.finished.json'
    marker=controller.read(marker_path)
    if marker.get('success') is not False or marker.get('complete') is not False or marker.get('selected_arms')!=ARMS:
        raise ValueError('Original failed controller marker differs')
    original_pid=state['supervisor_pid'];completed={};sources={}
    with overlay.installed_json_count_overlay():
        for arm in ARMS[:2]:
            completed[arm]=controller.validate_arm(plan['jobs'][arm],plan)
            paths=plan['jobs'][arm]['output_paths']
            worker=controller.module(controller.WORKER,'recovery_original_fullworker_'+arm)
            worker.verify_images(controller.read(paths['control_protocol']))
            for key in ('prediction','runtime','native_protocol','control_protocol','state','score','finished'):
                sources[paths[key]]=controller.sha(paths[key])
    beta1=plan['jobs']['EADP_beta1'];controller.protected(beta1)
    if (beta1['command']!=[str(PYTHON),str(controller.WORKER),'--run','--arm','EADP_beta1','--attempt','1']
        or beta1['parameters']['beta']!=1. or beta1['parameters']['anchorzip'] is not False):
        raise ValueError('Original single predefined beta1 command differs')
    dependency=controller.prerequisites()
    if not dependency['ready']:raise ValueError('Main queues/panel completion changed')
    result=dict(CPU_only=True,GPU_started=False,reason='Only actual_count_distribution JSON object keys normalized; all official numerical values compared exactly in original Python 3.10',
        original_controller_plan=str(controller.PLAN),original_controller_plan_sha256=controller.sha(controller.PLAN),
        original_controller_source_sha256=controller.sha(CONTROLLER),original_state_sha256=controller.sha(controller.STATE),
        original_marker_sha256=controller.sha(marker_path),original_supervisor_pid=original_pid,
        original_started_utc=state['started_utc'],registered_methods=ARMS,
        publication_expected_groups=state['publication_expected_groups'],completed_reused=completed,
        reused_artifact_sha256=sources,unchanged_generate_command=beta1['command'],
        launcher_sha256=controller.sha(__file__),overlay_sha256=controller.sha(OVERLAY),
        recovery_scope='Resume one already registered beta1 arm; no added experiment or changed predictions/scores',
        archive_manifest=str(ARCHIVE_MANIFEST),archive_manifest_sha256=controller.sha(ARCHIVE_MANIFEST))
    return result,plan,state

def verify_recovery_plan(controller,recovery):
    for path,key in ((Path(__file__),'launcher_sha256'),(OVERLAY,'overlay_sha256'),(CONTROLLER,'original_controller_source_sha256'),
                     (controller.PLAN,'original_controller_plan_sha256'),(ARCHIVE_MANIFEST,'archive_manifest_sha256')):
        if controller.sha(path)!=recovery[key]:raise ValueError('Recovery-frozen source/artifact changed: '+str(path))
    for path,digest in recovery['reused_artifact_sha256'].items():
        if controller.sha(path)!=digest:raise ValueError('Complete reused artifact changed: '+path)
    controller.verify(controller.read(controller.PLAN))

def verify_archive(controller,recovery,archive_dir):
    archive_dir=Path(archive_dir).resolve()
    if archive_dir!=DEFAULT_ARCHIVE.resolve():raise ValueError('Recovery requires the fixed original failure archive directory')
    manifest=controller.read(ARCHIVE_MANIFEST)
    preserved={item['original_path']:item for item in manifest['preserved_files']}
    for item in preserved.values():
        archive=Path(item['archive_path'])
        if archive.parent.resolve()!=archive_dir or controller.sha(archive)!=item['sha256'] or archive.stat().st_size!=item['bytes']:
            raise ValueError('Byte-exact failure archive is absent/changed')
    marker=controller.STABILITY/'next_text_full_streamwait.controller.finished.json'
    for path,key in ((controller.STATE,'original_state_sha256'),(marker,'original_marker_sha256'),(controller.PLAN,'original_controller_plan_sha256')):
        item=preserved.get(str(path))
        if not item or item['sha256']!=recovery[key] or controller.sha(path)!=recovery[key]:
            raise ValueError('Canonical original and archived failure SHA differ: '+str(path))

def prepare():
    controller=load(CONTROLLER,'recovery_frozen_controller');overlay=load(OVERLAY,'recovery_json_keys_overlay')
    report,_,_=preflight(controller,overlay)
    if RECOVERY_PLAN.exists():
        if controller.read(RECOVERY_PLAN)!=report:raise FileExistsError('Preserve differing existing recovery plan')
    else:controller.atomic(RECOVERY_PLAN,report)
    print(json.dumps(dict(report,plan=str(RECOVERY_PLAN),plan_sha256=controller.sha(RECOVERY_PLAN),
        run_command=[str(PYTHON),str(Path(__file__).resolve()),'--run','--plan',str(RECOVERY_PLAN),'--archive-dir',str(DEFAULT_ARCHIVE)]),indent=2))
    return report

def run(plan_path,archive_dir):
    fixed_runtime()
    if Path(plan_path).resolve()!=RECOVERY_PLAN:raise ValueError('Use the fixed separately prepared recovery plan')
    controller=load(CONTROLLER,'recovery_frozen_controller');overlay=load(OVERLAY,'recovery_json_keys_overlay')
    recovery=controller.read(RECOVERY_PLAN);verify_recovery_plan(controller,recovery)
    lock=(controller.STABILITY/'next_text_full_streamwait.controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fresh,plan,state=preflight(controller,overlay)
    if fresh!=recovery:raise ValueError('Fresh full evidence differs from prepared recovery plan')
    verify_archive(controller,recovery,archive_dir);controller.UTIL.environment()
    completed=copy.deepcopy(recovery['completed_reused']);child=None;log=None
    provenance=dict(plan=str(RECOVERY_PLAN),plan_sha256=controller.sha(RECOVERY_PLAN),source=str(Path(__file__).resolve()),
        source_sha256=controller.sha(__file__),overlay=str(OVERLAY),overlay_sha256=controller.sha(OVERLAY),
        original_supervisor_pid=recovery['original_supervisor_pid'],original_state_sha256=recovery['original_state_sha256'],
        original_marker_sha256=recovery['original_marker_sha256'],archive_manifest=str(ARCHIVE_MANIFEST),
        archive_manifest_sha256=controller.sha(ARCHIVE_MANIFEST),started_utc=controller.now(),pid=os.getpid(),
        process_identity=controller.process(os.getpid()),changed_values=False,reran_complete_beta2=False)
    marker=controller.STABILITY/'next_text_full_streamwait.controller.finished.json'
    marker.unlink()  # The immutable byte-exact failure copy was verified above.
    state.update(status='waiting_for_GPU_slot',supervisor_pid=os.getpid(),child_pids=[],recovery=provenance,
        original_supervisor_pid=recovery['original_supervisor_pid'],error=None,updated_utc=controller.now())
    state.pop('finished_utc',None)
    for arm in ARMS[:2]:
        state['jobs'][arm].update(status='complete',validation=completed[arm],returncode=0,reused_without_regeneration=True)
        state['jobs'][arm].pop('error',None)
    state['jobs']['EADP_beta1'].update(status='pending')
    state['jobs']['EADP_beta1'].pop('error',None)
    controller.atomic(controller.STATE,state)
    def interrupted(signum,frame):raise KeyboardInterrupt('Recovery supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while True:
            verify_recovery_plan(controller,recovery);dependency=controller.prerequisites()
            active={child.pid} if child and child.poll() is None else set()
            progress={pid:controller.has_progress(plan['jobs']['EADP_beta1']) for pid in active}
            gate=controller.gate(controller.UTIL.gpu_snapshot(),active,progress,dependency['ready'])
            state.update(child_pids=sorted(active),gpu_gate=gate,prerequisites=dependency,updated_utc=controller.now())
            if child is not None:
                rc=child.poll()
                if rc is not None:
                    log.close();log=None
                    if rc!=0:raise RuntimeError('Original beta1 worker exited '+str(rc))
                    with overlay.installed_json_count_overlay():
                        for arm in ARMS:completed[arm]=controller.validate_arm(plan['jobs'][arm],plan)
                    state['jobs']['EADP_beta1'].update(status='complete',returncode=0,validation=completed['EADP_beta1'],finished_utc=controller.now())
                    state.update(status='complete',error=None,child_pids=[],finished_utc=controller.now(),updated_utc=controller.now())
                    controller.atomic(controller.STATE,state)
                    if marker.exists():raise FileExistsError('Preserve unexpected canonical controller marker')
                    controller.atomic(marker,dict(success=True,complete=True,selected_arms=ARMS,completed=completed,failures={},
                        finished_utc=controller.now(),recovery=provenance,original_started_utc=recovery['original_started_utc']))
                    return
                if not dependency['ready']:raise RuntimeError('Main/panel completion changed while original beta1 is running')
                if len(gate['contexts'])>2:raise RuntimeError('External third CUDA context; stopping only owned beta1')
                state.update(status='running')
                state['jobs']['EADP_beta1'].update(status='generating' if progress[child.pid] else 'loading_reserved')
            elif gate['ready']:
                job=plan['jobs']['EADP_beta1'];controller.protected(job);controller.UTIL.environment()
                verify_recovery_plan(controller,recovery)
                last_gate=controller.gate(controller.UTIL.gpu_snapshot(),set(),{},controller.prerequisites()['ready'])
                if not last_gate['ready']:time.sleep(1);continue
                log=Path(job['output_paths']['log']).open('x')
                child=subprocess.Popen(job['command'],cwd=controller.ROOT/'LLaVA',env=dict(os.environ,**plan['environment']),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['jobs']['EADP_beta1'].update(status='loading_reserved',pid=child.pid,process_identity=controller.process(child.pid),
                    started_utc=controller.now(),generation_launch_gate=last_gate)
                state.update(status='running',child_pids=[child.pid])
            controller.atomic(controller.STATE,state);time.sleep(1)
    except BaseException as error:
        if child is not None:controller.UTIL.stop_owned(child)
        if log is not None:log.close()
        state['jobs']['EADP_beta1'].update(status='interrupted_preserved' if child is not None else 'failed',error=str(error),finished_utc=controller.now())
        state.update(status='failed',error=str(error),child_pids=[],updated_utc=controller.now())
        controller.atomic(controller.STATE,state)
        if not marker.exists():controller.atomic(marker,dict(success=False,complete=False,selected_arms=ARMS,completed=completed,
            failures={'EADP_beta1':str(error)},recovery=provenance,finished_utc=controller.now()))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path,default=RECOVERY_PLAN);parser.add_argument('--archive-dir',type=Path,default=DEFAULT_ARCHIVE)
    args=parser.parse_args()
    if args.run:run(args.plan,args.archive_dir)
    else:prepare()
if __name__=='__main__':main()
