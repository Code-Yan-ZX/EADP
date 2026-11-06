"""CPU-only nine-job orchestration contract. No Docker, NVML, CUDA or CLI.

Actual jobs must be run via a separately admitted and reviewed implementation
of the established e4_c_memory_real.guarded_lifecycle, not this module.
"""
import re

ARMS = {'t020_author', 't020_wavefront', 't0_author'}

def validate_design(jobs):
    if not isinstance(jobs, list) or len(jobs) != 9:
        raise ValueError('expected nine jobs')
    if [j.get('order') for j in jobs] != list(range(9)):
        raise ValueError('invalid frozen order')
    keys = [(j.get('task'), str(j.get('qid')), j.get('group_id')) for j in jobs]
    anchors = set(keys)
    if len(anchors) != 3 or any(not all(k) for k in anchors):
        raise ValueError('expected three identified anchors')
    if {(k, j.get('arm')) for k, j in zip(keys, jobs)} != {(k, a) for k in anchors for a in ARMS}:
        raise ValueError('expected complete three by three grid')
    for j in jobs:
        if any(not re.fullmatch(r'[0-9a-f]{64}', j.get(k, '')) for k in ('group_id', 'image_sha256', 'message_sha256')):
            raise ValueError('invalid anchor hashes')
    for key in anchors:
        ident = {(j['image_sha256'], j['message_sha256']) for j, k in zip(jobs, keys) if k == key}
        if len(ident) != 1: raise ValueError('anchor identity changed across arms')
    return True

def validate_budget(now, budget_end, per_job_seconds, cleanup_seconds, outer_seconds, jobs_left):
    values=(now,budget_end,per_job_seconds,cleanup_seconds,outer_seconds,jobs_left)
    if any(type(x) not in (int,float) for x in values) or type(jobs_left) is not int or jobs_left not in range(1,10):
        raise ValueError('invalid budget')
    import math
    if not all(math.isfinite(x) for x in values) or per_job_seconds<=0 or cleanup_seconds<=0:
        raise ValueError('invalid bounds')
    needed=jobs_left*per_job_seconds+cleanup_seconds
    if outer_seconds < needed or budget_end-now < outer_seconds:
        raise ValueError('insufficient worst case time')
    return needed

def validate_result(job, result, seen_containers, seen_processes, seen_nonces):
    """Candidate evidence contract, not a substitute for raw independent verification."""
    if not isinstance(result,dict) or result.get('job') != job or result.get('status') != 'passed':
        raise ValueError('job failed or identity mismatch')
    cid=result.get('container_id'); nonce=result.get('nonce'); proc=result.get('process')
    if not isinstance(cid,str) or not re.fullmatch(r'[0-9a-f]{64}',cid) or cid in seen_containers:
        raise ValueError('reused or invalid container')
    if not isinstance(nonce,str) or not re.fullmatch(r'[0-9a-f]{32}',nonce) or nonce in seen_nonces:
        raise ValueError('reused or invalid nonce')
    if not isinstance(proc,dict): raise ValueError('missing process identity')
    pid=proc.get('host_pid'); tick=proc.get('start_ticks'); boot=proc.get('boot_id')
    if type(pid) is not int or pid<=0 or type(tick) is not int or tick<=0 or not isinstance(boot,str) or not boot:
        raise ValueError('invalid process identity')
    identity=(boot,pid,tick)
    if identity in seen_processes or type(proc.get('container_pid')) is not int or proc['container_pid'] != 1:
        raise ValueError('reused process or worker not PID1')
    fields=('running_inspect_verified','host_argv_cgroup_verified','worker_ready_pre_cuda',
            'first_sample_before_release','release_nonce_uuid_pid_bound', 'stopped_inspect_verified',
            'sampler_completed','identity_verified','weights_unchanged','source_proof_verified',
            'no_foreign_compute_observed','stage_windows_valid','sampling_coverage_valid')
    if any(result.get(k) is not True for k in fields): raise ValueError('incomplete lifecycle evidence')
    if type(result.get('exit_code')) is not int or result['exit_code'] != 0 or type(result.get('final_container_pid')) is not int or result['final_container_pid'] != 0 or result.get('sampler_remaining_process_group') != []:
        raise ValueError('unclean worker or sampler exit')
    seen_containers.add(cid);seen_processes.add(identity);seen_nonces.add(nonce)

def run_nine(jobs, run_one, record):
    """Injected CPU candidate: serialize; record all returned evidence; abort on error.

    run_one must complete its own bounded cleanup and persist failure evidence
    even on exceptions. No built-in resource or signal handling is supplied here.
    record must durably append each attempt before the next starts.
    """
    validate_design(jobs)
    containers=set(); processes=set(); nonces=set(); completed=0
    for job in jobs:
        try:
            result=run_one(dict(job))
            record({'job':job,'result':result})
            validate_result(job,result,containers,processes,nonces)
            completed+=1
        except BaseException as exc:
            record({'job':job,'controller_error':{'type':type(exc).__name__,'message':str(exc)}})
            raise
    return {'status':'passed','completed':completed,'unique_containers':len(containers),'unique_processes':len(processes)}

if __name__=='__main__': raise SystemExit('CPU contract only: no GPU authority or launcher')
