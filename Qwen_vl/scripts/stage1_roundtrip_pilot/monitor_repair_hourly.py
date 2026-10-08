"""Read-only hourly repair-run checks; never kill or start generation jobs.

Default/--once prints one dry JSON snapshot. --record additionally writes the
monitor's own snapshot files. Scheduling, notifications, and repairs belong to
the external hourly runner. This program never changes a running experiment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'Qwen_vl/outputs/audit_followup_20261008'
DEFAULT_CONFIG = OUT / 'rerun_batch/hourly_monitor_config.json'


def utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def process_info(pid):
    result = dict(pid=pid, alive=False)
    if not isinstance(pid, int) or pid <= 0:
        return result
    try:
        root = Path(f'/proc/{pid}')
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        ticks = int(fields[19])
        elapsed = float(Path('/proc/uptime').read_text().split()[0]) - ticks / os.sysconf('SC_CLK_TCK')
        return dict(pid=pid, alive=fields[0] != 'Z', state=fields[0], command=command,
                    start_ticks=ticks, elapsed_seconds=max(0, elapsed))
    except (OSError, ValueError, IndexError):
        return result


def child_processes(pid):
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        return [process_info(int(value)) for value in children]
    except (OSError, ValueError):
        return []


def file_info(path, current_time, count_lines=False, log=False):
    result = dict(path=str(path), exists=False)
    if not path:
        return result
    path = Path(path)
    try:
        stat = path.stat()
        result.update(exists=True, size_bytes=stat.st_size, mtime_utc=utc(stat.st_mtime),
                      age_seconds=max(0, current_time - stat.st_mtime))
        if count_lines:
            with path.open('rb') as stream:
                result['complete_lines'] = sum(chunk.count(b'\n') for chunk in iter(lambda: stream.read(1048576), b''))
        if log:
            with path.open('rb') as stream:
                stream.seek(max(0, stat.st_size - 8192))
                tail = stream.read().decode(errors='replace')
            frames = [frame.strip() for frame in re.split(r'[\r\n]', tail) if frame.strip()]
            result['last_log_line'] = frames[-1][-500:] if frames else ''
            counters = re.findall(r'(\d+)/(\d+)', tail)
            result['last_tqdm_count'] = [int(v) for v in counters[-1]] if counters else None
            result['error_tail'] = any(token in tail for token in ('Traceback (most recent call last)', 'CUDA out of memory'))
    except OSError as error:
        result['read_error'] = str(error)
    return result


def assess(status, governor, expected_identity, child, output, log, state_age,
           expected_n, loading_grace=600, stall_seconds=1800, transition_grace=120):
    """Pure diagnostic policy; legal waits have no output-stall requirement."""
    issues = []
    if status == 'failed':
        issues.append('run_failed')
    elif status != 'complete' and not governor.get('alive'):
        issues.append('supervisor_missing')
    if governor.get('alive') and expected_identity not in governor.get('command', ''):
        issues.append('supervisor_identity_mismatch')
    n = output.get('complete_lines', 0)
    if expected_n is not None and n > expected_n:
        issues.append('prediction_count_exceeds_expected')
    if status == 'generating':
        if not child.get('alive') and state_age > transition_grace:
            issues.append('generation_process_missing')
        elif child.get('alive'):
            if n == 0 and child.get('elapsed_seconds', 0) > loading_grace:
                issues.append('loading_exceeds_grace')
            elif n > 0 and output.get('age_seconds', 0) >= stall_seconds:
                issues.append('generation_no_progress_30min')
        if log.get('error_tail'):
            issues.append('generation_error_in_log')
    if status in ('scoring', 'scoring_predecessor') and state_age >= stall_seconds:
        issues.append('scoring_no_progress_30min')
    return issues


def concurrency_wait_issues(state, current_time, loading_grace):
    """A between-model concurrency wait must not hide an indefinite gate stall."""
    if state.get('status') != 'waiting_for_next_concurrency':
        return []
    completed = state.get('completed_stages', [])
    if completed and current_time - dt.datetime.fromisoformat(completed[-1]['completed_utc']).timestamp() >= loading_grace:
        return ['concurrency_wait_exceeds_loading_grace']
    return []


def classify_gpu_processes(entries):
    """Count GPU slots conservatively; a confirmed idle Ollama context uses none."""
    active, ignored = [], []
    for entry in entries:
        pid_text, memory_text = entry.split(',', 1)
        process = process_info(int(pid_text.strip()))
        memory = int(memory_text.strip())
        command = process.get('command', '')
        item = dict(pid=int(pid_text.strip()), memory_mib=memory, command=command)
        words = command.split()
        if (memory == 0 and process.get('alive') and len(words) >= 2
                and Path(words[0]).name == 'ollama' and words[1] == 'runner'):
            ignored.append(item)
        else:
            active.append(item)
    return active, ignored


def snapshot_entity(entity, current_time, config):
    result = dict(name=entity['name'], needs_attention=False, issues=[])
    try:
        state = read_json(entity['state'])
        governor = process_info(state.get(entity['pid_field']))
        result.update(status=state.get('status'), process=governor,
                      completed_count=len(state.get('completed', state.get('completed_stages', []))),
                      updated_utc=state.get('updated_utc'), error=state.get('error'))
        age = max(0, current_time - dt.datetime.fromisoformat(state['updated_utc']).timestamp())
        job = state.get('current_job')
        child = process_info(state.get('child_pid'))
        output_path, log_path, expected_n = None, None, None
        if job:
            output_path, log_path = job['output'], job['generation_log']
            expected_n = 2017 if job['task'] == 'sqa' else 5000
        elif state.get('stage_output'):
            output_path = state['stage_output']
            plans = state.get('stage_plans', [state.get('plan')])
            plan = read_json(plans[state.get('current_stage', 1) - 1])
            log_path, expected_n = plan['log'], plan['question_count']
        elif state.get('status') == 'waiting' and state.get('wait_pid'):
            # NeXT's initial two-arm helper lives outside the lane supervisor.
            plan = read_json(entity['plan'])
            children = child_processes(state['wait_pid'])
            for predecessor in plan.get('predecessor_jobs', []):
                match = next((p for p in children if p.get('alive') and predecessor['output'] in p.get('command', '')), None)
                if match:
                    child = match
                    output_path, log_path, expected_n = predecessor['output'], predecessor['generation_log'], 5000
                    result['underlying_status'] = 'generating'
                    break
        output = file_info(output_path, current_time, count_lines=True)
        log = file_info(log_path, current_time, log=True)
        checked_status = result.get('underlying_status', result['status'])
        result.update(child_process=child, output=output, log=log, expected_n=expected_n,
                      legal_wait=result['status'] in ('waiting', 'waiting_after_predecessor', 'waiting_for_next_concurrency'))
        result['issues'] = assess(checked_status, governor, entity['identity'], child,
                                  output, log, age, expected_n,
                                  config['loading_grace_seconds'], config['stall_seconds'],
                                  config['transition_grace_seconds'])
        result['issues'].extend(concurrency_wait_issues(state, current_time, config['loading_grace_seconds']))
        if 'concurrency_wait_exceeds_loading_grace' in result['issues']:
            result['legal_wait'] = False
        if checked_status == 'generating' and child.get('alive') and output_path and str(output_path) not in child.get('command', ''):
            result['issues'].append('generation_process_identity_mismatch')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result.update(status='unreadable', issues=['state_or_plan_unreadable'], error=str(error))
    result['needs_attention'] = bool(result['issues'])
    return result


def take_snapshot(config, current_time=None):
    current_time = time.time() if current_time is None else current_time
    entities = [snapshot_entity(entity, current_time, config) for entity in config['entities']]
    complete = all(entity.get('status') == 'complete' for entity in entities)
    summary = config['summary']
    try:
        summary_process = process_info(int(Path(summary['pid_file']).read_text()))
        exported = read_json(summary['output'])
        summary_status = dict(name='summary_watcher', process=summary_process,
                              completed_rows=exported.get('completed_rows'), errors=exported.get('errors'),
                              output=file_info(summary['output'], current_time), issues=[])
        if not complete and not summary_process['alive']:
            summary_status['issues'].append('summary_watcher_missing')
        if summary_process['alive'] and summary['identity'] not in summary_process.get('command', ''):
            summary_status['issues'].append('summary_watcher_identity_mismatch')
        if exported.get('errors'):
            summary_status['issues'].append('summary_validation_error')
        summary_status['needs_attention'] = bool(summary_status['issues'])
    except (OSError, ValueError, TypeError) as error:
        summary_status = dict(name='summary_watcher', needs_attention=not complete,
                              issues=[] if complete else ['summary_unreadable'], error=str(error))
    gpu = {}
    try:
        queried = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=15)
        gpu = dict(returncode=queried.returncode, processes=queried.stdout.strip().splitlines())
        active, ignored = classify_gpu_processes(gpu['processes'])
        gpu.update(active_slot_processes=active, active_slot_count=len(active),
                   ignored_zero_memory_ollama_contexts=ignored)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        gpu = dict(query_error=str(error))
    global_issues = ['gpu_process_count_exceeds_two'] if gpu.get('active_slot_count', 0) > 2 else []
    if gpu.get('query_error') or gpu.get('returncode', 0) != 0:
        global_issues.append('gpu_status_query_failed')
    return dict(snapshot_utc=utc(current_time), all_complete=complete,
                needs_attention=bool(global_issues) or summary_status['needs_attention'] or any(e['needs_attention'] for e in entities),
                global_issues=global_issues, entities=entities, summary_watcher=summary_status,
                gpu=gpu, policy=dict(stall_seconds=config['stall_seconds'], loading_grace_seconds=config['loading_grace_seconds'],
                                    legal_marker_waits_exempt=True, concurrency_wait_grace_seconds=config['loading_grace_seconds'],
                                    mutations_to_generation=False))


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    output_dir = Path(config['snapshot_directory'])
    snapshot = take_snapshot(config)
    print(json.dumps(snapshot), flush=True)
    if args.record:
        record(output_dir / 'latest_snapshot.json', snapshot)
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        record(output_dir / f'snapshot_{stamp}.json', snapshot)


if __name__ == '__main__':
    main()
