"""Run one hourly Codex inspection with an exclusive lock and local alerts.

The systemd user timer starts this runner. The saved prompt limits recovery to
the authorized experiment queue; native ChatGPT scheduling is not involved.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008/rerun_batch/hourly_monitor'
TIMER = 'codex-eadp-repair-monitor-20261008.timer'
FOLLOWUP_REQUEST = OUT.parents[1] / 'post_batch_followup_20261009/request.json'


def retain_followup_monitoring(result, request_path=FOLLOWUP_REQUEST):
    """A finished original batch must not stop a pending user follow-up."""
    if not result['all_experiments_complete'] or not request_path.exists():
        return result
    request = json.loads(request_path.read_text())
    if request.get('status') == 'complete':
        return result
    result = dict(result)
    result.update(status='needs_attention', all_experiments_complete=False,
                  summary_zh='原批次已完成，但用户授权的三行核查和必要续跑尚未完成；巡检继续。')
    result['findings'] = [*result['findings'],
                          'Post-batch follow-up remains ' + str(request.get('status'))]
    return result


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def notification(title, body):
    try:
        result = subprocess.run(['/usr/bin/notify-send', '--app-name=EADP实验巡检',
                                 '--urgency=critical', title, body],
                                capture_output=True, text=True, timeout=10)
        return dict(attempted=True, exit_code=result.returncode)
    except Exception as exc:
        return dict(attempted=True, error=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prompt = OUT / 'inspection_prompt.txt'
    schema = OUT / 'inspection_schema.json'
    json.loads(schema.read_text())
    if not prompt.read_text().strip():
        raise ValueError('Inspection prompt is empty')
    if args.validate_only:
        print(json.dumps(dict(prompt=str(prompt), schema=str(schema), valid=True)))
        return
    lock = (OUT / 'inspection.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another hourly inspector is already active; no overlapping inspector.', flush=True)
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = OUT / stamp
    run.mkdir(exist_ok=False)
    last_message = run / 'result.json'
    events = run / 'agent.events.jsonl'
    command = ['/home/dell/.local/bin/codex', 'exec',
               '--cd', str(ROOT), '--sandbox', 'danger-full-access',
               '-c', 'approval_policy="never"', '--color', 'never', '--json',
               '--output-schema', str(schema), '--output-last-message', str(last_message), '-']
    # A standalone inspection uses the installed user's existing model/auth
    # configuration; it must not masquerade as a continuation of this live chat.
    env = dict(os.environ)
    env.pop('CODEX_THREAD_ID', None)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = dict(runner_pid=os.getpid(), status='inspecting', started_utc=start,
                 run_dir=str(run), command=command)
    save(OUT / 'runner_state.json', state)
    try:
        with prompt.open() as stdin, events.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=stdin,
                                       stdout=stream, stderr=subprocess.STDOUT)
            state['agent_pid'] = process.pid
            save(OUT / 'runner_state.json', state)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                # Only stop this inspector. Authorized detached GPU workers are
                # independent processes and must never be killed as a group.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError('Hourly Codex inspection exceeded30 minutes')
        if code != 0:
            raise RuntimeError(f'Hourly Codex inspection exited{code}; see{events}')
        result = json.loads(last_message.read_text())
        required = {'status', 'summary_zh', 'findings', 'actions', 'all_experiments_complete'}
        if not required.issubset(result) or result['status'] not in ('healthy', 'recovered', 'needs_attention', 'complete'):
            raise ValueError('Inspection response did not match the required schema')
        if (result['status'] == 'complete') != (result['all_experiments_complete'] is True):
            raise ValueError('Completion status and experiment-completion flag disagree')
        result = retain_followup_monitoring(result)
        save(last_message, result)
        if result['all_experiments_complete']:
            checked = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('monitor_repair_hourly.py')), '--once'],
                cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
            snapshot = json.loads(checked.stdout)
            if snapshot.get('all_complete') is not True or snapshot.get('needs_attention'):
                raise ValueError('Independent experiment snapshot rejects completion; timer stays enabled')
        state.update(status='checked', agent_returncode=code, result=result,
                     completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if result['status'] in ('recovered', 'needs_attention', 'complete'):
            state['desktop_notification'] = notification('EADP实验巡检', result['summary_zh'])
        save(OUT / 'latest_inspection.json', state)
        if result['all_experiments_complete']:
            stopped = subprocess.run(['/usr/bin/systemctl', '--user', 'disable', '--now', TIMER],
                                     capture_output=True, text=True, timeout=10)
            state['timer_stop_exit_code'] = stopped.returncode
            if stopped.returncode:
                state['timer_stop_error'] = stopped.stderr.strip()
        save(OUT / 'runner_state.json', state)
        print(json.dumps(dict(status=result['status'], summary_zh=result['summary_zh'],
                              result=str(last_message)), ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update(status='failed', error=str(exc),
                     completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        state['desktop_notification'] = notification('EADP巡检运行失败', str(exc))
        save(OUT / 'runner_state.json', state)
        raise


if __name__ == '__main__':
    main()
