"""Freeze observer proposal and CPU evidence; never starts the observer."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
D = Path(__file__).resolve().parent
PY = '/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    terminal = json.loads((D/'monitor_cpu001/terminal.json').read_text())
    if terminal.get('reason') != 'completed' or terminal.get('worker_returncode') != 0 or terminal.get('remaining_process_group') != []:
        raise ValueError('CPU tests did not complete cleanly')
    files = [D/x for x in ('watch.py', 'validate.py', 'test_validate.py', 'README.md', 'freeze_plan.py',
                           'monitor_cpu001/started.json', 'monitor_cpu001/terminal.json',
                           'monitor_cpu001/stdout.log', 'monitor_cpu001/stderr.log')]
    command = [PY, 'e4_c_latency/timing_supervise.py', '--run-dir', 'e4_c_multibudget_environment/monitor_watch001',
               '--timeout', '7620', '--snapshot-seconds', '3600', '--poll-seconds', '5', '--grace-seconds', '30', '--',
               PY, 'e4_c_multibudget_environment/watch.py', '--output', 'e4_c_multibudget_environment/samples001.jsonl',
               '--stop', 'e4_c_multibudget_environment/stop001', '--interval', '60', '--duration', '7500']
    data = {'stage': 'host background observation plan; no observer or GPU started',
            'source_sha256': {str(p): sha(p) for p in files}, 'command': command, 'cwd': str(BASE),
            'output': str(D/'samples001.jsonl'), 'stop_marker': str(D/'stop001'),
            'watch_supervisor': str(D/'monitor_watch001'), 'interval_seconds_minimum': 60,
            'ps_query_timeout_seconds': 10, 'duration_seconds': 7500, 'outer_timeout_seconds': 7620,
            'start': 'start before GPU worker; await first complete snapshot pair',
            'stop': 'create marker after GPU worker and supervisor are fully stopped; verify observer completed/0 and process group empty',
            'coverage': 'same time namespace; first query begins by model run start and watch_end no earlier than model run end; otherwise coverage unknown',
            'known_windows': ['ps query', 'ps query through first log fsync', 'watcher supervisor health', 'model supervisor health'],
            'unknown_windows': ['second log fsync', 'stop marker write', 'residual and unrecorded system interference'],
            'interpretation': 'ps CPU is process lifetime average, not instantaneous; no samples excluded; report all 1520 job overlaps separately',
            'GT_opened': False, 'scoring_performed': False, 'GPU_authorized': False}
    with (D/'plan001.json').open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
