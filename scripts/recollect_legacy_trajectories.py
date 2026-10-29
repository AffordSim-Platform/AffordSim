"""After the current pass, re-record trajectories made before action alignment was fixed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from collect_50_trajectories import classify, update_report, review, write_json

ROOT = Path(__file__).resolve().parents[1]


def process_is_running(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        return len(stat) > 2 and stat[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--wait-for-pid', type=int, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--max-retries', type=int, default=5)
    args = p.parse_args()
    output = args.output_dir.resolve()
    manifest = output / 'manifest.json'
    tasks = {task['task_id']: task for task in json.loads(manifest.read_text())['tasks']}
    print(f'Waiting for the primary collection scheduler pid={args.wait_for_pid}', flush=True)
    while process_is_running(args.wait_for_pid):
        time.sleep(10)
    progress = (output / 'progress.log').read_text()
    if not any(line.startswith('FINISHED accepted=') for line in progress.splitlines()):
        print('Primary scheduler did not finish normally; leaving recollection pending for inspection.', flush=True)
        return 1
    migration = json.loads((output / 'alignment_migration.json').read_text())
    status_path = output / 'collection_status.json'
    state = json.loads(status_path.read_text())
    accepted = state['accepted']
    for item in migration['items']:
        accepted.pop(item['task_id'], None)
    state['completed'] = len(accepted)
    write_json(status_path, state)
    write_json(output / 'accepted_manifest.json', {'tasks': list(accepted.values())})
    review(manifest, output)
    remaining = []
    for item in migration['items']:
        name = item['task_id']
        task = tasks[name]
        original_attempt = output / name / item['attempt']
        old_report = json.loads((original_attempt / 'report.json').read_text())
        original_seed = int(old_report['seed'])
        passed = False
        for retry in range(args.max_retries):
            existing = [int(q.name.split('_')[-1]) for q in (output / name).glob('attempt_*')
                        if q.is_dir() and q.name.split('_')[-1].isdigit()]
            attempt = output / name / f'attempt_{max(existing, default=0) + 1:04d}'
            attempt.mkdir(parents=True)
            config = attempt / 'task.yaml'
            shutil.copy2(original_attempt / 'task.yaml', config)
            seed = (original_seed + retry * 1299709) % (2**31)
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_benchmark_50.py'), '--worker',
                       '--config', str(config), '--task-id', name, '--robot', task['robot'],
                       '--env-task-name', task['env_task_name'], '--robot-cfg-name', task['robot_cfg_name'],
                       '--robot-type', task['robot_type'], '--trials', '1', '--num-envs', '1',
                       '--seed', str(seed), '--device', 'cuda:0', '--save-trajectories',
                       '--preserve-render-interval', '--output-dir', str(attempt)]
            print(f'RECOLLECT {name} {attempt.name} seed={seed}', flush=True)
            with (attempt / 'run.log').open('w') as log:
                try:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.timeout, check=False,
                                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    returncode = -1
            report_path = attempt / 'report.json'
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                report = {'completed': False, 'error': 'recollection worker produced no report'}
            report['returncode'] = returncode
            result, detail = classify(report, allow_legacy_npy=True)
            update_report(report_path, report, result, detail)
            state['history'].setdefault(name, []).append({'attempt': attempt.name, 'status': result,
                                                           'report': str(report_path), 'recollection': True})
            if result == 'accepted':
                trial = report['trials'][0]
                accepted[name] = {'task_id': name, 'robot': task['robot'], 'attempt': attempt.name,
                                  'trajectory': trial['trajectory_path'], 'directory': trial['directory'],
                                  'report': str(report_path), 'data_integrity': detail}
                passed = True
            state['completed'] = len(accepted)
            write_json(status_path, state)
            write_json(output / 'accepted_manifest.json', {'tasks': list(accepted.values())})
            print(f'RECOLLECT_RESULT {name} {result} accepted={len(accepted)}/50', flush=True)
            review(manifest, output)
            if passed:
                break
        if not passed:
            remaining.append(name)
    final = {'complete': len(accepted) == 50, 'accepted': len(accepted),
             'remaining': sorted(set(tasks) - set(accepted)), 'recollection_failures': remaining}
    write_json(output / 'final_status.json', final)
    print('RECOLLECTION_FINISHED', json.dumps(final), flush=True)
    return 0 if final['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
