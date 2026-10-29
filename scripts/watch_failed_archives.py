"""Archive eligible failed attempts while a collection scheduler is running.

No simulator is started. Only finalized, explicitly non-training attempts with
existing preview media are passed to the independently verified archive tool.
After the scheduler exits, the watcher refreshes previews and performs a final
sweep. Use the same collection-media lock as the review generator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from archive_failed_attempt import archive_attempt
from collection_media_lock import media_lock
import render_collection_review as review


ELIGIBLE = frozenset({'failure', 'collision_excluded', 'collision_unknown',
                      'out_of_view', 'visibility_unknown', 'data_error'})
SAFE_TASK = re.compile(r'^[A-Za-z0-9_-]+$')
SAFE_ATTEMPT = re.compile(r'^attempt_\d+$')
RETRY_ERROR_SECONDS = 180
_STOP_SIGNAL = None


def scheduler_identity(pid: int):
    """Return the Linux process start time, or None if it has exited/zombied."""
    if pid <= 1:
        raise ValueError('Scheduler PID must be greater than 1')
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        raise RuntimeError(f'Cannot inspect scheduler PID {pid}; refusing to assume it exited') from exc
    # The parenthesized command can contain spaces; fields after it start at #3.
    closing = text.rfind(')')
    if closing < 0:
        raise ValueError(f'Invalid /proc/{pid}/stat')
    fields = text[closing + 1:].strip().split()
    if len(fields) < 20:
        raise ValueError(f'Incomplete /proc/{pid}/stat')
    if fields[0] == 'Z':
        return None
    return fields[19]  # Linux stat field 22: creation time in clock ticks.


def write_event(log_path, event, **fields):
    entry = {'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf8') as log:
        log.write(json.dumps(entry, ensure_ascii=False) + '\n')
        log.flush()
    print('DIAGNOSTIC_WATCH', json.dumps(entry, ensure_ascii=False), flush=True)


def eligible_report(path: Path, root: Path):
    """Cheap prefilter; archive_attempt performs all destructive-operation checks."""
    task, attempt = path.parent.parent.name, path.parent.name
    if not SAFE_TASK.fullmatch(task) or not SAFE_ATTEMPT.fullmatch(attempt):
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if (data.get('task_id') != task or data.get('completed') is not True
            or data.get('returncode', 0) != 0 or data.get('error') or data.get('timed_out')
            or data.get('collection_classification') not in ELIGIBLE):
        return False
    trials = data.get('trials')
    if not isinstance(trials, list) or len(trials) != 1 or not isinstance(trials[0], dict):
        return False
    if trials[0].get('trial') != 1 or trials[0].get('diagnostic_hdf5_path'):
        return False
    media = root / 'review/media' / task / attempt / 'trial_001'
    return all((media / name).is_file() for name in ('trajectory.gif', 'trajectory.mp4', 'signature.json'))


def reviewer_process_active(root: Path, proc_root=Path('/proc')):
    """Also defer to an older review process that started before locking existed."""
    expected = (root / 'review').resolve()
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = [item.decode('utf8', errors='replace') for item in
                    (process / 'cmdline').read_bytes().split(b'\0') if item]
        except (OSError, PermissionError):
            continue
        if not any(Path(arg).name == 'render_collection_review.py' for arg in args):
            continue
        try:
            value = Path(args[args.index('--output-dir') + 1])
        except (ValueError, IndexError):
            continue
        if not value.is_absolute():
            try:
                value = (process / 'cwd').resolve() / value
            except OSError:
                value = ROOT / value
        if value.resolve() == expected:
            return True
    return False


def scan_once(root: Path, log_path: Path, failures: dict, *, blocking=False, now=None):
    """Perform one bounded sweep; skipped/failed attempts keep their sources."""
    now = time.monotonic() if now is None else now
    done = 0
    while reviewer_process_active(root):
        if not blocking:
            return {'archived': 0, 'locked': True}
        time.sleep(1)
    with media_lock(root / 'review', blocking=blocking) as acquired:
        if not acquired:
            return {'archived': 0, 'locked': True}
        for path in sorted(root.glob('*/attempt_*/report.json')):
            if not eligible_report(path, root):
                continue
            try:
                stat = path.stat()
            except OSError as exc:
                write_event(log_path, 'report_unavailable', report=str(path), error=str(exc))
                continue
            fingerprint = (stat.st_size, stat.st_mtime_ns)
            previous = failures.get(str(path))
            if previous and previous['fingerprint'] == fingerprint and now - previous['time'] < RETRY_ERROR_SECONDS:
                continue
            try:
                outcome = archive_attempt(path, root, cleanup=True)
            except Exception as exc:
                failures[str(path)] = {'fingerprint': fingerprint, 'time': now}
                write_event(log_path, 'archive_failed_requires_review', report=str(path),
                            error=f'{type(exc).__name__}: {exc}')
                if _STOP_SIGNAL is not None:
                    raise StopRequested(f'signal {_STOP_SIGNAL}')
                continue
            failures.pop(str(path), None)
            done += 1
            write_event(log_path, 'archived', report=str(path), **outcome)
    return {'archived': done, 'locked': False}


def refresh_review(root: Path, log_path: Path):
    try:
        # A process launched before the shared lock was introduced would not
        # honor it, so wait for that process before starting our own refresh.
        while reviewer_process_active(root):
            time.sleep(1)
        review.make_report(root / 'manifest.json', root, root / 'review')
    except Exception as exc:
        write_event(log_path, 'review_refresh_failed', error=f'{type(exc).__name__}: {exc}')
        return False
    return True


def run_watcher(root: Path, pid: int, interval: float, log_path: Path):
    if interval <= 0:
        raise ValueError('Polling interval must be positive')
    root = root.resolve()
    if not (root / 'manifest.json').is_file():
        raise FileNotFoundError(f'Collection manifest not found: {root / "manifest.json"}')
    identity = scheduler_identity(pid)
    failures = {}
    write_event(log_path, 'started', scheduler_pid=pid, scheduler_start_time=identity,
                collection=str(root), interval_seconds=interval)
    while identity is not None and scheduler_identity(pid) == identity:
        result = scan_once(root, log_path, failures)
        if result['archived']:
            refresh_review(root, log_path)
        time.sleep(interval)
    # Ensure any finalized attempt has its preview before the last sweep.
    write_event(log_path, 'scheduler_exited_final_sweep', scheduler_pid=pid)
    refresh_review(root, log_path)
    result = scan_once(root, log_path, failures, blocking=True)
    if result['archived']:
        refresh_review(root, log_path)
    write_event(log_path, 'finished', archived_in_final_sweep=result['archived'],
                unresolved_errors=len(failures))


class StopRequested(BaseException):
    pass


def _request_stop(signum, frame):
    global _STOP_SIGNAL
    del frame
    _STOP_SIGNAL = signum
    raise StopRequested(f'signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', required=True, type=Path)
    parser.add_argument('--scheduler-pid', required=True, type=int)
    parser.add_argument('--interval', type=float, default=15.0)
    parser.add_argument('--log', type=Path, help='Defaults to COLLECTION/diagnostic_watcher.jsonl')
    args = parser.parse_args()
    root = args.collection_root.expanduser().resolve()
    log = (args.log or root / 'diagnostic_watcher.jsonl').expanduser().resolve()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        run_watcher(root, args.scheduler_pid, args.interval, log)
    except StopRequested as exc:
        write_event(log, 'stopped', reason=str(exc))
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
