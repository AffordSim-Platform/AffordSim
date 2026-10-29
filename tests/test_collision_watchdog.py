"""CPU-only watchdog safety checks; no real evaluation process is started."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
watchdog_path = ROOT / 'outputs/full_retest_v2/diagnostics/watchdog_collision_v1.py'
if not watchdog_path.is_file():
    raise unittest.SkipTest('Local experiment watchdog is unavailable')
spec = importlib.util.spec_from_file_location('collision_watchdog_test', watchdog_path)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class CollisionWatchdogTest(unittest.TestCase):
    def test_source_and_final_result_changes_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'; source.mkdir()
            code = source / 'code.py'; code.write_text('original')
            standalone = root / 'workflow.py'; standalone.write_text('workflow')
            with patch.object(watch, 'SOURCE_DIRS', (source,)), patch.object(watch, 'SOURCE_FILES', (standalone,)):
                snapshot = watch.source_snapshot()
                watch.verify_sources(snapshot, full_hash=True)
                code.write_text('modified')
                with self.assertRaisesRegex(RuntimeError, 'Source or asset'):
                    watch.verify_sources(snapshot)
            original = {'a/report.json': {'report_sha256': 'a', 'config_sha256': 'b'}}
            self.assertEqual(watch.merge_completed(original, original, False), original)
            for changed in ({}, {'a/report.json': {'report_sha256': 'changed', 'config_sha256': 'b'}}):
                with self.assertRaisesRegex(RuntimeError, 'Finalized report'):
                    watch.merge_completed(original, changed, False)
            with self.assertRaisesRegex(RuntimeError, 'Untracked finalized'):
                watch.merge_completed(original, {**original, 'new/report.json': {}}, False)

    def test_stop_marker_and_foreign_process_prevent_lock_handover(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            marker = base / 'stop_after_task'; marker.write_text('user stop')
            with patch.object(watch, 'BASE', base), patch.object(watch, 'STOP', marker):
                self.assertIsNone(watch.idle_guards())
                self.assertEqual(marker.read_text(), 'user stop')
                marker.unlink()
                with patch.object(watch, 'local_processes', return_value=[{'pid': 99, 'script': 'worker'}]):
                    self.assertIsNone(watch.idle_guards())
                held = watch.acquire(base / 'evaluation.lock')
                try:
                    with patch.object(watch, 'local_processes', return_value=[]):
                        self.assertIsNone(watch.idle_guards())
                finally:
                    held.close()

    def test_requires_two_idle_observations_then_starts_one_controller(self):
        class EndFixture(Exception):
            pass
        class Guard:
            def close(self):
                pass
        class Child:
            pid = 123
            def poll(self):
                return None
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); diag = base / 'diagnostics'; diag.mkdir()
            ledger = diag / 'ledger.json'; marker = base / 'stop'
            statuses, calls = [], []
            def record(status, **items):
                statuses.append(status)
                if status == 'restarted':
                    raise EndFixture()
            def processes(names):
                return [{'pid': 999}] if names == {'update_progress.py'} else []
            def launch(command, **kwargs):
                calls.append(command)
                return Child()
            with patch.object(watch, 'BASE', base), patch.object(watch, 'DIAG', diag), \
                 patch.object(watch, 'STOP', marker), patch.object(watch, 'LEDGER', ledger), \
                 patch.object(watch, 'read', return_value={'status': 'baseline_testing'}), \
                 patch.object(watch, 'startup_ledger', return_value={'sources': {}, 'completed': {}, 'restart_count': 0}), \
                 patch.object(watch, 'verify_sources'), patch.object(watch, 'completed_reports', return_value={}), \
                 patch.object(watch, 'incomplete_fingerprint', return_value={}), \
                 patch.object(watch, 'local_processes', side_effect=processes), \
                 patch.object(watch, 'idle_guards', side_effect=lambda: (Guard(), Guard())), \
                 patch.object(watch, 'acquire', return_value=None), patch.object(watch.time, 'sleep'), \
                 patch.object(watch.subprocess, 'Popen', side_effect=launch), patch.object(watch, 'state', side_effect=record):
                with self.assertRaises(EndFixture):
                    watch.run(30, 2)
            self.assertEqual(statuses, ['watching', 'idle_first_observation', 'restarted'])
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0][2].endswith('continue_all.py'))
            self.assertEqual(json.loads(ledger.read_text())['restart_count'], 1)
            self.assertFalse(marker.exists())

    def test_source_change_stops_only_watchdog_without_touching_marker_or_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); diag = base / 'diagnostics'; diag.mkdir()
            marker = base / 'stop'; ledger = diag / 'ledger.json'
            with patch.object(watch, 'BASE', base), patch.object(watch, 'DIAG', diag), \
                 patch.object(watch, 'STOP', marker), patch.object(watch, 'LEDGER', ledger), \
                 patch.object(watch, 'read', return_value={'status': 'baseline_testing'}), \
                 patch.object(watch, 'startup_ledger', return_value={'sources': {}, 'completed': {}, 'restart_count': 0}), \
                 patch.object(watch, 'verify_sources', side_effect=RuntimeError('source changed')), \
                 patch.object(watch.time, 'sleep'), patch.object(watch, 'state'), \
                 patch.object(watch.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    watch.run(30, 2)
            launch.assert_not_called()
            self.assertFalse(marker.exists())

    def test_never_adopts_an_evaluation_without_visible_scheduler_or_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / 'ledger.json'
            ledger.write_text(json.dumps({'sources': {'old': {}}, 'completed': {}, 'restart_count': 0}))
            with patch.object(watch, 'LEDGER', ledger), patch.object(watch, 'completed_reports', return_value={}), \
                 patch.object(watch, 'local_processes', return_value=[]), patch.object(watch, 'verify_sources'):
                with self.assertRaisesRegex(RuntimeError, 'visible already-running benchmark'):
                    watch.startup_ledger()


if __name__ == '__main__':
    unittest.main()
