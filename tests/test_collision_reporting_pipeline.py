"""Check collision exclusions use one denominator across reports and media."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    source = ROOT / path
    if not source.is_file():
        raise unittest.SkipTest(f"Local experiment utility is unavailable: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module('pipeline_runner', 'scripts/run_benchmark_50.py')
audit = module('pipeline_audit', 'outputs/full_retest_v2/audit_retest.py')
finalizer = module('pipeline_finalizer', 'outputs/full_retest_v2/apply_final_ranges.py')
gallery = module('pipeline_gallery', 'scripts/render_benchmark_gallery.py')


class CollisionReportingPipelineTest(unittest.TestCase):
    def make_round(self, root, unknown=False):
        source = root / 'source.yaml'
        source.write_text(yaml.safe_dump({'objects': {'cup': {'usd_path': 'cup.usd',
            'position_range': {'x': [0.4, 0.6]}}},
            'primitives': [{'type': 'place', 'params': {'success_eval': True}}]}))
        task = {'task_id': 'cup', 'robot': 'robot', 'config_path': str(source), 'num_envs': 20,
                'env_task_name': 'env', 'robot_cfg_name': 'robot.yml', 'robot_type': 'robot'}
        directory = root / 'results/cup/round_00'
        directory.mkdir(parents=True)
        config = directory / 'task.yaml'
        runner.prepare_config(source, config, trials=10, scale=1, task=task, num_envs=10)
        physics = root / 'source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick_r2s_env_cfg.py'
        physics.parent.mkdir(parents=True)
        physics.write_text('fixture physics')
        detector = root / 'source/isaaclab/isaaclab/motion_collect/collision_flyaway.py'
        detector.parent.mkdir(parents=True)
        detector.write_text('fixture detector')
        monitor = detector.with_name('collision_flyaway_monitor.py')
        monitor.write_text('fixture collision monitor')
        collector = detector.with_name('datacollector.py')
        collector.write_text('fixture collector')
        trials = []
        for index in range(1, 11):
            trial_dir = directory / f'trial_{index:03d}'
            for camera in ('external_cam_side', 'wrist_cam'):
                folder = trial_dir / 'original_images' / camera
                folder.mkdir(parents=True)
                (folder / 'step_0000.webp').write_bytes(b'frame')
            (trial_dir / 'initial.png').write_bytes(b'initial')
            excluded = index in (2, 10)
            evidence = {'object': 'cup', 'other_object': 'rack', 'collision_time_s': 1.0,
                        'detection_time_s': 1.1, 'impulse_ns': 1.0, 'velocity_jump_m_s': 1.0,
                        'displacement_m': 0.2}
            detection = {'status': 'complete', 'excluded': excluded, 'events': [evidence] if excluded else [],
                         'observed_substeps': 100, 'thresholds': {'fixture_speed_m_s': 1.0}}
            if unknown and index == 6:
                detection = {'status': 'unavailable', 'excluded': None, 'events': []}
            trials.append({'trial': index, 'success': index <= 4, 'program_error': False,
                'failure_reason': 'task_failed', 'collision_flyaway': detection,
                'initial_metadata': [{'randomization': {'texture_randomization_applied': {
                    'enabled': True, 'assignments': [{'target': 'table', 'status': 'bound',
                                                     'texture_file': f'texture_{index % 2}.png'}]}}}],
                'directory': str(trial_dir), 'recorded_steps': 1,
                'frame_counts': {'original_images/external_cam_side': 1, 'original_images/wrist_cam': 1}})
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        report = {'task_id': 'cup', 'robot': 'robot', 'seed': runner.task_seed(42, 'cup'),
            'num_envs': 10, 'completed': True, 'returncode': 0, 'timed_out': False,
            'source_config_sha256': digest(source), 'config_sha256': digest(config),
            'range_scale': 1.0, 'expected_max_depenetration_velocity': 2.0,
            'expected_depenetration_overrides': {}, 'max_depenetration_velocity_verified': {'cup': [2.0]},
            'physics_source_sha256': digest(physics), 'modules_after_collection': {'collector': str(physics)},
            'collision_detector_source_sha256': digest(detector),
            'collision_detector_monitor_sha256': digest(monitor),
            'collision_detector_collector_sha256': digest(collector),
            'trials': trials, 'summary': runner.summarize(trials, 10)}
        path = directory / 'report.json'
        path.write_text(json.dumps(report))
        return task, path, source, report

    def test_excluded_attempts_are_removed_from_all_reported_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, path, source, report = self.make_round(root)
            with patch.object(audit, 'ROOT', root):
                check = audit.check_round(task, path)
            final = finalizer.check_round(root, runner, task, path.parent, 0, source, {})
            rendered = gallery.read_rounds(task, root / 'results')[0]
            self.assertTrue(check['completed'], check['issues'])
            self.assertTrue(rendered['completed'])
            self.assertEqual((check['successes'], check['valid_trials'], check['excluded_trials']), (3, 8, 2))
            self.assertEqual(check['success_rate'], 3 / 8)
            self.assertEqual(final['success_rate_percent'], 37.5)
            self.assertEqual(rendered['success_rate'], 3 / 8)
            self.assertEqual(gallery.trial_status(report['trials'][1]), 'EXCLUDED: FLYAWAY')

    def test_unknown_collision_evidence_never_produces_a_success_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, path, source, report = self.make_round(root, unknown=True)
            with patch.object(audit, 'ROOT', root):
                check = audit.check_round(task, path)
            final = finalizer.check_round(root, runner, task, path.parent, 0, source, {})
            rendered = gallery.read_rounds(task, root / 'results')[0]
            self.assertTrue(check['completed'], check['issues'])
            self.assertEqual(check['collision_unknown_trials'], 1)
            self.assertIsNone(check['success_rate'])
            self.assertIsNone(final['success_rate_percent'])
            self.assertIsNone(rendered['success_rate'])
            self.assertEqual(gallery.trial_status(report['trials'][5]), 'COLLISION UNKNOWN')

    def test_all_excluded_is_undefined_and_detector_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, path, source, report = self.make_round(root)
            example = report['trials'][1]['collision_flyaway']
            for trial in report['trials']:
                trial['collision_flyaway'] = example
            report['summary'] = runner.summarize(report['trials'], 10)
            path.write_text(json.dumps(report))
            with patch.object(audit, 'ROOT', root):
                check = audit.check_round(task, path)
            final = finalizer.check_round(root, runner, task, path.parent, 0, source, {})
            self.assertTrue(check['completed'], check['issues'])
            self.assertEqual((check['excluded_trials'], check['valid_trials']), (10, 0))
            self.assertIsNone(check['success_rate'])
            self.assertIsNone(final['success_rate_percent'])
            monitor = root / 'source/isaaclab/isaaclab/motion_collect/collision_flyaway_monitor.py'
            monitor.write_text('tampered collision monitor')
            with patch.object(audit, 'ROOT', root):
                self.assertIn('collision_detector_source_changed_or_missing', audit.check_round(task, path)['issues'])
            with self.assertRaisesRegex(ValueError, 'Collision detector source'):
                finalizer.check_round(root, runner, task, path.parent, 0, source, {})


if __name__ == '__main__':
    unittest.main()
