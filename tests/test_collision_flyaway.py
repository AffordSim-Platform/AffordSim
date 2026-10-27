"""Behavioral checks for evidence-based collision-flyaway classification."""
import importlib.util
from pathlib import Path
import sys
import unittest

path = Path(__file__).resolve().parents[1] / 'source/isaaclab/isaaclab/motion_collect/collision_flyaway.py'
spec = importlib.util.spec_from_file_location('collision_flyaway_test', path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def state(x=0, v=0):
    return m.BodyState((x, 0, 0), (v, 0, 0))


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.d = m.CollisionFlyawayDetector({0: ['cup', 'rack'], 1: ['cup', 'rack']})
    def snap(self, t, c0=state(), r0=state(), c1=state(), r1=state(), contacts=()):
        return self.d.step(t, {(0,'cup'):c0,(0,'rack'):r0,(1,'cup'):c1,(1,'rack'):r1},contacts)
    def test_collision_impulse_velocity_jump_and_displacement_required(self):
        self.snap(0)
        self.snap(.01, c0=state(v=3), contacts=[m.ObjectContact(0,'cup','rack',.03,{'actor0':'cup'})])
        self.assertFalse(self.d.excluded_envs())
        self.snap(.11,c0=state(x=.3,v=3))
        event = self.d.trial_result(0,contact_reporting_verified=True)
        self.assertTrue(event['excluded']); self.assertEqual(event['status'],'complete')
        self.assertEqual(event['events'][0]['other_object'],'rack')
        self.assertEqual(event['events'][0]['contact_details']['actor0'],'cup')
        self.assertFalse(self.d.trial_result(1,contact_reporting_verified=True)['excluded'])
    def test_plain_fall_or_fast_motion_without_object_contact_is_not_excluded(self):
        self.snap(0); self.snap(.01,c0=state(v=4)); self.snap(.11,c0=state(x=.4,v=4))
        self.assertFalse(self.d.excluded_envs())
    def test_robot_contact_and_table_contact_do_not_count(self):
        self.snap(0); self.snap(.01,c0=state(v=4),contacts=[m.ObjectContact(0,'cup','robot',1),m.ObjectContact(0,'cup','table',1)])
        self.snap(.11,c0=state(x=.4,v=4)); self.assertFalse(self.d.excluded_envs())
    def test_normal_contact_without_speed_or_displacement_does_not_count(self):
        self.snap(0);self.snap(.01,c0=state(v=.1),contacts=[m.ObjectContact(0,'cup','rack',1)])
        self.snap(.11,c0=state(x=.01,v=0));self.assertFalse(self.d.excluded_envs())
    def test_high_speed_already_present_before_contact_is_not_enough(self):
        self.snap(0,c0=state(v=3));self.snap(.01,c0=state(v=3),contacts=[m.ObjectContact(0,'cup','rack',1)])
        self.snap(.11,c0=state(x=.3,v=3));self.assertFalse(self.d.excluded_envs())
    def test_insufficient_impulse_does_not_count(self):
        self.snap(0);self.snap(.01,c0=state(v=3),contacts=[m.ObjectContact(0,'cup','rack',0)])
        self.snap(.11,c0=state(x=.3,v=3));self.assertFalse(self.d.excluded_envs())
    def test_delayed_displacement_after_window_does_not_count(self):
        self.snap(0);self.snap(.01,c0=state(v=3),contacts=[m.ObjectContact(0,'cup','rack',1)])
        self.snap(.52,c0=state(x=.3,v=3));self.assertFalse(self.d.excluded_envs())
    def test_unavailable_coverage_never_silently_marks_clean(self):
        self.snap(0);r=self.d.trial_result(0,contact_reporting_verified=False)
        self.assertEqual(r['status'],'unavailable');self.assertIsNone(r['excluded'])
    def test_per_episode_reset_drops_old_state_and_evidence(self):
        self.snap(0);self.snap(.01,c0=state(v=3),contacts=[m.ObjectContact(0,'cup','rack',1)])
        self.snap(.11,c0=state(x=.3,v=3));self.d.begin_episode()
        self.assertFalse(self.d.excluded_envs());self.snap(0)
    def test_partial_and_nonfinite_samples_are_rejected(self):
        with self.assertRaises(ValueError): self.d.step(0,{(0,'cup'):state()},[])
        with self.assertRaises(ValueError): self.snap(0,c0=state(v=float('nan')))
    def test_times_must_increase_and_thresholds_must_be_positive(self):
        self.snap(0)
        with self.assertRaises(ValueError): self.snap(0)
        with self.assertRaises(ValueError): m.FlyawayThresholds(min_speed_m_s=0)
    def test_different_environments_and_unknown_contacts_are_isolated(self):
        self.snap(0);self.snap(.01,c0=state(v=3),contacts=[m.ObjectContact(1,'cup','rack',1)])
        self.snap(.11,c0=state(x=.3,v=3));self.assertFalse(self.d.excluded_envs())

# Load the PhysX adapter under a minimal package without importing Isaac Lab.
import types
_pkg = types.ModuleType('flyaway_fake_pkg')
_pkg.__path__ = []
sys.modules[_pkg.__name__] = _pkg
sys.modules['flyaway_fake_pkg.collision_flyaway'] = m
_adapter_path = path.with_name('collision_flyaway_physx.py')
_adapter_spec = importlib.util.spec_from_file_location('flyaway_fake_pkg.collision_flyaway_physx', _adapter_path)
a = importlib.util.module_from_spec(_adapter_spec)
sys.modules[_adapter_spec.name] = a
_adapter_spec.loader.exec_module(a)


class ContactAdapterTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {'/World/envs/env_0/cup': (0,'cup'), '/World/envs/env_0/rack/link': (0,'rack'),
                        '/World/envs/env_1/cup': (1,'cup')}
        self.c = a.PhysxContactCapture(self.mapping, decode_path=lambda p:p)
    def header(self, p, q, n=1, offset=0):
        return types.SimpleNamespace(actor0=p,actor1=q,collider0=p+'/shape',collider1=q+'/shape',
            num_contact_data=n,contact_data_offset=offset,type=1,stage_id=42)
    def test_pair_reports_and_sums_impulses_per_substep(self):
        h=self.header('/World/envs/env_0/cup','/World/envs/env_0/rack/link',n=2)
        self.c.on_contact_report([h],[types.SimpleNamespace(impulse=(.03,.04,0)),types.SimpleNamespace(impulse=.02)])
        pair=self.c.drain();self.assertEqual(len(pair),1);self.assertAlmostEqual(pair[0].impulse_ns,.07)
        self.assertEqual((pair[0].env,pair[0].object_a,pair[0].object_b),(0,'cup','rack'))
        self.assertEqual(self.c.drain(),[])
    def test_robot_and_environment_contacts_are_ignored(self):
        h=self.header('/World/envs/env_0/cup','/World/envs/env_0/robot/finger')
        self.c.on_contact_report([h],[types.SimpleNamespace(impulse=1)])
        self.assertEqual(self.c.drain(),[])
    def test_bad_range_and_cross_env_fail_closed(self):
        self.c.on_contact_report([self.header('/World/envs/env_0/cup','/World/envs/env_0/rack/link',offset=2)],[])
        with self.assertRaises(RuntimeError):self.c.drain()
        c=a.PhysxContactCapture(self.mapping,decode_path=lambda p:p)
        c.on_contact_report([self.header('/World/envs/env_0/cup','/World/envs/env_1/cup')],[types.SimpleNamespace(impulse=1)])
        with self.assertRaises(RuntimeError):c.drain()
    def test_actor_environment_path_consistency(self):
        with self.assertRaises(ValueError): a.validate_actor_mapping({'/World/envs/env_0/cup':(1,'cup')})


if __name__ == '__main__':
    unittest.main()
