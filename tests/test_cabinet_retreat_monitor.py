"""Verify the scoped robot/cabinet contact evidence and unknown states."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cabinet_retreat_monitor import (ContactAccumulator, RobotCabinetContactMonitor,
                                     UnavailableRetreatContactMonitor)


def test_partial_or_unsampled_monitoring_is_never_called_clear():
    m = ContactAccumulator(2, [('/robot/link', '/cabinet/drawer')])
    assert all(r['status'] == 'unknown' and r['contact_detected'] is None for r in m.results())
    m.start()
    m.record_substep([[[0, 0, 0], [0, 0, 0]]])
    assert all(r['status'] == 'unknown' for r in m.results())
    m.failure = 'contact query failed'
    results = m.stop()
    assert all(r['status'] == 'unknown' and r['contact_detected'] is None for r in results)
    assert all(r['reason'] == 'contact query failed' for r in results)
    m.start()
    assert all(r['status'] == 'unknown' for r in m.stop())
    unavailable = UnavailableRetreatContactMonitor(2, 'failed setup')
    unavailable.start()
    assert all(r['status'] == 'unknown' and r['contact_detected'] is None for r in unavailable.stop())


def test_pair_impulses_are_mapped_to_exact_environment_and_actor():
    class View:
        def __init__(self, values): self.values = values
        def get_contact_force_matrix(self, dt):
            assert dt == 1.0
            return torch.tensor(self.values)
    m = RobotCabinetContactMonitor.__new__(RobotCabinetContactMonitor)
    m.num_envs = 2
    pairs = [('/robot/wrist', '/cabinet/drawer'), ('/robot/finger', '/cabinet/base')]
    m.accumulator = ContactAccumulator(2, pairs)
    # PhysX may return env rows in any order: first view returns env 1 then 0.
    m.views = [(View([[[0.1, 0.0, 0.0]], [[0, 0, 0]]]), [1, 0]),
               (View([[[0, 0, 0]], [[0, 0, 0]]]), [0, 1])]
    m.start()
    m.after_substep()
    result = m.stop()
    assert result[0]['status'] == 'complete' and result[0]['contact_detected'] is False
    assert result[1]['status'] == 'complete' and result[1]['contact_detected'] is True
    assert len(result[1]['contacts']) == 1
    assert result[1]['contacts'][0]['robot_body'] == '/robot/wrist'
    assert result[1]['contacts'][0]['cabinet_body'] == '/cabinet/drawer'
    assert abs(result[1]['contacts'][0]['max_impulse_ns'] - .1) < 1e-6
    assert result[1]['sampled_substeps'] == 1
    m.start()
    m.views = [(View([[[float('nan'), 0, 0]], [[0, 0, 0]]]), [0, 1]), m.views[1]]
    m.after_substep()
    assert all(r['status'] == 'unknown' for r in m.stop())
