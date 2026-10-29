"""Opt-in PhysX robot/cabinet contacts for the release-retreat and return phases.

Each view is an explicitly verified robot-link/cabinet-link pair in every env.
A missing pair, failed query, or unsampled phase is unknown, never no-contact.
This monitor does not change physics, trajectory actions, or flyaway results.
"""
from __future__ import annotations

import math


class ContactAccumulator:
    """Per-environment contact summaries from verified per-substep impulses."""
    def __init__(self, num_envs, pairs):
        if num_envs < 1 or not pairs:
            raise ValueError('Contact monitoring needs environments and actor pairs')
        self.num_envs = num_envs
        self.pairs = list(pairs)
        self.started = False
        self.finished = False
        self.failure = None
        self.sampled_substeps = 0
        self.contacts = [dict() for _ in range(num_envs)]

    def start(self):
        self.started = True
        self.finished = False
        self.failure = None
        self.sampled_substeps = 0
        self.contacts = [dict() for _ in range(self.num_envs)]

    def record_substep(self, values):
        """values: pair -> env -> impulse xyz, all in the verified env order."""
        if not self.started or self.finished:
            raise RuntimeError('Contact phase is not active')
        if len(values) != len(self.pairs):
            raise ValueError('Incomplete robot/cabinet pair sample')
        step = self.sampled_substeps
        for pair, rows in zip(self.pairs, values):
            if len(rows) != self.num_envs:
                raise ValueError('Incomplete environment sample')
            for env, xyz in enumerate(rows):
                if len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz):
                    raise ValueError('Invalid pair impulse')
                impulse = math.sqrt(sum(float(v) ** 2 for v in xyz))
                if impulse <= 0.0:
                    continue
                event = self.contacts[env].setdefault(pair, {
                    'robot_body': pair[0], 'cabinet_body': pair[1],
                    'first_substep': step, 'last_substep': step,
                    'nonzero_substeps': 0, 'max_impulse_ns': 0.0,
                })
                event['last_substep'] = step
                event['nonzero_substeps'] += 1
                event['max_impulse_ns'] = max(event['max_impulse_ns'], impulse)
        self.sampled_substeps += 1

    def stop(self):
        self.finished = True
        return self.results()

    def results(self):
        complete = bool(self.started and self.finished and not self.failure and self.sampled_substeps > 0)
        reason = self.failure or ('contact phase not started' if not self.started else
                                  'contact phase not stopped' if not self.finished else
                                  'no contact substeps sampled' if not self.sampled_substeps else None)
        return [{
            'status': 'complete' if complete else 'unknown',
            'contact_detected': bool(self.contacts[env]) if complete else None,
            'contacts': list(self.contacts[env].values()),
            'sampled_substeps': self.sampled_substeps,
            'configured_actor_pairs': len(self.pairs),
            'reason': reason,
            'source': 'physx_pairwise_normal_impulse',
            'limitation': 'Opposing normals may cancel in pairwise net impulse; zero impulse is not proof of no physical touch.',
        } for env in range(self.num_envs)]


class UnavailableRetreatContactMonitor:
    """Preserve the error and provide the same interface when setup fails."""
    def __init__(self, num_envs, reason):
        self.num_envs = num_envs
        self.reason = str(reason)

    def start(self):
        pass

    def stop(self):
        return self.results()

    def after_substep(self):
        pass

    def results(self):
        return [{'status': 'unknown', 'contact_detected': None, 'contacts': [],
                 'sampled_substeps': 0, 'configured_actor_pairs': 0,
                 'reason': self.reason, 'source': 'physx_pairwise_normal_impulse'}
                for _ in range(self.num_envs)]


class RobotCabinetContactMonitor:
    """Verified PhysX pairwise contact views, sampled only while phase is active."""
    def __init__(self, collector, cabinet_name):
        import omni.physics.tensors as tensors
        from pxr import Usd, UsdPhysics, PhysxSchema

        self.num_envs = collector.num_envs
        self.dt = float(collector.env.env.physics_dt)
        self.stage = collector.world.stage
        self.sim = tensors.create_simulation_view('torch')
        self.sim.set_subspace_roots('/')
        layouts = []
        for env in range(self.num_envs):
            roots = {
                'robot': str(collector.robot_prim_path).replace('env_0', f'env_{env}'),
                'cabinet': f'/World/envs/env_{env}/{cabinet_name}',
            }
            layout = {}
            for kind, root_path in roots.items():
                root = self.stage.GetPrimAtPath(root_path)
                if not root.IsValid():
                    raise RuntimeError(f'Missing {kind} root {root_path}')
                bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
                if not bodies:
                    raise RuntimeError(f'No rigid actors under {root_path}')
                actors = {}
                for body in bodies:
                    if not body.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        raise RuntimeError(f'Contact reporter missing on {body.GetPath()}')
                    path = str(body.GetPath())
                    token = path[len(root_path):] or '/'
                    if token in actors:
                        raise RuntimeError(f'Duplicate rigid actor {path}')
                    actors[token] = path
                layout[kind] = actors
            layouts.append(layout)
        for kind in ('robot', 'cabinet'):
            if any(sorted(x[kind]) != sorted(layouts[0][kind]) for x in layouts):
                raise RuntimeError(f'{kind} actors differ across environments')
        self.views = []
        pairs = []
        for robot_token in sorted(layouts[0]['robot']):
            for cabinet_token in sorted(layouts[0]['cabinet']):
                robot_paths = [x['robot'][robot_token] for x in layouts]
                cabinet_paths = [x['cabinet'][cabinet_token] for x in layouts]
                view = self.sim.create_rigid_contact_view(robot_paths,
                    filter_patterns=[[x] for x in cabinet_paths])
                sensors = [str(p) for p in view.sensor_paths]
                filters = list(view.filter_paths)
                if len(filters) == self.num_envs and all(isinstance(x, (list, tuple)) and len(x) == 1 for x in filters):
                    filters = [str(x[0]) for x in filters]
                else:
                    filters = [str(x) for x in filters]
                if (view.sensor_count != self.num_envs or view.filter_count != 1 or
                        len(sensors) != self.num_envs or len(filters) != self.num_envs):
                    raise RuntimeError(f'Incomplete robot/cabinet contact view: {robot_token}/{cabinet_token}')
                indices = []
                for sensor, filtered in zip(sensors, filters):
                    if sensor not in robot_paths:
                        raise RuntimeError('Unexpected robot sensor actor')
                    env = robot_paths.index(sensor)
                    if filtered != cabinet_paths[env]:
                        raise RuntimeError('Robot/cabinet contact filter env mismatch')
                    indices.append(env)
                if set(indices) != set(range(self.num_envs)):
                    raise RuntimeError('Robot/cabinet contact view omits or duplicates environments')
                pairs.append((robot_token, cabinet_token))
                self.views.append((view, indices))
        self.accumulator = ContactAccumulator(self.num_envs, pairs)

    def start(self):
        self.accumulator.start()

    def stop(self):
        return self.accumulator.stop()

    def results(self):
        return self.accumulator.results()

    def after_substep(self):
        if not self.accumulator.started or self.accumulator.finished or self.accumulator.failure:
            return
        try:
            import torch
            packets = [view.get_contact_force_matrix(1.0).reshape(-1) for view, _ in self.views]
            flat = torch.cat(packets).detach().cpu().tolist()
            offset = 0
            values = []
            for _, indices in self.views:
                rows = [None] * self.num_envs
                for row, env in enumerate(indices):
                    rows[env] = flat[offset + row*3:offset + (row+1)*3]
                offset += self.num_envs * 3
                values.append(rows)
            if offset != len(flat):
                raise RuntimeError('Unexpected contact impulse tensor length')
            self.accumulator.record_substep(values)
        except Exception as exc:
            self.accumulator.failure = f'{type(exc).__name__}: {exc}'
