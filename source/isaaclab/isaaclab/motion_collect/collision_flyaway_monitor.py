"""Opt-in substep PhysX tensor instrumentation for collision flyaways.

A report is complete only when all task object and articulation actors were
found in every environment and every cross-object pair view matches its actor
paths. Normal forces are pair-wise aggregate PhysX impulses; opposing contact
normals may cancel, so this conservative detector is not exhaustive.
"""
from __future__ import annotations
import math
import re

from .collision_flyaway import BodyState, CollisionFlyawayDetector, FlyawayThresholds, ObjectContact


def wrap_spawner_for_contact_reports(spawn):
    """Add report APIs to spawned task rigid actors without changing sleep/mass."""
    original = spawn.func
    def spawn_with_reports(*args, **kwargs):
        prim = original(*args, **kwargs)
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaaclab.sim.utils import find_matching_prims
        # A decorated spawner may already have cloned the asset into every
        # environment. Apply the reporter to all existing matches, not only
        # the source prim it returns. With replicate_physics=True only env_0
        # exists here and later environment clones inherit this authored API.
        pattern = str(args[0] if args else kwargs['prim_path'])
        roots = find_matching_prims(pattern)
        if not roots:
            roots = [prim]
        for root in roots:
            actor_count = 0
            for body in Usd.PrimRange(root):
                if body.HasAPI(UsdPhysics.RigidBodyAPI):
                    api = PhysxSchema.PhysxContactReportAPI.Apply(body)
                    api.CreateThresholdAttr().Set(0.0)
                    actor_count += 1
            if not actor_count:
                raise RuntimeError(f'No rigid actor under spawned task object {root.GetPath()}')
        return prim
    spawn.func = spawn_with_reports


def prepare_task_contact_spawners(env_cfg, task_config):
    """Instrument only configured task objects, including articulation links."""
    names = list(getattr(task_config, 'objects', {}).keys()) + list(getattr(task_config, 'articulations', {}).keys())
    if len(set(names)) != len(names):
        raise RuntimeError('Task object names are duplicated')
    for name in names:
        asset = getattr(env_cfg.scene, name, None)
        if asset is None or getattr(asset, 'spawn', None) is None:
            raise RuntimeError(f'No spawn config for task object {name}')
        wrap_spawner_for_contact_reports(asset.spawn)
    return names


class FlyawayMonitor:
    def __init__(self, collector, *, thresholds=None):
        import omni.physics.tensors as tensors
        from pxr import Usd, UsdPhysics, PhysxSchema
        self.num_envs = collector.num_envs
        self.dt = float(collector.env.env.physics_dt)
        self.stage = collector.world.stage
        task = collector.task_config
        names = list(getattr(task, 'objects', {}).keys()) + list(getattr(task, 'articulations', {}).keys())
        self.sim = tensors.create_simulation_view('torch')
        self.sim.set_subspace_roots('/')
        self.actor_info = {}
        layouts = []
        for env in range(self.num_envs):
            layout = {}
            for name in names:
                root_path = f'/World/envs/env_{env}/{name}'
                root = self.stage.GetPrimAtPath(root_path)
                if not root.IsValid():
                    raise RuntimeError(f'Missing task object {root_path}')
                bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
                if not bodies:
                    raise RuntimeError(f'No rigid actor below {root_path}')
                for body in bodies:
                    path = str(body.GetPath())
                    if not body.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        raise RuntimeError(f'Contact reporter not enabled: {path}')
                    # Actor names include task label and relative rigid-link path.
                    token = name + ':' + path[len(root_path):]
                    if token in layout:
                        raise RuntimeError(f'Duplicate task actor {token}')
                    layout[token] = path
                    self.actor_info[(env, token)] = {'label': name, 'path': path}
            layouts.append(layout)
        tokens = sorted(layouts[0]) if layouts else []
        if any(sorted(layout) != tokens for layout in layouts):
            raise RuntimeError('Task actors differ across environments')
        self.tokens = tokens
        self.detector = CollisionFlyawayDetector({e: tokens for e in range(self.num_envs)},
                                                  thresholds or FlyawayThresholds())
        self.bodies = []
        for token in tokens:
            paths = [layouts[e][token] for e in range(self.num_envs)]
            view = self.sim.create_rigid_body_view(paths)
            actual = [str(p) for p in view.prim_paths]
            if len(actual) != len(paths) or set(actual) != set(paths):
                raise RuntimeError(f'Rigid body tensor mapping mismatch: {token}')
            indices = [paths.index(p) for p in actual]
            self.bodies.append((token, view, indices))
        self.pairs = []
        for i, first in enumerate(tokens):
            for second in tokens[i+1:]:
                if self.actor_info[(0, first)]['label'] == self.actor_info[(0, second)]['label']:
                    continue
                a = [layouts[e][first] for e in range(self.num_envs)]
                b = [layouts[e][second] for e in range(self.num_envs)]
                view = self.sim.create_rigid_contact_view(a, filter_patterns=[[x] for x in b])
                sensors = [str(p) for p in view.sensor_paths]
                filters = list(view.filter_paths)
                # Versions may return flat one-per-sensor or nested one-per-sensor.
                if len(filters) == self.num_envs and all(isinstance(x, (list, tuple)) and len(x) == 1 for x in filters):
                    filters = [str(x[0]) for x in filters]
                else:
                    filters = [str(x) for x in filters]
                if (view.sensor_count != self.num_envs or view.filter_count != 1 or
                    len(sensors) != self.num_envs or len(filters) != self.num_envs):
                    raise RuntimeError(f'Incomplete tensor contact view for {first}/{second}')
                indices = []
                for sensor, filtered in zip(sensors, filters):
                    if sensor not in a:
                        raise RuntimeError('Unexpected PhysX contact sensor actor')
                    env = a.index(sensor)
                    if filtered != b[env]:
                        raise RuntimeError('PhysX contact filter actor/env mismatch')
                    indices.append(env)
                if set(indices) != set(range(self.num_envs)):
                    raise RuntimeError('PhysX tensor contact views omit/duplicate environments')
                self.pairs.append((first, second, view, indices))
        self.active = False
        self.failure = None
        self.step_index = 0
        self._nonzero_pair_substeps = [0] * self.num_envs
        self._max_pair_impulse_ns = [0.0] * self.num_envs

    def begin_episode(self):
        self.detector.begin_episode()
        self.active = False
        self.failure = None
        self.step_index = 0
        self._nonzero_pair_substeps = [0] * self.num_envs
        self._max_pair_impulse_ns = [0.0] * self.num_envs

    def after_reset(self):
        # Called immediately after episode env.reset, before any physics substep.
        self.active = True
        self._sample(include_contacts=False)

    def after_substep(self):
        if self.active and self.failure is None:
            try:
                self._sample(include_contacts=True)
            except Exception as exc:
                self.failure = f'{type(exc).__name__}: {exc}'
                self.active = False

    def _sample(self, include_contacts):
        import torch
        packets = []
        for token, view, indices in self.bodies:
            pos = view.get_transforms()[:, :3]
            vel = view.get_velocities()[:, :3]
            packets.append(torch.cat((pos, vel), dim=1).reshape(-1))
        for first, second, view, indices in self.pairs:
            if include_contacts:
                # dt=1 yields per-substep impulses instead of forces.
                packets.append(view.get_contact_force_matrix(1.0).reshape(-1))
            else:
                packets.append(torch.zeros(self.num_envs * 3, device=self.bodies[0][1].get_velocities().device))
        if not packets:
            raise RuntimeError('No task rigid actors to monitor')
        values = torch.cat(packets).detach().cpu().tolist()  # one GPU synchronization/substep
        offset = 0
        samples = {}
        for token, view, indices in self.bodies:
            for row, env in enumerate(indices):
                chunk = values[offset + row*6 : offset + (row+1)*6]
                samples[(env, token)] = BodyState(chunk[:3], chunk[3:])
            offset += self.num_envs * 6
        contacts = []
        substep_has_pair = [False] * self.num_envs
        for first, second, view, indices in self.pairs:
            for row, env in enumerate(indices):
                vec = values[offset + row*3 : offset + (row+1)*3]
                impulse = math.sqrt(sum(x*x for x in vec))
                if impulse > 0:
                    substep_has_pair[env] = True
                    self._max_pair_impulse_ns[env] = max(self._max_pair_impulse_ns[env], impulse)
                    a, b = self.actor_info[(env, first)], self.actor_info[(env, second)]
                    contacts.append(ObjectContact(env, first, second, impulse, details={
                        'actor0': a['path'], 'actor1': b['path'],
                        'task_object0': a['label'], 'task_object1': b['label'],
                        'normal_impulse_vector_ns': vec,
                        'source': 'physx_pairwise_normal_impulse'}))
            offset += self.num_envs * 3
        for env, observed in enumerate(substep_has_pair):
            self._nonzero_pair_substeps[env] += int(observed)
        self.detector.step(self.step_index*self.dt, samples, contacts)
        self.step_index += 1

    def results(self):
        results = []
        for env in range(self.num_envs):
            trial = self.detector.trial_result(env,
                contact_reporting_verified=self.failure is None and self.step_index > 1,
                coverage_error=self.failure or 'No complete physics substeps sampled')
            trial['contact_observation'] = {
                'source': 'physx_pairwise_normal_impulse',
                'configured_actor_pairs': len(self.pairs),
                'monitored_task_actors': len(self.tokens),
                'nonzero_pair_substeps': self._nonzero_pair_substeps[env],
                'max_pair_impulse_ns': self._max_pair_impulse_ns[env],
                'limitation': 'Net pair normal impulses can cancel; this detector is conservative.'}
            results.append(trial)
        return results
