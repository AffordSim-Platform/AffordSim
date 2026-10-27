"""PhysX object-object contact report adapter for collision_flyaway.

Requires an Isaac/PhysX runtime; importing this module itself does not. Call
``enable_contact_reporting`` after task assets are spawned and before PhysX
scene initialization whenever possible. Use actual rigid actor paths for all
per-env rigid objects and articulation links. Subscribe once per task, retain
this object for the entire collector lifetime, and drain exactly once after
*each* simulation substep (not once per 30-substep env.step). Feed drained
contacts with same-substep post-physics body states to the detector.

Coverage is not established by an empty contact queue. The caller must run a
known-contact GPU self-test and check all path/API setup and stream health.
"""
from __future__ import annotations

import math
import threading
from typing import Mapping
from .collision_flyaway import ObjectContact


def impulse_magnitude(value):
    """Get magnitude of a scalar or 3D PhysX impulse, in stage SI units."""
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        xyz = tuple(float(x) for x in value)
        if len(xyz) != 3:
            raise ValueError('PhysX impulse must be scalar or 3-vector')
        magnitude = math.sqrt(sum(x*x for x in xyz))
    if not math.isfinite(magnitude):
        raise ValueError('PhysX impulse must be finite')
    return magnitude


def validate_actor_mapping(actor_to_object: Mapping):
    """Validate exact PhysX actor path -> (environment index, task name)."""
    if not actor_to_object:
        raise ValueError('no task-object PhysX actors available')
    result = {}
    for raw_path, identity in actor_to_object.items():
        path = str(raw_path)
        if not path.startswith('/World/envs/env_') or path.endswith('/'):
            raise ValueError(f'invalid task actor path: {path}')
        if not isinstance(identity, (tuple, list)) or len(identity) != 2:
            raise ValueError(f'invalid actor mapping for {path}')
        env, name = identity
        if not isinstance(env, int) or env < 0 or not isinstance(name, str) or not name:
            raise ValueError(f'invalid actor identity for {path}')
        expected = f'/World/envs/env_{env}/'
        if not path.startswith(expected):
            raise ValueError(f'actor env does not match path: {path}')
        if path in result:
            raise ValueError(f'duplicate actor path: {path}')
        result[path] = (env, name)
    return result


def enable_contact_reporting(stage, actor_to_object: Mapping):
    """Add zero-threshold PhysX report API without changing sleep or mass.

    Must be used before the physics scene is constructed when possible. It
    touches only explicitly enumerated task rigid bodies and articulation
    links; it never enables sensors on the robot or table. Applying an API is
    *not* proof that the GPU event stream is functional.
    """
    from pxr import PhysxSchema, UsdPhysics
    actor_to_object = validate_actor_mapping(actor_to_object)
    results = []
    for path in actor_to_object:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f'not a valid USD rigid actor: {path}')
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr().Set(0.0)
        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI) or float(api.GetThresholdAttr().Get()) != 0.0:
            raise RuntimeError(f'could not enable contact reporting: {path}')
        results.append(path)
    return results


class PhysxContactCapture:
    """Capture same-environment contacts between explicitly tracked actors.

    ``decode_path`` is injectable for tests. By default it lazily uses the
    PhysX packed-path decoder. ``contact_interface`` is injectable as well.
    Invoke ``start`` only after reporter APIs are installed and keep the
    returned subscription alive. Call ``drain`` after *every* physics substep.
    """
    def __init__(self, actor_to_object: Mapping, *, decode_path=None, contact_interface=None):
        self.actor_to_object = validate_actor_mapping(actor_to_object)
        if decode_path is None:
            from pxr import PhysicsSchemaTools
            decode_path = lambda packed: str(PhysicsSchemaTools.intToSdfPath(packed))
        self.decode_path = decode_path
        self.interface = contact_interface
        self.subscription = None
        self._lock = threading.Lock()
        self._pending = []
        self._error = None
        self.callback_count = 0
        self.total_headers = 0
        self.pair_headers = 0
        self.drains = 0

    def start(self):
        if self.subscription is not None:
            raise RuntimeError('contact capture already started')
        if self.interface is None:
            from omni.physx import get_physx_simulation_interface
            self.interface = get_physx_simulation_interface()
        self.subscription = self.interface.subscribe_contact_report_events(self.on_contact_report)
        if self.subscription is None:
            raise RuntimeError('PhysX did not create a contact subscription')
        return self

    def on_contact_report(self, headers, data):
        """PhysX callback: do not let callback exceptions kill simulation."""
        try:
            captured = []
            count = 0
            for header in headers:
                count += 1
                actor0 = str(self.decode_path(header.actor0))
                actor1 = str(self.decode_path(header.actor1))
                identity0 = self.actor_to_object.get(actor0)
                identity1 = self.actor_to_object.get(actor1)
                if identity0 is None or identity1 is None:
                    continue  # robot/table/ground or an explicitly untracked actor
                env0, object0 = identity0
                env1, object1 = identity1
                if env0 != env1:
                    raise RuntimeError(f'contact crosses task environments: {actor0} / {actor1}')
                if object0 == object1:
                    continue
                offset, n = int(header.contact_data_offset), int(header.num_contact_data)
                if offset < 0 or n < 0 or offset+n > len(data):
                    raise RuntimeError('PhysX contact report has an invalid data range')
                if not n:  # e.g. contact-lost header, no impulse
                    continue
                impulse = sum(impulse_magnitude(data[i].impulse) for i in range(offset, offset+n))
                collider0 = str(self.decode_path(header.collider0))
                collider1 = str(self.decode_path(header.collider1))
                captured.append(ObjectContact(env0, object0, object1, impulse, details={
                    'actor0': actor0, 'actor1': actor1, 'collider0': collider0, 'collider1': collider1,
                    'contact_count': n, 'header_type': int(header.type), 'stage_id': int(header.stage_id)}))
            with self._lock:
                self.callback_count += 1
                self.total_headers += count
                self.pair_headers += len(captured)
                self._pending.extend(captured)
        except Exception as exc:
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'

    def drain(self):
        """Consume only the reports delivered since the previous substep."""
        with self._lock:
            self.drains += 1
            result, self._pending = self._pending, []
            error = self._error
        if error:
            raise RuntimeError(f'PhysX contact stream is unreliable: {error}')
        return result

    def stats(self):
        with self._lock:
            return {'callback_count': self.callback_count, 'total_headers': self.total_headers,
                    'pair_headers': self.pair_headers, 'drains': self.drains,
                    'error': self._error, 'actor_count': len(self.actor_to_object)}

    def close(self):
        self.subscription = None
