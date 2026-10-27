"""Opt-in phase-specific PhysX depenetration limits for task rigid bodies.

The dynamic-control API reads and writes the live PhysX actors. Updating the
USD attribute alone is not sufficient for a running GPU simulation.
"""
from __future__ import annotations

import math


def _vector(value):
    return (float(value.x), float(value.y), float(value.z))


def _other_properties(properties):
    return {
        "mass": float(properties.mass),
        "moment": _vector(properties.moment),
        "center_of_mass": _vector(properties.cMassLocalPose),
        "max_contact_impulse": float(properties.max_contact_impulse),
        "solver_position_iteration_count": int(properties.solver_position_iteration_count),
        "solver_velocity_iteration_count": int(properties.solver_velocity_iteration_count),
    }


def _same_properties(left, right):
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, tuple):
            if not all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9) for x, y in zip(a, b)):
                return False
        elif not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9):
            return False
    return True


class PhaseDepenetrationPolicy:
    """Transactionally set selected rigid bodies in every vectorized environment."""

    def __init__(self, stage, num_envs, config, valid_object_names, *, usd=None, physx_schema=None, dc=None, invalid_handle=None):
        if usd is None or physx_schema is None:
            from pxr import PhysxSchema, Usd
            usd = usd or Usd
            physx_schema = physx_schema or PhysxSchema
        if dc is None:
            from omni.isaac.dynamic_control import _dynamic_control
            dc = _dynamic_control.acquire_dynamic_control_interface()
            invalid_handle = _dynamic_control.INVALID_HANDLE
        self._dc = dc
        self._config = config
        self._num_envs = num_envs
        self.events = []
        self.episode = None
        names = config["objects"]
        if (isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0
                or not isinstance(names, list) or not names
                or any(not isinstance(n, str) or not n or not all(c.isascii() and (c.isalnum() or c == '_') for c in n) for n in names)
                or len(names) != len(set(names)) or not set(names) <= set(valid_object_names)
                or "robot" in names):
            raise ValueError("Phase policy needs unique task rigid objects and a positive environment count")
        self._objects = list(names)
        self._bodies = []
        seen_paths, seen_handles = set(), set()
        for env in range(num_envs):
            for name in names:
                path = f"/World/envs/env_{env}/{name}"
                root = stage.GetPrimAtPath(path)
                if not root.IsValid():
                    raise ValueError(f"Missing selected task object: {path}")
                found = []
                for prim in usd.PrimRange(root):
                    if not prim.HasAPI(physx_schema.PhysxRigidBodyAPI):
                        continue
                    prim_path = str(prim.GetPath())
                    if prim_path in seen_paths:
                        continue
                    handle = dc.get_rigid_body(prim_path)
                    if handle == invalid_handle:
                        raise RuntimeError(f"Cannot resolve live PhysX rigid body: {prim_path}")
                    if handle in seen_handles:
                        raise RuntimeError(f"Multiple selected paths resolve to the same PhysX actor: {prim_path}")
                    seen_paths.add(prim_path)
                    seen_handles.add(handle)
                    found.append(prim_path)
                    self._bodies.append((env, name, prim_path, handle))
                if not found:
                    raise ValueError(f"Selected object has no PhysX rigid body: {path}")
        # Capture each actor's original value before any phase switch.
        self._original = self._snapshot()

    def _snapshot(self):
        rows = []
        for env, name, path, handle in self._bodies:
            properties = self._dc.get_rigid_body_properties(handle)
            value = float(properties.max_depeneration_velocity)
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"Invalid PhysX depenetration value for {path}: {value}")
            rows.append((env, name, path, handle, properties, value, _other_properties(properties)))
        return rows

    def _rollback(self, rows):
        failures = []
        for _, _, path, handle, properties, expected, other in rows:
            try:
                if self._dc.set_rigid_body_properties(handle, properties) is False:
                    raise RuntimeError("dynamic-control setter returned false")
                actual = self._dc.get_rigid_body_properties(handle)
                if (not math.isclose(float(actual.max_depeneration_velocity), expected, rel_tol=1e-6, abs_tol=1e-12)
                        or not _same_properties(other, _other_properties(actual))):
                    raise RuntimeError("PhysX rollback readback differs from its snapshot")
            except Exception as exc:
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError("PhysX rollback failed: " + "; ".join(failures))

    def begin_episode(self, number):
        self.episode = int(number)
        self.events = []

    def _set_values(self, values):
        before = self._snapshot()
        if len(values) != len(before):
            raise ValueError("Phase values must match the selected actor count")
        try:
            for row, value in zip(before, values):
                env, name, path, handle, _, old_value, other = row
                if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                        or value <= 0 or value > 3.4028234663852886e38):
                    raise ValueError(f"Invalid phase velocity for {path}: {value!r}")
                if not math.isclose(old_value, value, rel_tol=1e-6, abs_tol=1e-12):
                    properties = self._dc.get_rigid_body_properties(handle)
                    properties.max_depeneration_velocity = float(value)
                    if self._dc.set_rigid_body_properties(handle, properties) is False:
                        raise RuntimeError(f"PhysX update rejected for {path}")
                actual = self._dc.get_rigid_body_properties(handle)
                if (not math.isclose(float(actual.max_depeneration_velocity), value, rel_tol=1e-6, abs_tol=1e-12)
                        or not _same_properties(other, _other_properties(actual))):
                    raise RuntimeError(f"PhysX readback or unrelated properties changed for {path}")
            readback = {}
            for env, name, path, handle in self._bodies:
                value = float(self._dc.get_rigid_body_properties(handle).max_depeneration_velocity)
                readback.setdefault(str(env), {}).setdefault(name, {})[path] = value
            return readback
        except Exception as exc:
            try:
                self._rollback(before)
            except Exception as rollback_exc:
                raise RuntimeError(f"Phase update failed: {exc}; rollback also failed: {rollback_exc}") from exc
            raise RuntimeError(f"Phase update failed and was rolled back: {exc}") from exc

    def apply(self, phase, value, active_envs=None, *, active_only=False, objects=None):
        active = [True] * self._num_envs if active_envs is None else [bool(v) for v in active_envs]
        if len(active) != self._num_envs:
            raise ValueError("Phase active mask must match the environment count")
        # None preserves the original public API; an explicit list restricts
        # the update to managed rigid objects, silently excluding articulations
        # and unknown labels.
        selected = list(self._objects) if objects is None else [n for n in self._objects if n in objects]
        event = {"episode": self.episode, "phase": str(phase), "value": float(value),
                 "objects": list(self._objects), "backend": "omni.isaac.dynamic_control",
                 "selected_objects": selected, "active_envs": active}
        try:
            if active_only or objects is not None:
                before = self._snapshot()
                values = [value if (not active_only or active[row[0]]) and row[1] in selected else row[5]
                          for row in before]
            else:
                values = [value] * len(self._bodies)
            event["physx_readback"] = self._set_values(values)
            event["status"] = "verified"
        except Exception as exc:
            event.update(status="error", error=f"{type(exc).__name__}: {exc}")
            self.events.append(event)
            raise
        self.events.append(event)
        print(f"[PHASE_DEPENETRATION] episode={self.episode} phase={phase} value={value} bodies={len(self._bodies)}", flush=True)

    def apply_reset(self):
        self.apply("reset", self._config["reset"])

    def apply_primitive(self, primitive, active_envs, *, objects=None):
        values = self._config["primitives"]
        if primitive in values:
            if primitive == "release" and not any(active_envs):
                return
            self.apply(primitive, values[primitive], active_envs,
                       active_only=(primitive == "release" or objects is not None), objects=objects)

    def restore_original(self):
        # Change only the cap, preserving any unrelated legitimate later edits.
        self._set_values([row[5] for row in self._original])
