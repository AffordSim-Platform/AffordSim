"""Conservative classification of collision-associated object flyaways.

This module has no PhysX or Isaac dependency. Its caller must sample *every*
physics substep: supply the post-step positions and linear velocities of every
tracked task object and the actual contacts and contact impulses emitted by
PhysX for the same substep. Establish an initial pre-step state before the
first contact. Contact reporting must be verified separately by the caller;
no events is not proof that contact reporting works.

Only separate task objects in the same environment can form a qualifying pair;
contact with robot links, furniture, the ground, or a missing body is ignored.
The velocity jump must occur in the reported contact substep and be followed by
large displacement during a short window. Thresholds are intentionally
conservative heuristics, not a mathematical proof of causation; save the
thresholds and complete evidence with each excluded attempt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


def _vector(value, name):
    if len(value) != 3:
        raise ValueError(f'{name} must be a 3-vector')
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f'{name} must be finite')
    return result


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


@dataclass(frozen=True)
class FlyawayThresholds:
    min_contact_impulse_ns: float = 0.001
    min_speed_m_s: float = 2.0
    min_velocity_jump_m_s: float = 1.0
    min_displacement_m: float = 0.2
    max_followup_s: float = 0.5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class BodyState:
    position: Sequence[float]
    velocity: Sequence[float]


@dataclass(frozen=True)
class ObjectContact:
    env: int
    object_a: str
    object_b: str
    impulse_ns: float
    # Caller can attach exact PhysX actor/collider paths and substep details.
    details: Mapping | None = None


class CollisionFlyawayDetector:
    """Classify one episode; ``step`` consumes one true physics substep.

    ``objects`` maps integer environment indices to the allowed task object
    names. ``samples`` maps ``(env, name)`` to BodyState. ``contacts`` contains
    same-substep actual pair contacts with nonnegative impulse magnitude. The
    collision event remains a candidate until sufficient net displacement
    occurs; absence of a qualifying event is not a collision-coverage check.
    """
    def __init__(self, objects: Mapping[int, Sequence[str]], thresholds=None):
        self.thresholds = thresholds or FlyawayThresholds()
        if not isinstance(self.thresholds, FlyawayThresholds):
            raise TypeError('thresholds must be FlyawayThresholds')
        self.objects = {}
        for env, names in objects.items():
            if not isinstance(env, int) or env < 0:
                raise ValueError('environment indices must be nonnegative integers')
            names = list(names)
            if any(not isinstance(x, str) or not x for x in names) or len(names) != len(set(names)):
                raise ValueError('object names must be unique nonempty strings per environment')
            self.objects[env] = frozenset(names)
        self.begin_episode()

    def begin_episode(self):
        self.previous = {}
        self.candidates = []
        self.events = []
        self.last_time = None
        self.substeps = 0

    def step(self, time_s: float, samples: Mapping, contacts: Sequence[ObjectContact], *, phase=None):
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('time_s must be nonnegative and finite')
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError('physics substep times must be strictly increasing')
        current = {}
        for key, state in samples.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError('sample keys must be (env, object_name)')
            env, name = key
            if env not in self.objects or name not in self.objects[env]:
                raise ValueError(f'unknown sampled task object: {key}')
            current[key] = BodyState(_vector(state.position, 'position'), _vector(state.velocity, 'velocity'))
        if set(current) != {(env, name) for env, names in self.objects.items() for name in names}:
            raise ValueError('must sample every tracked task object each physics substep')
        contacts = list(contacts)
        for contact in contacts:
            if not isinstance(contact, ObjectContact):
                raise TypeError('contacts must contain ObjectContact entries')
            if not math.isfinite(float(contact.impulse_ns)) or float(contact.impulse_ns) < 0:
                raise ValueError('contact impulse must be finite and nonnegative')
        dt = None if self.last_time is None else time_s - self.last_time
        keep = []
        for candidate in self.candidates:
            age = time_s - candidate['collision_time_s']
            if age > self.thresholds.max_followup_s:
                continue
            state = current[candidate['key']]
            candidate['peak_speed_m_s'] = max(candidate['peak_speed_m_s'], _norm(state.velocity))
            displacement = _norm(_difference(state.position, candidate['collision_position_m']))
            if displacement >= self.thresholds.min_displacement_m:
                event = {k: v for k, v in candidate.items() if k != 'key'}
                event.update({'detection_time_s': time_s, 'detection_substep': self.substeps,
                              'detection_position_m': list(state.position),
                              'displacement_m': displacement, 'thresholds': asdict(self.thresholds)})
                self.events.append(event)
            else:
                keep.append(candidate)
        self.candidates = keep
        if self.last_time is not None:
            seen = set()
            for contact in contacts:
                names = self.objects.get(contact.env, frozenset())
                a, b = contact.object_a, contact.object_b
                if a == b or a not in names or b not in names:
                    continue
                impulse = float(contact.impulse_ns)
                if impulse < self.thresholds.min_contact_impulse_ns:
                    continue
                for name, other in ((a, b), (b, a)):
                    key = (contact.env, name)
                    if key in seen or any(c['key'] == key for c in self.candidates):
                        continue
                    previous, state = self.previous[key], current[key]
                    speed = _norm(state.velocity)
                    jump = _norm(_difference(state.velocity, previous.velocity))
                    if speed < self.thresholds.min_speed_m_s or jump < self.thresholds.min_velocity_jump_m_s:
                        continue
                    self.candidates.append({'key': key, 'env': contact.env, 'object': name,
                        'other_object': other, 'phase': phase, 'collision_time_s': time_s,
                        'collision_substep': self.substeps, 'physics_dt_s': dt,
                        'impulse_ns': impulse, 'contact_details': dict(contact.details or {}),
                        'previous_velocity_m_s': list(previous.velocity),
                        'collision_velocity_m_s': list(state.velocity), 'velocity_jump_m_s': jump,
                        'collision_position_m': list(state.position), 'peak_speed_m_s': speed})
                    seen.add(key)
        self.previous = current
        self.last_time = time_s
        self.substeps += 1
        return [dict(event) for event in self.events]

    def excluded_envs(self):
        return frozenset(event['env'] for event in self.events)

    def trial_result(self, env: int, *, contact_reporting_verified: bool, coverage_error=None):
        """Return the per-trial tri-state contract used by the benchmark.

        The caller must pass True only after validating that every requested
        physics substep and object pair was monitored by the actual PhysX
        instrumentation. A detector must never infer coverage from an absence
        of reports. Failed/unsupported instrumentation gives ``unavailable``.
        """
        if env not in self.objects:
            raise ValueError('unknown environment')
        events = [dict(event) for event in self.events if event['env'] == env]
        if not contact_reporting_verified:
            return {'status': 'unavailable', 'excluded': None, 'events': events,
                    'reason': str(coverage_error or 'PhysX contact coverage not verified'),
                    'thresholds': asdict(self.thresholds), 'observed_substeps': self.substeps}
        return {'status': 'complete', 'excluded': bool(events), 'events': events,
                'thresholds': asdict(self.thresholds), 'observed_substeps': self.substeps}
