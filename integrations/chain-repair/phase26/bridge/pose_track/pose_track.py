"""pose_track: trajectory-following for managed scene objects (CPU core, NumPy only).

Problem addressed
-----------------
The current managed-object contract (SceneSession.edit / EditRequest) carries a
single static pose. Moving an inserted vehicle along a trajectory by calling
edit(set_pose) every frame increments ``revision``, which changes
``model_cache_token=f'{generation}:{revision}'`` and therefore resets the
UniAD temporal (tracking) history every frame. Native recorded actors move
every frame WITHOUT any such reset; a managed object following a trajectory
must behave the same way.

Design
------
Two event classes are kept strictly separate:

* Edit events (structural): insert / remove / set_trajectory. They change what
  exists or where it is allowed to be, so they bump ``revision`` and hence the
  cache token. Downstream history reset is CORRECT for these.
* Normal advancement: ``advance(t)`` evaluates each trajectory at the current
  simulation time and updates render pose + collision pose. It never touches
  ``revision``/``generation``, so the UniAD history cache token is stable.

``reset()`` (and only reset / a new scene) assigns a new ``generation``, which
invalidates the token and clears per-scene state -- the necessary cleanup for
reset/scene-switch is preserved.

Render/collision/time consistency
---------------------------------
Each object stores ONE current 4x4 world pose; ``render_pose`` and
``collision_pose`` return that same array object, so the rendered geometry and
the collision geometry cannot diverge. ``advance`` records the simulation time
and enforces monotonic non-decreasing time within a generation.

Diagnostic mode only: no training, no formal evaluation admission.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from uuid import uuid4
import numpy as np

_ROT_TOL = 1e-6
_TIME_TOL = 1e-9


def _check_pose(pose):
    m = np.asarray(pose, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError('pose must be a finite 4x4 matrix')
    if not np.allclose(m[3], [0., 0., 0., 1.], atol=1e-12):
        raise ValueError('pose last row must be [0,0,0,1]')
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=_ROT_TOL):
        raise ValueError('pose rotation block is not orthogonal')
    if not np.isclose(np.linalg.det(r), 1.0, atol=_ROT_TOL):
        raise ValueError('pose rotation must be right-handed (det=+1)')
    return m


def _rot_to_quat(r):
    """Rotation matrix -> unit quaternion (w, x, y, z). Shepard-free trace method."""
    t = np.trace(r)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (1, 2) if i == 0 else ((2, 0) if i == 1 else (0, 1))
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
        q = np.empty(4)
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q  # canonical hemisphere for deterministic output
    return q


def _quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0, q1, u):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9999995:  # nearly identical: lerp + normalize
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - u) * theta0) / np.sin(theta0)
    s1 = np.sin(u * theta0) / np.sin(theta0)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


class PoseTrajectory:
    """Piecewise time-parameterised rigid trajectory.

    times: (N,) strictly increasing seconds; poses: (N,4,4) rigid transforms.
    N == 1 is a static object (constant pose at any in-range time).
    bounds: 'strict' (default; evaluating outside [t0, tN] raises) or
            'hold' (clamp to the nearest endpoint pose).
    """

    def __init__(self, times, poses, bounds='strict'):
        times = np.asarray(times, dtype=float).reshape(-1)
        poses = np.asarray(poses, dtype=float)
        if times.ndim != 1 or times.shape[0] < 1:
            raise ValueError('trajectory needs at least one keyframe')
        if poses.shape != (times.shape[0], 4, 4):
            raise ValueError('poses must be (N,4,4) matching times')
        if not np.isfinite(times).all():
            raise ValueError('times must be finite')
        if times.shape[0] > 1 and not (np.diff(times) > 0).all():
            raise ValueError('times must be strictly increasing (no duplicates)')
        if bounds not in ('strict', 'hold'):
            raise ValueError("bounds must be 'strict' or 'hold'")
        self.poses = np.stack([_check_pose(p) for p in poses])
        self.times = times
        self.bounds = bounds
        self._quats = np.stack([_rot_to_quat(p[:3, :3]) for p in self.poses])
        # canonical hemisphere continuity so slerp never flips mid-trajectory
        for i in range(1, len(self._quats)):
            if np.dot(self._quats[i - 1], self._quats[i]) < 0.0:
                self._quats[i] = -self._quats[i]

    @property
    def t_start(self):
        return float(self.times[0])

    @property
    def t_end(self):
        return float(self.times[-1])

    def evaluate(self, t):
        t = float(t)
        if not np.isfinite(t):
            raise ValueError('evaluation time must be finite')
        if len(self.times) == 1:
            if self.bounds == 'strict' and abs(t - self.times[0]) > _TIME_TOL:
                raise ValueError('static trajectory evaluated away from its keyframe')
            return self.poses[0].copy()
        if t < self.times[0] - _TIME_TOL or t > self.times[-1] + _TIME_TOL:
            if self.bounds == 'strict':
                raise ValueError('time %g outside trajectory window [%g, %g]'
                                 % (t, self.times[0], self.times[-1]))
            return self.poses[0].copy() if t < self.times[0] else self.poses[-1].copy()
        i = int(np.searchsorted(self.times, t, side='right')) - 1
        i = min(max(i, 0), len(self.times) - 2)
        t0, t1 = self.times[i], self.times[i + 1]
        u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        u = min(max(u, 0.0), 1.0)
        pos = (1.0 - u) * self.poses[i][:3, 3] + u * self.poses[i + 1][:3, 3]
        rot = _quat_to_rot(_slerp(self._quats[i], self._quats[i + 1], u))
        out = np.eye(4)
        out[:3, :3] = rot
        out[:3, 3] = pos
        return out


@dataclass
class _Entry:
    trajectory: PoseTrajectory
    current_pose: np.ndarray  # single source of truth for render AND collision


class PoseTrackRegistry:
    """Owns trajectories, current poses, revision/generation and sim time."""

    def __init__(self):
        self._objects = {}
        self._revision = 0
        self._generation = str(uuid4())
        self._sim_time = None

    # -- cache-token contract -------------------------------------------------
    @property
    def revision(self):
        return self._revision

    @property
    def generation(self):
        return self._generation

    @property
    def cache_token(self):
        return f'{self._generation}:{self._revision}'

    @property
    def sim_time(self):
        return self._sim_time

    # -- edit events (structural; bump revision -> downstream history reset) --
    def insert(self, object_id, trajectory):
        if object_id in self._objects:
            raise KeyError('duplicate managed object: ' + object_id)
        self._objects[object_id] = _Entry(trajectory, trajectory.evaluate(
            trajectory.t_start if self._sim_time is None else self._clamp_time(self._sim_time, trajectory)))
        self._revision += 1

    def remove(self, object_id):
        if object_id not in self._objects:
            raise KeyError('unknown managed object: ' + object_id)
        del self._objects[object_id]
        self._revision += 1

    def set_trajectory(self, object_id, trajectory):
        """Replace the allowed motion of an existing object (structural edit)."""
        if object_id not in self._objects:
            raise KeyError('unknown managed object: ' + object_id)
        self._objects[object_id] = _Entry(trajectory, trajectory.evaluate(
            trajectory.t_start if self._sim_time is None else self._clamp_time(self._sim_time, trajectory)))
        self._revision += 1

    @staticmethod
    def _clamp_time(t, trajectory):
        if trajectory.bounds == 'hold':
            return min(max(t, trajectory.t_start), trajectory.t_end)
        if t < trajectory.t_start - _TIME_TOL or t > trajectory.t_end + _TIME_TOL:
            # A strict trajectory that does not cover the current sim time is a
            # contract violation: refuse rather than silently fabricating a pose.
            raise ValueError('sim time %g outside trajectory window [%g, %g]'
                             % (t, trajectory.t_start, trajectory.t_end))
        return t

    # -- normal advancement (never bumps revision) ----------------------------
    def advance(self, t):
        t = float(t)
        if not np.isfinite(t):
            raise ValueError('simulation time must be finite')
        if self._sim_time is not None and t < self._sim_time - _TIME_TOL:
            raise ValueError('simulation time moved backwards without reset')
        # Evaluate everything first: a rejected evaluation must leave the
        # registry fully unchanged (time AND poses), not half-advanced.
        updated = {oid: entry.trajectory.evaluate(t)
                   for oid, entry in self._objects.items()}
        self._sim_time = t
        for oid, entry in self._objects.items():
            entry.current_pose = updated[oid]
        return {oid: entry.current_pose for oid, entry in self._objects.items()}

    # -- pose access: render and collision share one array --------------------
    def render_pose(self, object_id):
        return self._objects[object_id].current_pose

    def collision_pose(self, object_id):
        return self._objects[object_id].current_pose

    def object_ids(self):
        return tuple(self._objects)

    # -- reset / scene switch: full state cleanup, new generation -------------
    def reset(self):
        self._objects = {}
        self._revision = 0
        self._generation = str(uuid4())
        self._sim_time = None
