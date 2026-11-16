"""Per-frame pose track for animated managed assets (diagnostic design lib).

Separates two semantics that must never be conflated:
- ANIMATION SAMPLING: evaluating a committed trajectory at the current
  simulation time. No scene edit, no edit_revision bump, UniAD
  history/BEV cache untouched (model_cache_token unchanged).
- EDIT EVENTS: insert / remove / set_static_pose / replace_trajectory.
  Each bumps edit_revision, which must propagate to the scene token so the
  UniAD temporal state resets (existing reset chain preserved, NOT removed).

Conventions: poses are 4x4 rigid proper matrices in the source world
(first-Waymo-ego, x fwd / y left / z up), fps from the scene manifest,
quaternions wxyz. Sampling outside the committed window is an error, never a
clamp (matches waymo_adapter.frame_index semantics).
"""
from dataclasses import dataclass, field
import math
import numpy as np


def _check_pose(P):
    P = np.asarray(P, dtype=float)
    if P.shape != (4, 4) or not np.isfinite(P).all():
        raise ValueError('pose must be a finite 4x4')
    R = P[:3, :3]
    if not np.allclose(P[3], [0, 0, 0, 1], atol=1e-9) or \
       not np.allclose(R.T@R, np.eye(3), atol=1e-6) or \
       not np.isclose(np.linalg.det(R), 1., atol=1e-6):
        raise ValueError('pose must be rigid and right-handed')
    return P


def _quat_from_matrix(R):
    m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
    tr = m00+m11+m22
    if tr > 0:
        s = math.sqrt(tr+1.)*2
        return np.array([0.25*s, (m21-m12)/s, (m02-m20)/s, (m10-m01)/s])
    i = int(np.argmax([m00, m11, m22]))
    if i == 0:
        s = math.sqrt(1.+m00-m11-m22)*2
        q = [(m21-m12)/s, 0.25*s, (m01+m10)/s, (m02+m20)/s]
    elif i == 1:
        s = math.sqrt(1.+m11-m00-m22)*2
        q = [(m02-m20)/s, (m01+m10)/s, 0.25*s, (m12+m21)/s]
    else:
        s = math.sqrt(1.+m22-m00-m11)*2
        q = [(m10-m01)/s, (m02+m20)/s, (m12+m21)/s, 0.25*s]
    return np.array(q)/np.linalg.norm(q)


def _matrix_from_quat(q):
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def slerp(q0, q1, u):
    q0 = q0/np.linalg.norm(q0); q1 = q1/np.linalg.norm(q1)
    dot = float(q0@q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0+u*(q1-q0)
        return q/np.linalg.norm(q)
    th = math.acos(min(1., dot))
    return (math.sin((1-u)*th)*q0+math.sin(u*th)*q1)/math.sin(th)


@dataclass(frozen=True)
class PoseTrack:
    """Immutable committed trajectory snapshot; sampling is not an edit."""
    object_id: str
    fps: float
    poses: tuple          # tuple of (4,4) ndarrays, length = frames
    visible: tuple        # tuple of bool, length = frames
    edit_revision: int    # revision at which this track was committed
    interpolation: str = 'linear_slerp'

    @classmethod
    def create(cls, object_id, fps, poses, visible, edit_revision):
        if not object_id or not isinstance(object_id, str):
            raise ValueError('object_id required')
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('invalid fps')
        poses = tuple(_check_pose(p) for p in poses)
        visible = tuple(bool(v) for v in visible)
        if len(poses) != len(visible) or not poses:
            raise ValueError('poses/visible must be equal non-zero length')
        return cls(object_id, float(fps), poses, visible, int(edit_revision))

    @property
    def frames(self):
        return len(self.poses)

    @property
    def duration_s(self):
        return (self.frames-1)/self.fps

    def frame_at(self, t):
        if not math.isfinite(t) or t < 0:
            raise ValueError('invalid sample time')
        pos = t*self.fps
        if pos > self.frames-1+1e-9:
            raise ValueError('track time exhausted; extend by edit event, never clamp')
        return pos

    def sample(self, t):
        """Pose at sim time t. Not visible -> None (asset absent, not frozen)."""
        pos = self.frame_at(t)
        i = int(math.floor(pos+1e-9))
        j = min(i+1, self.frames-1)
        u = min(max(pos-i, 0.), 1.)
        if not (self.visible[i] or self.visible[j]):
            return None
        if i == j or u < 1e-12:
            return self.poses[i].copy() if self.visible[i] else None
        P0, P1 = self.poses[i], self.poses[j]
        q = slerp(_quat_from_matrix(P0[:3, :3]), _quat_from_matrix(P1[:3, :3]), u)
        out = np.eye(4)
        out[:3, :3] = _matrix_from_quat(q)
        out[:3, 3] = (1-u)*P0[:3, 3]+u*P1[:3, 3]
        return out

    def to_dict(self):
        return dict(object_id=self.object_id, fps=self.fps,
                    poses=[p.tolist() for p in self.poses], visible=list(self.visible),
                    edit_revision=self.edit_revision, interpolation=self.interpolation)

    @classmethod
    def from_dict(cls, d):
        return cls.create(d['object_id'], d['fps'],
                          [np.asarray(p, dtype=float) for p in d['poses']],
                          d['visible'], d['edit_revision'])


# --- edit-event vs animation bookkeeping ---------------------------------

@dataclass
class SceneEditState:
    """Minimal model of the revision/token contract (mirrors scene_api semantics).

    revision bumps ONLY on edit events; animation sampling does not touch it.
    model_cache_token = f'{generation}:{revision}' — any bump invalidates the
    UniAD temporal cache through the existing scene_token chain (preserved).
    """
    generation: str
    revision: int = 0
    tracks: dict = field(default_factory=dict)

    def _edit(self):
        self.revision += 1

    @property
    def model_cache_token(self):
        return f'{self.generation}:{self.revision}'

    def insert_track(self, track: PoseTrack):
        if track.object_id in self.tracks:
            raise ValueError('object exists; use replace_track')
        if track.edit_revision != self.revision+1:
            raise ValueError('track must commit at the next revision')
        self._edit()
        self.tracks[track.object_id] = track
        return self.revision

    def remove(self, object_id):
        if object_id not in self.tracks:
            raise KeyError(object_id)
        del self.tracks[object_id]
        self._edit()
        return self.revision

    def replace_track(self, track: PoseTrack):
        if track.object_id not in self.tracks:
            raise KeyError('replace requires an existing object')
        if track.edit_revision != self.revision+1:
            raise ValueError('track must commit at the next revision')
        self._edit()
        self.tracks[track.object_id] = track
        return self.revision

    def animate(self, object_id, t):
        """Sample the committed track; never bumps the revision."""
        track = self.tracks[object_id]
        return track.sample(t)
