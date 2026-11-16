"""Coordinate/clock contract for frozen Waymo scene integration (no training)."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import json
pass  # Path is provided by the portable boundary above
import numpy as np

# SR scene coordinates: first Waymo ego frame. HUGSIM: right/down/forward.
SR_TO_HUGSIM = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                         [1., 0., 0., 0.], [0., 0., 0., 1.]])


def rigid(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 4x4 pose')
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('Expected a rigid right-handed pose')
    return matrix


def sr_ego_pose(hugsim_ego_pose):
    """Convert both ego-local axes and world axes; camera poses use only left conversion."""
    return SR_TO_HUGSIM.T @ rigid(hugsim_ego_pose) @ SR_TO_HUGSIM


def frame_index(timestamp, fps, frames):
    if not np.isfinite(timestamp) or timestamp < 0 or not np.isfinite(fps) or fps <= 0 or frames < 1:
        raise ValueError('Invalid scene clock')
    frame = int(np.floor(timestamp * fps + 1e-8))
    if frame >= frames:
        raise ValueError('Scene time exhausted; reset or terminate instead of clamping')
    return frame


def scene_manifest(source_report):
    path = Path(source_report).resolve()
    source = _hugsim_expand(json.loads(path.read_text()))
    for key in ('checkpoint', 'source_config', 'inputs'):
        if not Path(source[key]).exists():
            raise FileNotFoundError(source[key])
    cameras = source.get('trained_cameras', source.get('camera_ids'))
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError('Missing or duplicate trained camera IDs')
    names = {0: 'CAM_FRONT', 1: 'CAM_FRONT_LEFT', 2: 'CAM_FRONT_RIGHT', 3: 'CAM_LEFT', 4: 'CAM_RIGHT'}
    return dict(schema_version=1, dataset='waymo', scene=source['scene'],
                source_report=str(path), checkpoint=source['checkpoint'],
                config=source['source_config'], calibration_root=source['inputs'],
                trained_cameras=cameras, camera_names={str(c): names[c] for c in cameras},
                output_camera_order=source.get('camera_order', [names[c] for c in cameras]),
                frames=source['frames'], start_timestep=source['start_timestep'],
                source_world='first_waymo_ego', hugsim_world='right_down_forward',
                sr_to_hugsim=SR_TO_HUGSIM.tolist(), training_enabled=False,
                checkpoint_compatibility='pbr_code_ported_probe_required',
                main_renderer_verified=False, integration_complete=False)
