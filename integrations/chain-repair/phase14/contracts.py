"""CPU-only contracts shared by cache building, training and DAgger."""

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

pass  # Path is provided by the portable boundary above
import hashlib
import json
import os
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[5]
TRAIN_SCENES = {3, 23, 31, 233, 551}
EVAL_SCENES = {172, 552, 621}
STEPS, DT = 6, .5
MODEL_BASIS = np.array([[0., -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
INPUT_NAMES = ('bev_embed', 'bev_pos', 'sdc_traj_query', 'sdc_track_query', 'command')
COORDINATES = 'future_virtual_lidar_origin_in_current_virtual_lidar_x_right_y_forward'


def training_scene(value):
    scene = int(value)
    if isinstance(value, bool) or scene not in TRAIN_SCENES:
        raise ValueError('Training accepts only 3/23/31/233/551; evaluation scenes are forbidden')
    return scene


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temp, path)


def atomic_pickle(path, value):
    import pickle
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f: pickle.dump(value, f, protocol=4)
    os.replace(temp, path)


def load_source(path):
    import yaml
    source = _hugsim_expand(json.loads(Path(path).read_text())); scene = training_scene(source['scene'])
    config = yaml.safe_load(Path(source['source_config']).read_text())
    if int(config['data']['scene_idx']) != scene:
        raise ValueError('Source scene identity differs from reconstruction configuration')
    if config['data']['pixel_source']['cameras'] != [0, 1, 2]:
        raise ValueError('Requires a reconstruction trained with three front cameras')
    if int(config['data']['start_timestep']) != int(source.get('start_timestep', 0)):
        raise ValueError('Source start frame differs from reconstruction configuration')
    if yaml.safe_load((Path(source['inputs'])/'config.yaml').read_text()) != config:
        raise ValueError('Input configuration differs from checkpoint configuration')
    return source


def poses_from_source(source):
    start = int(source.get('start_timestep', 0))
    poses = np.stack([np.loadtxt(Path(source['inputs'])/'ego_pose'/f'{start+i:03d}.txt')
                      for i in range(int(source['frames']))])
    if poses.shape[1:] != (4, 4) or not np.isfinite(poses).all(): raise ValueError('Invalid source poses')
    if not np.allclose(poses[:, 3], [0, 0, 0, 1]): raise ValueError('Non-rigid pose')
    if not np.allclose(poses[:, :3, :3].transpose(0, 2, 1)@poses[:, :3, :3], np.eye(3), atol=1e-5):
        raise ValueError('Non-orthogonal source pose')
    return np.linalg.inv(poses[0])[None]@poses


def horizon_frames(fps):
    if not np.isfinite(fps) or fps <= 0 or not np.isclose(DT*fps, round(DT*fps)):
        raise ValueError('Planning timestamps must align with source frames')
    return int(round(STEPS*DT*fps))


def expert_label(relative, frame, fps, mount, *, ego_pose=None):
    """Native UniAD target: future LiDAR origins in today's LiDAR frame.

    ego_pose may be a policy-visited source-world pose for DAgger relabelling.
    Source-end truncation is rejected, never replaced with a repeated last pose.
    """
    horizon = horizon_frames(fps)
    if frame < 0 or frame+horizon >= len(relative): raise ValueError('Incomplete expert horizon')
    ego = relative[frame] if ego_pose is None else np.asarray(ego_pose, dtype=float)
    current = MODEL_BASIS@ego@MODEL_BASIS.T@mount
    stride = horizon//STEPS
    futures = MODEL_BASIS[None]@relative[frame+stride:frame+horizon+1:stride]@MODEL_BASIS.T@mount
    local = np.linalg.inv(current)[None]@futures
    label = local[:, :2, 3].astype(np.float32)
    if label.shape != (STEPS, 2) or not np.isfinite(label).all(): raise ValueError('Invalid expert label')
    return label


def validate_split(records):
    """Disallow evaluation scenes and overlapping train/val source windows."""
    train, val = [], []
    for record in records:
        training_scene(record['scene'])
        if record['split'] not in ('train', 'val'): raise ValueError('Unknown split')
        (train if record['split']=='train' else val).append(record)
    if not train or not val: raise ValueError('Both train and val samples are required')
    for a in train:
        for b in val:
            if int(a['scene'])==int(b['scene']) and max(a['frame'],b['frame'])<=min(a['label_end_frame'],b['label_end_frame']):
                raise ValueError('Training and validation source horizons overlap')
    return train, val


def motion_coverage(records):
    """Minimal data admission check; passing it is not a generalization proof."""
    counts={s:sum(r.get('split')==s and r.get('speed_mps',-1)>=1. for r in records)
            for s in ('train','val')}
    admitted=all(counts[s]>0 for s in counts)
    return dict(pilot_training_admitted=admitted,moving_speed_threshold_mps=1.,
        moving_samples=counts,reason=None if admitted else
        'Training and validation each need moving examples; stationary-only ADE can reward stopping')
