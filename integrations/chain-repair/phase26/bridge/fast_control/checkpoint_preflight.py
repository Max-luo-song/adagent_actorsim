"""Fail before GPU loading if tracking driver objects cannot be checkpointed."""

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

import argparse
import json
pass  # Path is provided by the portable boundary above
import tempfile
from types import SimpleNamespace

import numpy as np

import tracking_common as tc
import planning_guard_tracking as pgt
from closed_loop_resume import load_checkpoint, save_checkpoint
from depth_sensor_map import DepthObstacleMemory


def verify():
    with tempfile.TemporaryDirectory(prefix='phase26-checkpoint-') as folder:
        base = Path(folder)
        path = base / 'outputs/driver.pkl'
        path.parent.mkdir()
        memory = pgt.ObstacleMemory(np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]))
        memory.tracks = [dict(id=1, box=np.arange(9, dtype=float))]
        depth = DepthObstacleMemory(tc.load_vehicle_config())
        depth.cells[(1,2,3)] = dict(point=np.array([.5,1.,1.5]))
        save_checkpoint(path, implementation={'preflight':'tracking'}, source_result=path.parent,
            next_index=10, token='preflight', motion=SimpleNamespace(speed=2.7),
            warm=np.arange(12).reshape(6,2), controller=pgt.RecoveryController(),
            obstacle_memory=memory, depth_memory=depth,
            report={'experiment_signature':'preflight'}, minimum_gap=1.3)
        result = load_checkpoint(path, {'preflight':'tracking'}, base, 'preflight')
        checks = dict(obstacle_class=type(result['obstacle_memory']) is pgt.ObstacleMemory,
            controller_class=type(result['controller']) is pgt.RecoveryController,
            track=bool(np.array_equal(result['obstacle_memory'].tracks[0]['box'], np.arange(9))),
            depth=bool(np.array_equal(result['depth_memory'].cells[(1,2,3)]['point'], [.5,1.,1.5])),
            observed_ground=result['obstacle_memory'].ground_tree.query([0.,0.])[0] == 0,
            warm=bool(np.array_equal(result['warm'], np.arange(12).reshape(6,2))),
            state=result['motion']['speed'] == 2.7 and result['next_index'] == 10)
        if not all(checks.values()):
            raise RuntimeError('tracking driver checkpoint preflight failed')
        return dict(passed=True, checks=checks, gpu_used=False, formal_evaluation_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2))
    print('Tracking driver checkpoint preflight passed (CPU save/load).', flush=True)
