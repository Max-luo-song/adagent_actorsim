"""Real driver checkpoint save/load under the tracking module substitutions."""

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
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tracking_common as tc
import planning_guard_tracking as pgt
from depth_sensor_map import DepthObstacleMemory
from closed_loop_resume import load_checkpoint, save_checkpoint


def write_checkpoint(base):
    path = base / 'outputs/driver.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    memory = pgt.ObstacleMemory(ground_points_world=np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]))
    memory.tracks = [dict(id=17, box=np.arange(9, dtype=float))]
    memory.time, memory.scene, memory.next_id = 8.9, 'scene552', 18
    depth = DepthObstacleMemory(tc.load_vehicle_config())
    depth.cells[(1,2,3)] = dict(point=np.array([.5,1.,1.5]), count=2, seen=8.9)
    controller = pgt.RecoveryController()
    controller.clear_observations = 3
    save_checkpoint(path, implementation={'control':'unchanged'}, source_result=base/'outputs/source',
        next_index=10, token='same-scene-token', motion=SimpleNamespace(speed=2.676, timestamp=9.),
        warm=np.arange(12, dtype=float).reshape(6,2), controller=controller,
        obstacle_memory=memory, depth_memory=depth,
        report={'experiment_signature':'signature', 'trace':[{'step':i} for i in range(10)]}, minimum_gap=1.3)
    return path


def read_checkpoint(base):
    data = load_checkpoint(base/'outputs/driver.pkl', {'control':'unchanged'}, base, 'signature')
    assert type(data['obstacle_memory']) is pgt.ObstacleMemory
    assert type(data['controller']) is pgt.RecoveryController
    assert data['next_index'] == 10 and data['token'] == 'same-scene-token'
    assert data['motion'] == dict(speed=2.676, timestamp=9.)
    assert data['minimum_gap'] == 1.3
    assert data['controller'].clear_observations == 3
    memory = data['obstacle_memory']
    assert memory.time == 8.9 and memory.scene == 'scene552' and memory.next_id == 18
    assert np.array_equal(memory.tracks[0]['box'], np.arange(9))
    assert memory.ground_tree.query([0.,0.])[0] == 0
    assert np.array_equal(data['warm'], np.arange(12).reshape(6,2))
    assert np.array_equal(data['depth_memory'].cells[(1,2,3)]['point'], [.5,1.,1.5])
    assert len(data['report']['trace']) == 10


class CheckpointTests(unittest.TestCase):
    def test_reviewed_modules_registered_with_their_actual_objects(self):
        self.assertIs(sys.modules['planning_guard_reviewed'], pgt._real)
        self.assertIs(sys.modules['closed172_common_reviewed'], tc._real)
        self.assertIs(sys.modules[pgt.ObstacleMemory.__module__].ObstacleMemory, pgt.ObstacleMemory)

    def test_real_driver_payload_roundtrip_and_identity_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_checkpoint(base)
            read_checkpoint(base)
            with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
                load_checkpoint(base/'outputs/driver.pkl', {'control':'changed'}, base, 'signature')
            with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
                load_checkpoint(base/'outputs/driver.pkl', {'control':'unchanged'}, base, 'different-signature')

    def test_driver_payload_loads_in_fresh_tracking_process(self):
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--write', folder], check=True)
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--read', folder], check=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('--write', '--read'):
        (write_checkpoint if sys.argv[1] == '--write' else read_checkpoint)(Path(sys.argv[2]))
    else:
        unittest.main()
