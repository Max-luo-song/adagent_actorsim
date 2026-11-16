
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

import copy, sys, unittest
pass  # Path is provided by the portable boundary above
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
sys.path[:0] = [str(HERE), str(ROOT / 'waymo23-native-import'), str(ROOT / 'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from evidence import EvidenceHistory as QueryLifecycle
CFG = dict(max_gap_seconds=0.75, confirmation_seconds=0.2, confirmation_observations=3, obstacle_margin_m=0.5, caution_speed_mps=2.0)

def fixture(visible=False, surface=False, occluded=False):
    l2c = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    intr = dict(fovx=float(2 * np.arctan(100 / 60)), fovy=float(2 * np.arctan(100 / 60)), cx=50.0, cy=50.0, W=100, H=100)
    info = dict(timestamp=0.0, scene_token='test', explicit_l2g_r_mat=np.eye(3), explicit_l2g_t=np.zeros(3), depth_encoding='expected_camera_z_v1', cam_params={'cam': dict(l2c=l2c, intrinsic=intr)})
    box = np.array([0.0, 10.0, -1.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0]) if visible else np.array([30.0, 1.0, -1.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.0])
    depth = np.full((100, 100), 10.0, dtype=np.float32)
    valid = np.ones((100, 100), bool)
    obs = dict(depth={'cam': depth}, surface_depth={'cam': depth}, surface_valid={'cam': valid})
    return (obs, info, box)

class Tests(unittest.TestCase):

    def replay(self, visible=False, velocity=0.0, missing=False):
        (obs, info, box) = fixture(visible)
        gate = QueryLifecycle(CFG)
        results = []
        if missing:
            obs = {}
        for i in range(4):
            b = box.copy()
            b[0] += i * 0.4
            b[7] = velocity
            info = dict(info, timestamp=i * 0.1)
            results.append(gate.observe(b[None], [731], [0], obs, info))
        return (results, gate, obs, info, box)

    def test_inconsistent_unobserved_history_flags_untrusted_geometry(self):
        (result, *_) = self.replay()
        self.assertEqual(result[0]['untrusted_geometry'], [])
        self.assertEqual(result[-1]['untrusted_geometry'], [731])
        self.assertFalse(result[-1]['current_detections_filtered'])
        self.assertFalse(result[-1]['free_space_asserted'])

    def test_motion_explained_by_model_velocity_is_retained(self):
        (result, *_) = self.replay(velocity=4.0)
        self.assertEqual(result[-1]['untrusted_geometry'], [])

    def test_visible_object_and_missing_depth_are_retained(self):
        for (visible, missing) in [(True, False), (False, True)]:
            (result, *_) = self.replay(visible=visible, missing=missing)
            self.assertEqual(result[-1]['untrusted_geometry'], [])

    def test_previously_observed_object_is_protected_after_leaving_view(self):
        (result, gate, obs, info, box) = self.replay(visible=True)
        for i in range(4, 10):
            box[0] += 8
            info = dict(info, timestamp=i * 0.1)
            r = gate.observe(box[None], [731], [0], obs, info)
            self.assertEqual(r['untrusted_geometry'], [])

    def test_scene_and_nonmonotonic_time_reset_history(self):
        (_, gate, obs, info, box) = self.replay()
        info = dict(info, timestamp=0.0, scene_token='next')
        r = gate.observe(box[None], [731], [0], obs, info)
        self.assertTrue(r['reset'])
        self.assertEqual(r['untrusted_geometry'], [])

    def test_duplicate_ids_do_not_manufacture_temporal_evidence(self):
        (obs, info, box) = fixture()
        g = QueryLifecycle(CFG)
        with self.assertRaises(ValueError):
            g.observe(np.stack([box, box]), [12, 12], [0, 0], obs, info)

    def test_partial_view_surface_or_occluder_protects_unknown_center(self):
        for observed_depth in (6.9, 1.0):
            (obs, info, box) = fixture()
            box[:2] = [9.0, 5.0]
            obs['depth']['cam'][:] = observed_depth
            gate = QueryLifecycle(CFG)
            for i in range(4):
                b = box.copy()
                b[0] += i * 0.3
                result = gate.observe(b[None], [731], [0], obs, dict(info, timestamp=i * 0.1))
                self.assertEqual(result['untrusted_geometry'], [])

    def test_actual_worker_help_in_clean_process_without_inherited_import_paths(self):
        import os, subprocess, tempfile
        env = dict(os.environ, BASE=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for name in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH'):
            env.pop(name, None)
        python = ROOT / 'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([str(python), '-B', str(HERE / 'worker.py'), '--help'], cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('--diagnostic-closed-loop', result.stdout)
if __name__ == '__main__':
    unittest.main()
