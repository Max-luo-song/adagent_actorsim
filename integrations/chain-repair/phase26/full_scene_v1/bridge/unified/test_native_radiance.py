"""Editing must rotate SH by delta without changing baseline or geometry."""

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
import sys
import unittest
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import test_native_rotation as fixtures
command = fixtures.command

for parent in Path(__file__).resolve().parents:
    if (parent/'waymo23-native-import/asset_radiance.py').is_file():
        sys.path.insert(0,str(parent/'waymo23-native-import'));break
from asset_radiance import evaluator
from native_radiance import rotated_rest


class RadianceTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.RotationTests();fixture.setUp()
        self.fixture=fixture
        self.editor,self.model,self.backend,self.session=fixture.editor,fixture.model,fixture.backend,fixture.session
        self.model.point_ids=torch.tensor([[0],[1],[1],[0]])
        generator=torch.Generator().manual_seed(1)
        self.model._features_rest=torch.randn(4,15,3,generator=generator,dtype=torch.float64)*.2
        self.model._features_dc=torch.randn(4,3,generator=generator,dtype=torch.float64)
        self.rest=self.model._features_rest.clone();self.dc=self.model._features_dc.clone()
        self.q=self.model.instances_quats.clone()

    def test_gpu_probe_independent_sh_oracle_accepts_compensated_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=self.fixture.probe(Path(tmp))
        self.assertTrue(report['passed'])
        self.assertTrue(report['sh_edit_direction_consistent'])
        self.assertLess(report['sh_edit_direction_max_abs'],2e-5)

    def test_radiance_matches_rigidly_rotated_camera_and_object(self):
        self.editor.edit(command())
        c0=torch.cat((self.dc[:,None],self.rest),dim=1).numpy().transpose(0,2,1)
        c1=torch.cat((self.dc[:,None],self.model._features_rest),dim=1).numpy().transpose(0,2,1)
        rng=np.random.default_rng(2);directions=rng.normal(size=(4,3));directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
        delta=Rotation.from_euler('z',15,degrees=True).as_matrix()
        # Rotation of directions with object/camera cancels rotation of SH.
        actual=evaluator()(3,c1[1:3],directions[1:3]@delta.T)
        expected=evaluator()(3,c0[1:3],directions[1:3])
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        self.assertTrue(torch.equal(self.model._features_rest[[0,3]],self.rest[[0,3]]))
        self.assertTrue(torch.equal(self.model._features_dc,self.dc))

    def test_composition_matches_total_delta_and_exact_reset(self):
        self.editor.edit(command(30));self.editor.edit(command(-15))
        expected=rotated_rest(self.rest[1:3],3,15)
        torch.testing.assert_close(self.model._features_rest[1:3],expected,atol=1e-12,rtol=0)
        self.editor.restore_native()
        self.assertTrue(torch.equal(self.model._features_rest,self.rest))
        self.assertTrue(torch.equal(self.model.instances_quats,self.q))

    def test_history_failure_restores_sh_and_quaternions(self):
        def failed():raise RuntimeError('history failed')
        self.backend.clear_observation_history=failed
        with self.assertRaises(RuntimeError):self.editor.edit(command())
        self.assertTrue(torch.equal(self.model._features_rest,self.rest))
        self.assertTrue(torch.equal(self.model.instances_quats,self.q))
        self.assertFalse(self.editor.original_radiance);self.assertEqual(self.session.revision,0)

    def test_zero_delta_and_inverse_preserve_all_degrees(self):
        for degree in (0,1,2,3):
            rest=self.rest[:,:((degree+1)**2-1)]
            torch.testing.assert_close(rotated_rest(rest,degree,0),rest,atol=1e-12,rtol=0)
            actual=rotated_rest(rotated_rest(rest,degree,27),degree,-27)
            torch.testing.assert_close(actual,rest,atol=1e-12,rtol=0)

    def test_sh_parameter_replacement_prevents_partial_reset(self):
        self.editor.edit(command());q=self.model.instances_quats.clone()
        self.model._features_rest=self.model._features_rest.clone()
        with self.assertRaisesRegex(RuntimeError,'SH parameter identity'):self.editor.restore_native()
        self.assertTrue(torch.equal(self.model.instances_quats,q))

    def test_invalid_sh_is_rejected_before_any_geometry_changes(self):
        self.model._features_rest[1,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):self.editor.edit(command())
        self.assertTrue(torch.equal(self.model.instances_quats,self.q));self.assertEqual(self.session.revision,0)

if __name__=='__main__':unittest.main()
