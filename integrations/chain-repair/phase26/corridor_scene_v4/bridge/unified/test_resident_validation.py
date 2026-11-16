
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

import copy
pass  # Path is provided by the portable boundary above
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from run_resident_validation import compare_perception, preflight, wait_json


class ValidationTests(unittest.TestCase):
    def test_identical_outputs_compare_all_required_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.npz';b=Path(folder)/'b.npz'
            data={name:np.arange(6,dtype=float) for name in (
                'trajectory','occupancy','candidate_boxes','candidate_scores','candidate_labels')}
            np.savez(a,**data);np.savez(b,**data)
            self.assertTrue(all(v==0 for v in compare_perception(a,b).values()))
            data['candidate_scores']=data['candidate_scores']+.01
            np.savez(b,**data)
            self.assertGreater(compare_perception(a,b)['candidate_scores'],1e-4)
            data['trajectory']=np.ones(2);np.savez(b,**data)
            with self.assertRaisesRegex(RuntimeError,'shape/key mismatch'):compare_perception(a,b)
    def test_nan_output_is_never_reported_as_matching(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.npz';b=Path(folder)/'b.npz'
            data={name:np.zeros(1) for name in (
                'trajectory','occupancy','candidate_boxes','candidate_scores','candidate_labels')}
            np.savez(a,**data);data['trajectory'][0]=np.nan;np.savez(b,**data)
            row=compare_perception(a,b)
            self.assertFalse(all(np.isfinite(v) and v<=1e-4 for v in row.values()))
    def test_same_scene_or_same_model_is_not_a_scene_switch(self):
        from scene_edit_session import saf
        for scenes,models in [([552,552],['a','b']),([552,551],['a','a'])]:
            specs=[SimpleNamespace(source=dict(scene=s,checkpoint=m)) for s,m in zip(scenes,models)]
            with self.subTest(scenes=scenes),patch.object(saf.SceneSpec,'read',side_effect=specs):
                with self.assertRaisesRegex(ValueError,'Different scene IDs'):preflight('a','b')
    def test_early_worker_exit_has_a_concrete_error(self):
        with tempfile.TemporaryDirectory() as folder:
            process=SimpleNamespace(poll=lambda:7,returncode=7)
            with self.assertRaisesRegex(RuntimeError,'UniAD exited.*7'):
                wait_json(Path(folder)/'missing.json',process,1)


if __name__=='__main__':unittest.main()
