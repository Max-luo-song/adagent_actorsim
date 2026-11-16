"""Actual vendored SMPL template forward with cache present and absent; CPU."""

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

import os,pickle,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import numpy as np

class CanonicalSMPL(unittest.TestCase):
    def test_template_bypasses_pickle_and_matches_both_cache_states(self):
        import torch
        from contracts import P26
        from framework import bootstrap
        from smpl_runtime import assets,canonical_layer
        bundle=P26/'corridor_scene_v4';bootstrap(bundle)
        import models.human_body as human
        resources=assets();original=human.SMPLLayer;states=[]
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'SMPL_NEUTRAL.pkl';cache.write_bytes(b'pickle must never be read')
            for path in [cache,Path(directory)/'missing.pkl']:
                with patch.dict(os.environ,SMPL_NEUTRAL_NPZ=resources['npz']['path']),patch.object(pickle,'load',side_effect=AssertionError('pickle path used')):
                    with canonical_layer(resources,bundle):
                        template=human.SMPLTemplate(str(path),1,torch.zeros(1,10),'da_pose',use_voxel_deformer=False)
                        vertices,faces=template.get_init_vf()
                        self.assertEqual(tuple(vertices.shape),(1,6890,3));self.assertEqual(tuple(faces.shape),(13776,3))
                        self.assertTrue(torch.isfinite(vertices).all());states.append({k:v.clone() for k,v in template.state_dict().items()})
                self.assertIs(human.SMPLLayer,original)
            self.assertEqual(set(states[0]),set(states[1]))
            self.assertTrue(all(torch.equal(states[0][k],states[1][k]) for k in states[0]))
            with self.assertRaisesRegex(RuntimeError,'initialization fault'):
                with canonical_layer(resources,bundle):raise RuntimeError('initialization fault')
            self.assertIs(human.SMPLLayer,original)

if __name__=='__main__':unittest.main()
