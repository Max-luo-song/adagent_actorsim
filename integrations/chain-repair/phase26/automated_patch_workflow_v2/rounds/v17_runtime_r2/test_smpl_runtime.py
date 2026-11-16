
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

import sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import torch
R=Path(_hugsim_expand('${HUGSIM_ROOT}'));C=R/'integrations/light-execution-horizon-v17-r2'
sys.path[:0]=[str(C),str(R/'integrations/light-cutin-diagnostic')]
from common import configure
configure();sys.path.insert(0,str(C))
import smpl_runtime as smpl
class SMPLRuntimeTests(unittest.TestCase):
 def test_actual_template_constructs_despite_existing_invalid_pickle(self):
  r=smpl.preflight();self.assertTrue(r['passed']);self.assertTrue(r['existing_pickle_bypassed']);self.assertEqual(r['vertices'],[1,6890,3])
 def test_canonical_matches_old_verified_npz_fallback_exactly(self):
  module=smpl.body_module()
  with tempfile.TemporaryDirectory() as d:
   old=module.SMPLTemplate(str(Path(d)/'missing.pkl'),1,torch.zeros(1,10),'da_pose',use_voxel_deformer=False)
   p=Path(d)/'existing.pkl';p.write_bytes(b'not a pickle')
   with smpl.canonical_layer():new=module.SMPLTemplate(str(p),1,torch.zeros(1,10),'da_pose',use_voxel_deformer=False)
   self.assertEqual(set(old.state_dict()),set(new.state_dict()))
   for k,t in old.state_dict().items():self.assertTrue(torch.equal(t,new.state_dict()[k]),k)
 def test_corrupt_resource_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.npz';p.write_bytes(b'bad')
   with patch.object(smpl,'ASSETS',{str(p):'wrong'}):
    with self.assertRaisesRegex(ValueError,'asset changed'):smpl.verified_data()
 def test_constructor_is_restored_on_exception(self):
  module=smpl.body_module();old=module.SMPLLayer
  with self.assertRaisesRegex(RuntimeError,'fixture'):
   with smpl.canonical_layer():
    self.assertIsNot(module.SMPLLayer,old);raise RuntimeError('fixture')
  self.assertIs(module.SMPLLayer,old)
 def test_real_backend_constructor_hook_uses_canonical_layer(self):
  module=smpl.body_module();original=module.SMPLLayer
  class Base:
   def __init__(self):
    self.layer=module.SMPLLayer(model_path='/invalid/SMPL_NEUTRAL.pkl');self.active=module.SMPLLayer is not original
  backend=smpl.verified_backend(Base)();self.assertTrue(backend.active);self.assertIs(module.SMPLLayer,original)
  out=backend.layer(betas=torch.zeros(1,10));self.assertTrue(torch.isfinite(out.vertices).all())
if __name__=='__main__':unittest.main(verbosity=2)
