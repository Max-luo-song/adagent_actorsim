
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

import sys,importlib.util,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
R=Path(__file__).resolve().parents[9] if False else Path(_hugsim_expand('${HUGSIM_ROOT}'))
C=R/'integrations/light-reference-horizon-v14-r3';sys.path[:0]=[str(C/'tests'),str(C)]
from test_repair import V,Motion,JointReference,mapping,observation,intent
spec=importlib.util.spec_from_file_location('_previous_join',R/'integrations/light-reference-join-v13-r2/joint_safety.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
class Tests(unittest.TestCase):
 def pair(self,speed):
  m=Motion(.1,2.7,speed);m.steer=.02;a,info=observation(m,0,empty=True);r=JointReference(V,mapping());b=old.JointReference(V,mapping());r.observed({},info,a);b.observed({},info,a);return m,r,b
 def test_blocked_guides_unchanged_at_multiple_speeds(self):
  for speed in (0.,2.,4.,8.):
   m,r,b=self.pair(speed);raw=intent(m,max(speed,2.));box=np.array([[0,10,0,2,4.5,1.5,0,0,0.]])
   one=r.guides(raw,box,m,max(speed,2.));two=b.guides(raw,box,m,max(speed,2.));self.assertEqual(len(one),len(two))
   for x,y in zip(one,two):np.testing.assert_array_equal(x['x'],y['x']);np.testing.assert_array_equal(x['y'],y['y'])
 def test_clear_guides_keep_current_origin_and_far_tail(self):
  for speed in (0.,2.,4.,8.):
   m,r,b=self.pair(speed);raw=intent(m,max(speed,2.));one=r.guides(raw,np.empty((0,9)),m,max(speed,2.))[0];two=b.guides(raw,np.empty((0,9)),m,max(speed,2.))[0];tail=one['y']>=max(10,max(speed,2.)*3)
   self.assertEqual(one['x'][0],-r.mount[0,3]);np.testing.assert_allclose(one['x'][tail],two['x'][tail],atol=1e-12);self.assertTrue(np.isfinite(one['x']).all())
 def test_clear_join_changes_only_current_curve_and_not_inputs(self):
  m,r,b=self.pair(4.);raw=intent(m);raw[:,0]=.05*raw[:,1];saved=raw.copy();one=r.guides(raw,np.empty((0,9)),m,4.)[0];two=b.guides(raw,np.empty((0,9)),m,4.)[0];self.assertGreater(np.max(np.abs(one['x']-two['x'])),.001);np.testing.assert_array_equal(raw,saved)
if __name__=='__main__':unittest.main(verbosity=2)
