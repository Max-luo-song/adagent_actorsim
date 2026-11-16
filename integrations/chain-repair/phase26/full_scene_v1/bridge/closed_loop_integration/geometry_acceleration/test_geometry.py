
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

import json,pickle,sys,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_edited_closed_loop as bridge
sys.path.insert(0,str(bridge.SOURCE))
import bootstrap
from bounded_visibility_geometry import polygon_clearance as original,projected_cell
import accelerated as fast

class PolygonTests(unittest.TestCase):
 def test_random_exact_arrays_tree_and_full_scan(self):
  rng=np.random.default_rng(20261106)
  from scipy.spatial.transform import Rotation
  maximum=0.
  for count in (1,7,61):
   for tree_mode in (False,True):
    for repeat in range(12):
     polygon=projected_cell(Rotation.random(random_state=rng).as_matrix());centers=rng.normal(size=(220,2))*20
     states=np.zeros((count,7));states[:,:2]=rng.normal(size=(count,2))*15;states[:,2]=rng.uniform(-np.pi,np.pi,count);states[:,5:]=rng.uniform(.2,3,(count,2))
     tree=cKDTree(centers) if tree_mode else None
     old=original(states,centers,polygon,.5,tree=tree);new=fast.polygon_function(original)(states,centers,polygon,.5,tree=tree)
     maximum=max(maximum,float(np.max(np.abs(old-new))))
     np.testing.assert_array_equal(new,old)
  self.assertEqual(maximum,0.)
 def test_empty_and_finite_difference(self):
  p=np.array([[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]]);s=np.array([[0,0,0,0,0,.8,1.5]],float);c=np.array([[1.55,0.],[5.,3.]])
  f=fast.polygon_function(original)
  np.testing.assert_array_equal(f(s,[],p,.5),original(s,[],p,.5))
  for eps in (0.,1.49e-8,-1.49e-8,1e-12):
   q=s.copy();q[0,0]+=eps
   np.testing.assert_array_equal(f(q,c,p,.5,tree=cKDTree(c)),original(q,c,p,.5,tree=cKDTree(c)))
 def test_installer_road_and_shared_aliases_and_restoration(self):
  from drivable_area import DrivableArea
  from runtime_configuration import load_vehicle_config
  from mpc_reuse_v1.cached_geometry import CachedPolygon
  source=bridge.HERE/'runs/20261106T143547Z/uniad_worker/step_000'
  rel=_hugsim_expand(json.loads((source/'guard_memory_report.json').read_text()));old=_hugsim_expand(json.loads((source/'optimization.json').read_text()));vehicle=load_vehicle_config()
  road=DrivableArea(rel['predicted_drivable_area'],vehicle);states=[np.asarray(x['states']) for x in old['trials']];expected=[road.clearance(s) for s in states];method=DrivableArea.clearance
  audit=fast.install()
  try:
   self.assertGreaterEqual(len(audit['bindings']),4)
   for s,ref in zip(states,expected):np.testing.assert_array_equal(road.clearance(s),ref)
   with self.assertRaises(RuntimeError):fast.install()
  finally:fast.uninstall()
  self.assertIs(DrivableArea.clearance,method)
 def test_unsupported_degenerate_polygon_preserves_reference(self):
  states=np.array([[0.,0.,0.,0.,0.,.8,1.5]]);centers=np.array([[2.,0.]])
  p=np.array([[0.,0.],[0.,0.],[1.,0.]])
  with np.errstate(invalid='ignore',divide='ignore'):
   np.testing.assert_array_equal(fast.polygon_function(original)(states,centers,p,.5),original(states,centers,p,.5))
if __name__=='__main__':unittest.main(verbosity=2)
