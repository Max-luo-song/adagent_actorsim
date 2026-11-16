
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

import importlib.util,json,sys,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
import torch
LOCAL=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('policy_under_test',LOCAL/'native_collision.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Head:
 def __init__(self):self.use_col_optim=False;self.planning_steps=6;self.bev_h=self.bev_w=200;self.occ_filter_range=5.;self.calls=0
 def collision_optimization(self,raw,occ):self.calls+=1;return raw+1 if bool(occ.any()) else raw
 def forward(self,bev_embed=None,occ_mask=None):
  raw=torch.zeros((1,6,2));out=self.collision_optimization(raw,occ_mask) if self.use_col_optim and occ_mask is not None else raw
  return dict(sdc_traj=out,sdc_traj_all=out)
def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def write(p,v):Path(p).write_text(json.dumps(v))
class NativeCollisionTests(unittest.TestCase):
 def invoke(self,index,occ):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp)/('step_%03d'%index);folder.mkdir();h=Head();old=h.forward;restore=m.install(types.SimpleNamespace(planning_head=h),types.SimpleNamespace(JOB=Path(tmp)),read,write)
   result=h.forward(occ_mask=occ);audit=read(folder/'native_collision_audit.json');restore();self.assertEqual(h.forward,old);self.assertFalse(h.use_col_optim);return result,audit,h.calls
 def test_original_common_prefix_is_unchanged(self):
  out,a,c=self.invoke(5,torch.ones((1,5,200,200)));self.assertFalse(a['enabled']);self.assertEqual(c,0);self.assertEqual(a['trajectory_max_change_m'],0)
 def test_native_postprocessor_activated_after_shared_history(self):
  out,a,c=self.invoke(6,torch.ones((1,5,200,200)));self.assertTrue(a['enabled']);self.assertEqual(c,1);self.assertEqual(a['native_method_calls'],1);self.assertEqual(a['trajectory_max_change_m'],1.)
 def test_no_occupancy_skips_without_fabricating_array(self):
  out,a,c=self.invoke(7,None);self.assertEqual(c,0);self.assertFalse(a['occupancy_available']);self.assertFalse(a['missing_occupancy_replaced']);self.assertEqual(a['skip_reason'],'occupancy_unavailable_native_skip')
 def test_empty_occupancy_keeps_native_identity(self):
  out,a,c=self.invoke(7,torch.zeros((1,5,200,200)));self.assertEqual(c,1);self.assertFalse(a['native_filter_requires_solver']);self.assertEqual(a['trajectory_max_change_m'],0)
 def test_both_native_occupancy_shapes_are_supported(self):
  out,a,c=self.invoke(7,torch.zeros((1,5,1,200,200)));self.assertTrue(a['enabled']);self.assertEqual(c,1)
 def test_module_has_no_scene_or_actor_truth_dependency(self):
  source=(LOCAL/'native_collision.py').read_text();self.assertNotIn('target_track',source);self.assertNotIn('RigidNodes',source);self.assertNotIn('ego_to_world',source)
if __name__=='__main__':unittest.main(verbosity=2)
