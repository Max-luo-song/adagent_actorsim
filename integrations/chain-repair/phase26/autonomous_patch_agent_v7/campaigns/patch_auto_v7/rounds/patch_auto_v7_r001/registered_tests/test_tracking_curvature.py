
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

import sys,importlib.util,unittest,json
pass  # Path is provided by the portable boundary above
import numpy as np
R=Path(_hugsim_expand('${HUGSIM_ROOT}'));C=R/'integrations/light-agent-patch-auto-v7-r001';sys.path[:0]=[str(C/'tests'),str(C)]
from test_repair import V,Motion,JointReference,mapping,observation
spec=importlib.util.spec_from_file_location('_baseline_curvature',R/'integrations/light-reference-join-v13-r2/joint_safety.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
class Tests(unittest.TestCase):
 def planner(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,8.);a,inf=observation(m,0,empty=True);r.observed({},inf,a);return r,m
 def guide(self,step,sign=1):
  y=np.arange(0,40+step/2,step);x=sign*np.interp(y,[0,10,20,40],[0,.1,.1,.1]);return dict(y=y,x=x)
 def test_same_geometric_polyline_is_grid_invariant(self):
  r,m=self.planner();rolls=[r.rollout(self.guide(h),m,8.,3.)[0] for h in (.4,.2,.1)]
  for s in rolls[1:]:np.testing.assert_allclose(s,rolls[0],rtol=0,atol=1e-9)
 def test_baseline_grid_spike_and_candidate_control_response(self):
  r,m=self.planner();b=old.JointReference(V,mapping());a,inf=observation(m,0,empty=True);b.observed({},inf,a)
  olds=[b.rollout(self.guide(h),m,8.,3.)[0] for h in (.4,.1)];news=[r.rollout(self.guide(h),m,8.,3.)[0] for h in (.4,.1)]
  delta=float(np.max(np.abs(olds[0][:,3]-olds[1][:,3])));self.assertGreater(delta,.01);self.assertLess(np.max(np.abs(news[0][:,3]-news[1][:,3])),1e-9)
  print('baseline_grid_speed_difference_mps',delta)
 def test_mirror_symmetry_and_physical_step_limits(self):
  r,m=self.planner();p,pc=r.rollout(self.guide(.2,1),m,8.,3.);n,nc=r.rollout(self.guide(.2,-1),m,8.,3.);np.testing.assert_allclose(p[:,[0,2,4]],-n[:,[0,2,4]],atol=1e-10);np.testing.assert_allclose(p[:,[1,3]],n[:,[1,3]],atol=1e-10)
  self.assertTrue(np.all(np.abs(pc[:,1])<=np.radians(V.max_steer_rate_degrees)+1e-12));self.assertTrue(np.all(pc[:,0]>=V.min_acc_mps2-1e-12));self.assertTrue(np.all(pc[:,0]<=V.max_acc_mps2+1e-12));self.assertTrue(np.all(np.abs(np.diff(pc[:,0]))<=.3+1e-9))
 def test_current_steering_still_limits_speed_during_unwind(self):
  r,m=self.planner();m.steer=np.radians(V.max_steer_degrees);g=self.guide(.2);g['x'][:]=0;s,u=r.rollout(g,m,8.,.5);self.assertLess(u[0,0],0.);self.assertLess(s[1,3],m.speed);self.assertTrue(np.isfinite(s).all())
 def test_retained_grid_variant_is_exactly_original(self):
  r,m=self.planner();b=old.JointReference(V,mapping());a,inf=observation(m,0,empty=True);b.observed({},inf,a)
  for h in (.4,.2,.1):
   for sign in (-1,1):
    current=r._rollout_variant(self.guide(h,sign),m,8.,3.,False);previous=b.rollout(self.guide(h,sign),m,8.,3.)
    for x,y in zip(current,previous):np.testing.assert_array_equal(x,y)
 def test_original_safe_stop_survives_unsafe_faster_profile(self):
  r,m=self.planner();g=self.guide(.1);baseline=r._rollout_variant(g,m,8.,3.,False);candidate=r._rollout_variant(g,m,8.,3.,True);found=False
  for distance in np.arange(25.,60.,.5):
   boxes=np.array([[0,distance,0,2,4.5,1.5,0,0,0.]])
   before=r.joint_certificate(r.terminal_stop(baseline[0]),boxes)['passed'];after=r.joint_certificate(r.terminal_stop(candidate[0]),boxes)['passed']
   if before and not after:
    found=True;r._risk_boxes=boxes;r._risk_tracks=[];r._speed_reference_states=baseline[0][:31].copy();chosen=r.rollout(g,m,8.,3.)
    for x,y in zip(chosen,baseline):np.testing.assert_array_equal(x,y)
    self.assertTrue(r.joint_certificate(r.terminal_stop(chosen[0]),boxes)['passed']);break
  self.assertTrue(found,'Fixture must exercise a real geometric fallback')
if __name__=='__main__':unittest.main(verbosity=2)
