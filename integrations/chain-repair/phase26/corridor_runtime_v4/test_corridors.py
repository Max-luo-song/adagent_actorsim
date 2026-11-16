
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
import sys,types,unittest,json
import numpy as np
P=next(p for p in Path(__file__).resolve().parents if (p/'corridor_runtime_v4').is_dir())/'corridor_runtime_v4';sys.path.insert(0,str(P))
from bootstrap import *
from corridor_geometry import forward_constraints,side_constraints
from cruise_seeds import propose
from empty_controller import rollout
from runtime_configuration import load_vehicle_config
from road_soft_cost import SoftRoadPenalty
from road_priority import road_priority_cost
class CorridorTests(unittest.TestCase):
 def test_real_archived_spinning_left_trajectory_is_rejected(self):
  f=P.parent/'bridge/closed_loop_integration/runs/20261107T145401Z/uniad_worker/step_000/optimization.json';d=_hugsim_expand(json.loads(f.read_text()));old=next(r for r in d['trials'] if r['seed']=='escape_left');self.assertLess(forward_constraints(np.asarray(old['states']),load_vehicle_config()).min(),-1.)
 def test_fitted_left_and_right_seeds_are_smooth_forward_and_symmetric(self):
  v=load_vehicle_config();initial=np.array([0.,0.,0.,8.,0.]);guard=types.SimpleNamespace(hard_boxes=[],hard_radii=[],hard_margins=[]);seeds,audit=propose(initial,v,guard,None,6.,1.);paths={}
  for name,controls in seeds:
   states,_=rollout(controls,8.,0.,np.zeros(2),v.dt,v.wheelbase_m,1.);paths[name]=states
   self.assertGreaterEqual(forward_constraints(states,v).min(),-1e-5);self.assertGreaterEqual(side_constraints(states,initial,audit['corridors'][name]).min(),-1e-2)
  np.testing.assert_allclose(paths['escape_left'][:,0],-paths['escape_right'][:,0],atol=1e-6)
 def test_left_start_cannot_collapse_into_right_or_return_before_obstacle(self):
  states=np.zeros((5,5));states[:,1]=[0,1,2,3,4];states[:,0]=[0,-1,-2,-2,0];initial=states[0].copy();spec=dict(sign=-1,displacement_m=-2.,anchor_index=2,release_forward_m=10.)
  self.assertLess(side_constraints(states,initial,spec).min(),0.);states[-1,0]=-2.;self.assertGreaterEqual(side_constraints(states,initial,spec).min(),0.);states[:,0]*=-1;self.assertLess(side_constraints(states,initial,spec).min(),0.)
 def test_road_peak_buffer_is_symmetric_and_not_diluted_by_safe_samples(self):
  v=load_vehicle_config();mask=np.ones((40,40),bool);road=types.SimpleNamespace(mask=mask,bounds=np.array([-10.,-10.,10.,10.]),dx=.5,dy=.5,clearance=lambda s:2.-np.abs(s[:,0]));soft=SoftRoadPenalty(road,v)
  a=np.zeros((31,5));a[:,3]=1.;a[-1,0]=1.5;b=a.copy();b[:,0]*=-1
  self.assertAlmostEqual(soft.peak_violation(a,buffer_m=1.,start_s=1.),.25);self.assertAlmostEqual(soft.peak_violation(a,1.,1.),soft.peak_violation(b,1.,1.));self.assertAlmostEqual(soft.peak_violation(np.repeat(a,3,axis=0),1.,1.),.25)
 def test_peak_excludes_fixed_prefix_but_integrated_road_cost_keeps_it(self):
  v=load_vehicle_config();road=types.SimpleNamespace(mask=np.ones((40,40),bool),bounds=np.array([-10.,-10.,10.,10.]),dx=.5,dy=.5,clearance=lambda s:2.-np.abs(s[:,0]));soft=SoftRoadPenalty(road,v);s=np.zeros((21,5));s[0,0]=4.
  self.assertEqual(soft.peak_violation(s,1.,1.),0.);self.assertGreater(soft.cost(s,5),0.)
 def test_missing_road_does_not_invent_free_space_or_direction(self):
  v=load_vehicle_config();road=types.SimpleNamespace(mask=np.zeros((40,40),bool),bounds=np.array([-10.,-10.,10.,10.]),dx=.5,dy=.5,clearance=lambda s:np.full(len(s),-100.));soft=SoftRoadPenalty(road,v);self.assertEqual(soft.peak_violation(np.zeros((20,5)),1.,1.),0.);self.assertEqual(road_priority_cost(dict(road_support_available=False,road_loss=100.,road_peak_violation_m2=100.)),0.)
if __name__=='__main__':unittest.main()
