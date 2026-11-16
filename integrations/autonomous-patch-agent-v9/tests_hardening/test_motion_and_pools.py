
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

import copy,math,sys,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL))
from core import AgentError,read
from trajectory_quality import assess
from pool_stratification import sample,validate
V={'dt':.1,'wheelbase_m':2.7,'min_acc_mps2':-2.,'max_acc_mps2':2.,'max_steer_rate_degrees':15.,'max_steer_degrees':80.,'max_speed_mps':30.}
P=read(LOCAL/'config.example.json')['host_governance']['motion']
def plan(t=0.,speed=5.,brake=False):
 ts=np.arange(31)*.1;v=np.maximum(0,speed-2*ts) if brake else np.full(31,speed);x=np.r_[0,np.cumsum((v[:-1]+v[1:])*.05)];states=np.c_[x,np.zeros(31),np.zeros(31),v,np.zeros(31)];pose=np.eye(4);pose[0,3]=t*speed
 return {'timestamp':t,'ego_to_world':pose.tolist(),'states':states.tolist(),'action':{'acc_mps2':-2. if brake else 0.,'steer_rate_radps':0.}}
class MotionTests(unittest.TestCase):
 def test_time_and_pose_aligned_straight_plans_pass(self):self.assertTrue(assess([plan(),plan(.1)],V,P)['passed'])
 def test_common_global_rotation_translation_do_not_change_verdict(self):
  plans=[plan(),plan(.1)];a=.7;transform=np.eye(4);transform[:2,:2]=[[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]];transform[:2,3]=[100,200]
  for p in plans:p['ego_to_world']=(transform@np.array(p['ego_to_world'])).tolist()
  self.assertTrue(assess(plans,V,P)['passed'])
 def test_low_speed_inferred_geometry_masked_but_positions_checked(self):
  p=plan(speed=.1);s=np.asarray(p['states']);s[:,2]=np.sin(np.arange(31))*2;p['states']=s.tolist();self.assertTrue(assess([p],V,P)['passed']);s[8,0]+=1;p['states']=s.tolist();self.assertFalse(assess([p],V,P)['passed'])
 def test_low_speed_never_waives_physical_steering_rate(self):
  p=plan(speed=.1);p['action']['steer_rate_radps']=1.;self.assertFalse(assess([p],V,P)['passed'])
 def test_high_speed_bad_curvature_is_not_masked(self):
  p=plan(speed=5.);s=np.asarray(p['states']);s[:,2]=np.sin(np.arange(31))*2;p['states']=s.tolist();self.assertFalse(assess([p],V,P)['passed'])
 def test_one_way_urgent_brake_shock_can_be_waived(self):
  ps=[plan(speed=10),plan(.1,speed=10,brake=True)];risks={1:{'host_verified':True,'ttc_s':.5,'timestamp':.1}};self.assertTrue(assess(ps,V,P,risks)['passed']);self.assertFalse(assess(ps,V,P)['passed'])
 def test_false_or_stale_ttc_does_not_authorize_waiver(self):
  ps=[plan(speed=10),plan(.1,speed=10,brake=True)]
  for risk in [{'host_verified':False,'ttc_s':.1,'timestamp':.1},{'host_verified':True,'ttc_s':.1,'timestamp':0.},{'host_verified':True,'ttc_s':5.,'timestamp':.1}]:self.assertFalse(assess(ps,V,P,{1:risk})['passed'])
 def test_alternating_brake_accelerate_is_not_waived(self):
  ps=[plan(i*.1,speed=10,brake=True) for i in range(3)]
  for p,a in zip(ps,[-1.,1.,-1.]):p['action']['acc_mps2']=a
  r=assess(ps,V,P,{i:{'host_verified':True,'ttc_s':.1,'timestamp':i*.1} for i in range(3)});self.assertFalse(r['passed']);self.assertTrue(any('alternating_execution' in x for x in r['reasons']))
 def test_alternating_steering_is_not_waived(self):
  ps=[plan(i*.1) for i in range(3)]
  for p,a in zip(ps,[.1,-.1,.1]):p['action']['steer_rate_radps']=a
  self.assertTrue(any('alternating_execution' in x for x in assess(ps,V,P)['reasons']))
 def test_jerk_waiver_never_relaxes_acceleration_bounds(self):
  p=plan(speed=10,brake=True);p['action']['acc_mps2']=-10;self.assertFalse(assess([p],V,P,{0:{'host_verified':True,'ttc_s':.1,'timestamp':0.}})['passed'])
 def test_missing_clock_or_empty_reference_fails_closed(self):
  self.assertFalse(assess([],V,P)['passed']);self.assertFalse(assess([plan(),plan(.2)],V,P)['passed'])
 def test_candidate_coordinate_overwrite_is_detected(self):
  ps=[plan(),plan(.1)];ps[1]['states'][0][0]=1.;self.assertFalse(assess(ps,V,P)['passed'])
class PoolTests(unittest.TestCase):
 def policy(self):return {'required':True,'topology_quotas':{'a':1,'b':1},'severity_quotas':{'routine':1,'critical':1},'minimum_distinct_scenes':2}
 def candidates(self):return [{'scene':101,'interaction_topology':'a','severity_band':'routine'},{'scene':102,'interaction_topology':'a','severity_band':'routine'},{'scene':101,'interaction_topology':'b','severity_band':'critical'}]
 def test_feasible_strata_found_even_with_overlapping_scene_options(self):
  p=sample(self.candidates(),[],self.policy(),42);self.assertEqual({x['scene'] for x in p['cases']},{101,102});self.assertTrue(validate(p,self.policy()))
 def test_seed_is_reproducible_and_outcomes_not_inputs(self):
  self.assertEqual(sample(self.candidates(),[],self.policy(),42),sample(self.candidates(),[],self.policy(),42));c=self.candidates();c[0]['scores']=.9
  with self.assertRaises(AgentError):sample(c,[],self.policy(),42)
 def test_retired_scenes_cannot_be_reintroduced_under_new_names(self):
  with self.assertRaises(AgentError):sample(self.candidates(),[101],self.policy(),42)
 def test_insufficient_stratum_assets_wait_instead_of_relaxing(self):
  with self.assertRaises(AgentError):sample(self.candidates()[:1],[],self.policy(),42)

class IntegrationBoundaryTests(unittest.TestCase):
 def test_semi_implicit_bicycle_at_exact_steering_rate_limit_passes(self):
  s=np.zeros((31,5));s[0,3]=3.;rate=math.radians(V['max_steer_rate_degrees'])
  for i in range(1,31):
   s[i]=s[i-1];s[i,3]+=.1;s[i,4]+=rate*.1;s[i,0]+=s[i,3]*math.cos(s[i-1,2])*.1;s[i,1]+=s[i,3]*math.sin(s[i-1,2])*.1;s[i,2]+=s[i,3]*math.tan(s[i,4])/V['wheelbase_m']*.1
  p=plan();p['states']=s.tolist();p['action']={'acc_mps2':.3,'steer_rate_radps':rate};self.assertTrue(assess([p],V,P)['passed'])
 def test_frozen_motion_gate_is_injected_before_gpu_and_not_twice(self):
  import tempfile
  from offline_motion_gate import insert
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);cfg=read(LOCAL/'config.example.json');stages=[{'name':'cpu','gpu':False},{'name':'gpu','gpu':True}];b={'code':str(p),'candidate_pins':{'synthetic':'hash'}};out=insert(stages,p,b,cfg);self.assertEqual([x['name'] for x in out],['cpu','host_motion_contract','gpu']);self.assertEqual(insert(out,p,b,cfg),out)

class InitialJerkBoundaryTests(unittest.TestCase):
 def test_first_control_is_not_exempt_from_jerk_limit(self):
  p=plan(speed=10);p['action']['acc_mps2']=2.;self.assertFalse(assess([p],V,P)['passed'])
 def test_first_brake_needs_verified_low_ttc(self):
  p=plan(speed=10,brake=True);self.assertFalse(assess([p],V,P)['passed']);self.assertTrue(assess([p],V,P,{0:{'host_verified':True,'ttc_s':.1,'timestamp':0.}})['passed'])
