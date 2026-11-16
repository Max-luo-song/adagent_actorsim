
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

import sys,unittest,ast,copy
pass  # Path is provided by the portable boundary above
import numpy as np
R=Path(_hugsim_expand('${HUGSIM_ROOT}'));C=R/'integrations/light-agent-smoke-20261112t080213';sys.path[:0]=[str(C/'tests'),str(C)]
from test_repair import V,Motion,mapping,observation,intent,prime,JointReference
from joint_safety import GeometryReference
class GeometrySpeedTests(unittest.TestCase):
 def setup_path(self,speed=4.,sign=1.):
  r=JointReference(V,mapping(20,-20));m=Motion(.1,2.7,speed);a,inf=observation(m,0,empty=True);r.observed({},inf,a)
  y=np.arange(0,80.01,.2);guide=dict(y=y,x=sign*.1*np.sin(y/6.),side=0,end_y=30,clear_y=None)
  states,controls=r.rollout(guide,m,4.,8.)
  return r,m,states,controls
 def test_geometry_selection_ast_and_rollout_are_unchanged(self):
  def cls(p,name):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
  old=cls(R/'integrations/light-reference-join-v13-r2/joint_safety.py','JointReference');new=cls(C/'joint_safety.py','GeometryReference')
  for name in ('process','rollout','guides','intent_pool','admit','joint_certificate','actor_certificate'):
   a=next(n for n in old.body if getattr(n,'name','')==name);b=next(n for n in new.body if getattr(n,'name','')==name)
   self.assertEqual(ast.dump(a),ast.dump(b),name)
 def test_retimed_controls_reproduce_native_motion(self):
  r,m,s,u=self.setup_path();proposal,d=r.retime_selected(s,u,m,4.);self.assertIsNotNone(proposal,d)
  ns,nu=proposal
  for i,(acc,rate) in enumerate(nu):
   m.advance(dict(acc=acc,steer_rate=rate));np.testing.assert_allclose(ns[i+1],[m.x,m.z,m.yaw,m.speed,m.steer],atol=1e-10)
 def test_spatial_support_never_extrapolates(self):
  r,m,s,u=self.setup_path();p,d=r.retime_selected(s[:4],np.zeros((500,2)),m,12.)
  self.assertIsNone(p);self.assertIn(d['reason'],('spatial_support_exhausted','spatial_footprint_discrepancy'))
 def test_multiple_speeds_and_mirrors_preserve_bounded_geometry(self):
  for speed in (0.,2.,4.,8.):
   for sign in (-1.,1.):
    r,m,s,u=self.setup_path(speed,sign);p,d=r.retime_selected(s,u,m,4.)
    if p is not None:
     ns,nu=p;self.assertLessEqual(d['maximum_discrepancy_m'],.25+1e-9);self.assertLessEqual(d['traveled_m'],d['support_m']+1e-9)
     acc=np.r_[m.last_accel,nu[:,0]];self.assertLessEqual(np.max(np.abs(np.diff(acc))),.3+1e-8)
     self.assertTrue(r.joint_certificate(ns,np.empty((0,9)))['dynamics_bounds_passed'])
 def test_rejected_proposal_returns_baseline_arrays_and_state(self):
  r=JointReference(V,mapping());b=GeometryReference(V,mapping());m=Motion(.1,2.7,4.)
  r.retime_selected=lambda *a:(None,dict(reason='forced_failed_certificate_fixture'))
  for i in range(3):
   arrays,inf=observation(m,i*.1,20.);raw=intent(m)
   x,ba=b.process(raw,arrays,{},inf,m);y,ra=r.process(raw,arrays,{},inf,m)
   np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(b.selected_states,r.selected_states)
   self.assertEqual(b.side,r.side);self.assertEqual(b.previous_target_speed,r.previous_target_speed)
   self.assertEqual(ba['intent_recovery'],ra['intent_recovery']);self.assertFalse(ra['spatial_retiming']['accepted'])
 def test_retiming_cannot_reselect_candidate_geometry(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.);original=r.retime_selected;seen=[]
  def wrapped(s,u,motion,target):
   seen.append((r.selected['side'],r.side,r.selected['speed_mps'],target));return original(s,u,motion,target)
  r.retime_selected=wrapped
  out,a,inf=prime(r,m)
  self.assertTrue(seen)
  for side,current,speed,target in seen:self.assertEqual(side,current);self.assertEqual(speed,target)
  self.assertEqual(a['selected_candidate']['side'],r.side)
 def test_new_timing_must_recertify_actor_and_road(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.);original=r.retime_selected
  def corrupt(s,u,motion,target):
   ns=s.copy();ns[:,0]+=1000;return (ns,u.copy()),dict(reason='unsafe_fixture')
  r.retime_selected=corrupt;out,a,_=prime(r,m)
  self.assertFalse(a['spatial_retiming']['accepted']);self.assertFalse(a['spatial_retiming']['certificate']['passed']);self.assertLess(np.max(np.abs(out[:,0])),20)
 def test_healthy_native_plan_and_missing_memory_remain_unchanged(self):
  r=JointReference(V,mapping());b=GeometryReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(9):
   arrays,inf=observation(m,i*.1,car=20,empty=(i>2));raw=intent(m)
   x,ba=b.process(raw,arrays,{},inf,m);y,ra=r.process(raw,arrays,{},inf,m)
   self.assertEqual([(t['id'],t['last'],t['latched']) for t in b.tracks],[(t['id'],t['last'],t['latched']) for t in r.tracks])
   self.assertEqual(b.intent_recovery_audit,r.intent_recovery_audit)
class ExecutionBenefitTests(unittest.TestCase):
 def fixture(self,accels):
  state=np.array([0.,0.,0.,2.,0.]);rows=[state.copy()];u=np.c_[accels,np.zeros(len(accels))]
  for acc,rate in u:
   state[3]+=acc*.1;state[1]+=state[3]*.1;rows.append(state.copy())
  return np.array(rows),u
 def evaluate(self,base,proposal,original=None):
  r=JointReference(V,mapping(20,-20));m=Motion(.1,2.7,2.);a,inf=observation(m,0,empty=True);r.observed({},inf,a)
  ref=np.zeros((31,5));ref[:,1]=np.arange(31)*.4
  return r.execution_benefit(*base,*proposal,ref if original is None else original,np.empty((0,9)),[])
 def test_deferred_only_gain_is_not_promoted(self):
  a=np.zeros(60);b=a.copy();b[35:]=.1
  d=self.evaluate(self.fixture(a),self.fixture(b));self.assertFalse(d['executed_prefix_improved']);self.assertAlmostEqual(d['executed_prefix_gain'],0.)
 def test_prefix_gain_without_current_acceleration_gain_is_not_promoted(self):
  a=np.zeros(60);b=a.copy();b[2:]=.1
  d=self.evaluate(self.fixture(a),self.fixture(b));self.assertGreater(d['executed_prefix_gain'],0.);self.assertFalse(d['executed_prefix_improved'])
 def test_first_acceleration_gain_without_prefix_gain_is_not_promoted(self):
  a=np.zeros(60);b=a.copy();b[0]=.1;b[1:]=-1./60
  baseline=self.fixture(a);d=self.evaluate(baseline,self.fixture(b),baseline[0][:31]);self.assertGreater(d['first_acceleration_gain_mps2'],0.);self.assertLess(d['executed_prefix_gain'],0.);self.assertFalse(d['executed_prefix_improved'])
 def test_real_immediate_and_prefix_gain_remains_eligible(self):
  d=self.evaluate(self.fixture(np.zeros(60)),self.fixture(np.full(60,.1)))
  self.assertTrue(d['executed_prefix_improved']);self.assertGreater(d['executed_prefix_gain'],0.);self.assertGreater(d['first_acceleration_gain_mps2'],0.)
if __name__=='__main__':unittest.main(verbosity=2)
