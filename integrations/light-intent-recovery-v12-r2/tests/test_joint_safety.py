
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

import json,pickle,sys,unittest,copy,time
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(LOCAL),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from common import configure,P26,read,write
configure()
from joint_safety_v7 import JointReference
from runtime_configuration import load_vehicle_config
from live_env import Motion
from tracker import Tracker

V=load_vehicle_config()
MAP=dict(vertices=[[-20,-8],[200,-8],[200,8],[-20,8]],longitudinal_domain_m=[-20,200],right_line_coefficients=[0,-8],left_stations=[[-20,8],[200,8]],uncertainty_m=.3,online_static_map_allowed=True)
def info(t,z=0,x=0):return dict(timestamp=t,scene_token='synthetic',model_lidar_frame=dict(virtual_to_ego=np.eye(4)),explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.array([x,z,0.]))
def arrays(y=None,x=0):
 b=np.empty((0,9)) if y is None else np.array([[x,y,0,2.,4.5,1.5,0.,0.,99.]])
 return dict(boxes=b,scores=np.full(len(b),.7),labels=np.zeros(len(b),int))
def raw(speed=4.,x=0):return np.c_[np.linspace(x/6,x,6),np.arange(1,7)*.5*speed]
NOM=dict(acc_mps2=2.,steer_rate_radps=0.)

class JointTests(unittest.TestCase):
 def test_legal_lateral_native_reference_is_exactly_preserved(self):
  r=JointReference(V,MAP);m=Motion(.1,2.7,4.)
  original=raw(x=-2.5);out,a=r.process(original,arrays(),{},info(0),m)
  np.testing.assert_array_equal(out,original);self.assertTrue(a['legal_native_reference_preserved'])
 def test_boundary_projection_rechecked_against_obstacle(self):
  r=JointReference(V,MAP);m=Motion(.1,2.7,4.)
  for i in range(4):out,a=r.process(raw(x=12),arrays(10),{},info(i*.1),m)
  self.assertTrue(a['reference_changed']);action,g=r.admit(m,NOM)
  self.assertTrue(g['override']);self.assertTrue(action is None or action['acc_mps2']<=0)
 def test_missing_queries_do_not_release_front_risk(self):
  r=JointReference(V,MAP);m=Motion(.1,2.7,4.)
  for i in range(3):r.process(raw(),arrays(12),{},info(i*.1),m)
  action,g=r.admit(m,NOM);self.assertTrue(g['risk_latched']);self.assertLess(action['acc_mps2'],0)
  last=r.tracks[0]['last']
  for i in range(3,15):
   r.process(raw(),arrays(),{},info(i*.1),m);action,g=r.admit(m,NOM)
   self.assertTrue(g['risk_latched']);self.assertTrue(action is None or action['acc_mps2']<=0)
  self.assertEqual(r.tracks[0]['last'],last)
 def test_latched_geometry_survives_single_reacquisition_after_gap(self):
  r=JointReference(V,MAP);m=Motion(.1,2.7,4.)
  for i in range(3):r.process(raw(),arrays(12),{},info(i*.1),m)
  r.admit(m,NOM)
  for i in range(3,11):r.process(raw(),arrays(),{},info(i*.1),m)
  r.process(raw(),arrays(12),{},info(1.1),m)
  boxes,tracks=r.actor_boxes();self.assertTrue(any(t['latched'] for t in tracks));self.assertGreater(len(boxes),0)
  action,g=r.admit(m,NOM);self.assertLessEqual(action['acc_mps2'],0)
 def test_duplicate_boxes_are_not_independent_confirmation(self):
  r=JointReference(V,MAP);a=arrays(12);a={k:np.repeat(v,3,axis=0) for k,v in a.items()}
  r.process(raw(),a,{},info(0),Motion(.1,2.7,4.));self.assertEqual(len(r.tracks),1);self.assertEqual(len(r.actor_boxes()[0]),0)
 def test_adjacent_actor_and_valid_empty_do_not_force_stop(self):
  for x in (-6,6):
   r=JointReference(V,MAP);m=Motion(.1,2.7,4.)
   for i in range(5):r.process(raw(),arrays(8,x),{},info(i*.1),m)
   a,g=r.admit(m,NOM);self.assertFalse(g['override']);self.assertEqual(a,NOM)
 def test_malformed_evidence_and_repeated_timestamp_rejected(self):
  r=JointReference(V,MAP);m=Motion(.1,2.7,4.);r.process(raw(),arrays(),{},info(0),m)
  with self.assertRaises(ValueError):r.process(raw(),arrays(),{},info(0),m)
  a=arrays(12);a['boxes'][0,0]=np.nan
  with self.assertRaises(ValueError):r.process(raw(),a,{},info(.1),m)
 def test_risk_clears_only_after_repeated_fresh_separated_reference(self):
  r=JointReference(V,MAP);m=Motion(.1,2.7,4.)
  for i in range(3):r.process(raw(),arrays(12),{},info(i*.1),m)
  r.admit(m,NOM)
  for i in range(3,6):
   r.process(raw(x=-4),arrays(12),{},info(i*.1),m);a,g=r.admit(m,NOM)
  self.assertFalse(g['risk_latched']);self.assertEqual(a,NOM)
 def test_stationary_front_brakes_at_multiple_initial_speeds(self):
  for speed in (2.,4.,8.):
   r=JointReference(V,MAP);m=Motion(.1,2.7,speed);obstacle=15.+speed**2/4.;min_gap=999.;braked=False
   for i in range(100):
    # Every seventh frame has missing queries. Truth creates this test's
    # synthetic sensor, never enters the controller itself.
    a=arrays() if i%7==6 else arrays(obstacle-m.z)
    r.process(raw(speed),a,{},info(i*.1,m.z,m.x),m)
    action,g=r.admit(m,dict(acc_mps2=0.,steer_rate_radps=0.))
    self.assertIsNotNone(action,(speed,i,g));braked|=g['override']
    m.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
    min_gap=min(min_gap,obstacle-m.z-2.25-V.half_length_m)
   self.assertTrue(braked);self.assertGreater(min_gap,.2);self.assertLess(m.speed,.01)
 def test_saved_A4_all_observations_remain_without_spurious_actor_stops(self):
  root=P26/'light_map551_v4/A';r=JointReference(V,read(root.parent/'map/map.json'));report=read(root/'run_report.json');rows=[]
  for item in report['observations']:
   i=item['index'];step=root/f'uniad_worker/step_{i:03d}'
   if not (step/'perception.npz').exists():continue
   with (step/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   with np.load(step/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   state=item['motion_state'];m=Motion(.1,2.7,state[3]);m.x,m.z,m.yaw,m.speed,m.steer=state
   adopted,selection=r.process(a['trajectory'],a,{},inf,m)
   if i<len(report['trace']):
    action,g=r.admit(m,report['trace'][i]['action'])
    rows.append(dict(index=i,actor_reference_passed=g['reference_actor_certificate']['passed'],latched=g['risk_latched'],selected=g['selected']))
  write(P26/'light_intent_recovery_v12_r2/checks/saved_A4_joint_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  self.assertFalse(any(x['latched'] for x in rows))
 def test_rotation_translation_and_clock_reset(self):
  r=JointReference(V,MAP);m=Motion(.1,2.7,4.)
  for i in range(3):r.process(raw(),arrays(12),{},info(i*.1),m)
  r.admit(m,NOM);self.assertTrue(r.tracks[0]['latched'])
  r.process(raw(),arrays(),{},info(0),m);self.assertEqual(len(r.tracks),0)
  # A box's model yaw is transformed by the full pose, then transformed back.
  for i in range(3):
   inf=info((i+1)*.1);theta=.12;inf['explicit_l2g_r_mat']=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
   r.process(raw(),arrays(12,1),{},inf,m)
  b,_=r.actor_boxes();np.testing.assert_allclose(b[0,:2],[1,12],atol=1e-6);self.assertAlmostEqual(b[0,6],0.,places=6)
 def test_saved_B6_actual_tracker_and_joint_gate_replay(self):
  root=P26/'light_map551_cutin_native_v6/B';r=JointReference(V,read(root.parent/'map/map.json'));tracker=Tracker(V)
  report=read(root/'run_report.json');trace={t['index']:t for t in report['trace']};rows=[]
  for i in range(153):
   step=root/f'uniad_worker/step_{i:03d}'
   with (step/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   obs={}
   with np.load(step/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   state=report['observations'][i]['motion_state'];m=Motion(.1,2.7,state[3]);m.x,m.z,m.yaw,m.speed,m.steer=state
   adopted,selection=r.process(a['trajectory'],a,obs,inf,m)
   if i in (117,120,125,130,131,135,140):
    nominal,_=tracker.control(adopted,inf,m);action,gate=r.admit(m,nominal)
    rows.append(dict(index=i,old_action=trace[i]['action'],new_action=action,gate=gate,selection=selection))
   elif i<153:r.admit(m,dict(acc_mps2=0.,steer_rate_radps=0.))
  check={row['index']:row for row in rows}
  self.assertGreater(check[130]['old_action']['acc_mps2'],1.)
  self.assertTrue(check[130]['new_action'] is None or check[130]['new_action']['acc_mps2']<=0.)
  self.assertTrue(check[131]['new_action'] is None or check[131]['new_action']['acc_mps2']<=0.)
  out=P26/'light_intent_recovery_v12_r2/checks';write(out/'saved_B6_joint_replay.json',dict(passed=True,rows=rows,counterfactual_closed_loop=False,GPU_executed=False))

if __name__=='__main__':unittest.main(verbosity=2)
