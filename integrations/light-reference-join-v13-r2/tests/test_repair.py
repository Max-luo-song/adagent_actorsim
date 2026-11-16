
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

import sys,time,pickle,unittest,json
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from common import configure,P26,read,write
configure()
from joint_safety import JointReference,POLICY
from runtime_configuration import load_vehicle_config
from live_env import Motion
from tracker import Tracker
from reference_adapter import control as track_reference
V=load_vehicle_config()
def mapping(left=9.2,right=-1.88):return dict(vertices=[[-20,right],[200,right],[200,left],[-20,left]],longitudinal_domain_m=[-20,200],right_line_coefficients=[0,right],left_stations=[[-20,left],[200,left]],uncertainty_m=.3,online_static_map_allowed=True)
def info(m,t):
 c,s=np.cos(m.yaw),np.sin(m.yaw)
 return dict(timestamp=t,scene_token='synthetic',model_lidar_frame=dict(virtual_to_ego=np.eye(4)),explicit_l2g_r_mat=np.array([[c,s,0],[-s,c,0],[0,0,1]]),explicit_l2g_t=np.array([m.x,m.z,0.]))
def observation(m,t,car=20.,empty=False,lateral=0):
 inf=info(m,t);point=np.array([lateral,car,0]);loc=(point-inf['explicit_l2g_t'])@inf['explicit_l2g_r_mat']
 boxes=np.empty((0,9)) if empty else np.array([[*loc,2.,4.5,1.5,-m.yaw,0.,99.]])
 return dict(boxes=boxes,scores=np.full(len(boxes),.7),labels=np.zeros(len(boxes),int)),inf
def intent(m,speed=4.):
 inf=info(m,0);world=np.c_[np.zeros(6),m.z+np.arange(1,7)*speed*.5,np.zeros(6)]
 return ((world-inf['explicit_l2g_t'])@inf['explicit_l2g_r_mat'])[:,:2]
def prime(r,m,car=20.):
 for i in range(3):a,inf=observation(m,i*.1,car);out,sel=r.process(intent(m),a,{},inf,m)
 return out,sel,inf
class RepairTests(unittest.TestCase):
 def test_real_stopped_frame_generates_road_inside_obstacle_free_left_path(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,0.);out,s,inf=prime(r,m,12.)
  self.assertTrue(s['selected_candidate']['clear_conflict_in_preview'] if s['selected_candidate'] else False)
  self.assertIsNotNone(s['selected_candidate']);self.assertEqual(s['selected_candidate']['side'],-1)
  self.assertLess(out[-1,0],-.3);self.assertTrue(s['reference_certificate']['passed'])
  self.assertLessEqual(len(s['candidates']),POLICY['max_candidates'])
 def test_mirrored_road_selects_right_without_forced_left_rule(self):
  r=JointReference(V,mapping(1.88,-9.2));m=Motion(.1,2.7,4.);out,s,_=prime(r,m)
  self.assertIsNotNone(s['selected_candidate']);self.assertEqual(s['selected_candidate']['side'],1)
 def test_no_corridor_returns_wait_not_clipped_collision_path(self):
  r=JointReference(V,mapping(1.88,-1.88));m=Motion(.1,2.7,4.);out,s,_=prime(r,m)
  self.assertTrue(s['selected_candidate'] is None or s['selected_candidate']['side']==0);a,g=r.admit(m,dict(acc_mps2=2.,steer_rate_radps=0.))
  self.assertIsNotNone(a);self.assertLess(a['acc_mps2'],0)
 def test_missing_queries_keep_geometry_and_deny_reacceleration(self):
  r=JointReference(V,mapping(1.88,-1.88));m=Motion(.1,2.7,4.);prime(r,m,12.)
  last=r.tracks[0]['last']
  for i in range(3,9):
   a,inf=observation(m,i*.1,empty=True);r.process(intent(m),a,{},inf,m)
  action,g=r.admit(m,dict(acc_mps2=2.,steer_rate_radps=0.))
  self.assertEqual(r.tracks[0]['last'],last);self.assertTrue(g['risk_latched']);self.assertTrue(action is None or action['acc_mps2']<=0)
 def test_rollout_matches_native_motion_and_physical_limits(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.);a,inf=observation(m,0,empty=True);r.observed({},inf,a)
  guide=r.guides(intent(m),np.empty((0,9)),m,4.)[0];states,controls=r.rollout(guide,m,4.,5.)
  for i,(acc,rate) in enumerate(controls):
   m.advance(dict(acc=acc,steer_rate=rate));np.testing.assert_allclose(states[i+1],[m.x,m.z,m.yaw,m.speed,m.steer],atol=1e-10)
  self.assertLessEqual(np.abs(controls[:,1]).max(),np.radians(V.max_steer_rate_degrees)+1e-10)
  self.assertTrue(np.all(controls[:,0]>=V.min_acc_mps2));self.assertTrue(np.all(controls[:,0]<=V.max_acc_mps2))
 def test_safe_moving_lead_does_not_trigger_unnecessary_lane_change(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(9):
   a,inf=observation(m,i*.1,20.+4.*i*.1)
   out,s=r.process(intent(m),a,{},inf,m)
  self.assertIsNotNone(s['selected_candidate'])
  self.assertEqual(s['selected_candidate']['side'],0)
  self.assertGreater(s['selected_candidate']['speed_mps'],3.5)
  self.assertLess(np.max(np.abs(out[:,0])),.1)
 def test_multiple_initial_speeds_have_physical_candidates(self):
  for speed in (0.,2.,4.,8.):
   r=JointReference(V,mapping());m=Motion(.1,2.7,speed);out,s,_=prime(r,m,max(12.,8.+speed*speed/4.+2*speed))
   self.assertIsNotNone(s['selected_candidate'],(speed,s['mode']))
   self.assertTrue(s['selected_candidate']['certificate']['dynamics_bounds_passed'])
 def test_dense_reference_has_actual_origin_velocity_and_steering(self):
  from types import SimpleNamespace
  r=JointReference(V,mapping());m=Motion(.1,2.7,0.);out,s,inf=prime(r,m,12.)
  tracker=Tracker(V);captured={}
  def solve(initial,states):
   captured.update(initial=initial.copy(),states=states.copy())
   return [SimpleNamespace(input_trajectory=np.zeros((30,2))) ]
  tracker.solver.solve=solve
  track_reference(tracker,r,out,inf,m)
  np.testing.assert_allclose(captured['states'][0],captured['initial'])
  np.testing.assert_allclose(captured['states'][:,4],r.selected_states[:31,4])
 def test_saved_normal_A_has_no_forced_wait(self):
  root=P26/'light_map551_v4/A';d=read(root/'run_report.json');r=JointReference(V,read(root.parent/'map/map.json'));rows=[]
  for t in d['trace']:
   i=t['index'];step=root/f'uniad_worker/step_{i:03d}'
   with (step/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   with np.load(step/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state']
   out,s=r.process(a['trajectory'],a,{},inf,m)
   rows.append(dict(index=i,mode=s['mode'],road=s['reference_certificate']['passed'],seconds=s['seconds']))
  write(P26/'light_reference_join_v13_r2/checks/saved_normal_A_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  self.assertFalse(any(x['mode']=='verified_wait_no_feasible_corridor' for x in rows),[x for x in rows if x['mode']=='verified_wait_no_feasible_corridor'])
 def test_saved_v7_failure_frames_replay(self):
  root=P26/'light_joint_safety_v7/B';d=read(root/'run_report.json');r=JointReference(V,read(root.parent/'map/map.json'));rows=[]
  for i in range(100,154):
   t=d['trace'][i];step=root/f'uniad_worker/step_{i:03d}'
   with (step/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   with np.load(step/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state']
   out,s=r.process(a['trajectory'],a,{},inf,m)
   rows.append(dict(index=i,mode=s['mode'],selected=s['selected_candidate'],seconds=s['seconds'],candidate_count=len(s['candidates']),road=s['reference_certificate']))
  write(P26/'light_reference_join_v13_r2/checks/saved_v7_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  for i in (119,120,125,150):
   x=next(x for x in rows if x['index']==i);self.assertIsNotNone(x['selected'],x);self.assertEqual(x['selected']['side'],-1)
 def test_continuous_rollout_with_actual_ilqr(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.);tracker=Tracker(V);rows=[];gaps=[]
  for i in range(300):
   a,inf=observation(m,i*.1);out,s=r.process(intent(m),a,{},inf,m);nominal,c=track_reference(tracker,r,out,inf,m);action,g=r.admit(m,nominal)
   if action is None:self.fail(str(dict(index=i,selection=s,gate=g)))
   m.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
   from shapely.geometry import Polygon
   c0,s0=np.cos(m.yaw),np.sin(m.yaw);corners=np.array([[-.8,-1.5],[.8,-1.5],[.8,1.5],[-.8,1.5]])
   ego=Polygon(corners@np.array([[c0,-s0],[s0,c0]])+np.array([m.x,m.z]));car=Polygon([[-1,17.75],[1,17.75],[1,22.25],[-1,22.25]])
   self.assertFalse(ego.intersects(car),(i,m.state()));gaps.append(ego.distance(car))
   self.assertGreater(r.geometry.pose_clearance(np.array([[c0,s0,0,m.z],[-s0,c0,0,-m.x],[0,0,1,0],[0,0,0,1]])),-.01)
   rows.append(dict(index=i,state=m.state(),mode=s['mode'],selected=s['selected_candidate'],gate=g['selected']))
   if m.z>30 and abs(m.x)<1 and abs(m.yaw)<.1 and m.speed>1:break
  write(P26/'light_reference_join_v13_r2/checks/continuous_ilqr.json',dict(rows=rows,min_body_gap_m=min(gaps),GPU_executed=False,synthetic_observation=True))
  self.assertGreater(m.z,30.,m.state());self.assertLess(abs(m.x),1.,m.state());self.assertGreater(m.speed,1.)
if __name__=='__main__':unittest.main(verbosity=2)
