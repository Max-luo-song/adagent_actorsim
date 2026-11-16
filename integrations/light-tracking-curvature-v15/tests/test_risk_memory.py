
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

import copy,json,pickle,sys,time,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,mapping,info,observation,intent,prime,JointReference,Motion,Tracker,track_reference,P26,read,write

class RiskMemoryTests(unittest.TestCase):
 def setup_risk(self,road=None,distance=12.):
  r=JointReference(V,road or mapping());m=Motion(.1,2.7,4.);prime(r,m,distance)
  return r,m
 def test_stale_risk_can_be_bypassed_with_geometry_intact(self):
  r,m=self.setup_risk(distance=20.);old=r.tracks[0];last=old['last']
  for i in range(3,9):
   a,inf=observation(m,i*.1,empty=True);out,s=r.process(intent(m),a,{},inf,m)
  self.assertTrue(old['latched']);self.assertEqual(old['last'],last)
  self.assertTrue(s['stale_risk']);self.assertIsNotNone(s['selected_candidate'])
  self.assertNotEqual(s['selected_candidate']['side'],0)
  self.assertTrue(s['selected_candidate']['certificate']['passed'])
  self.assertGreater(len(r.actor_boxes()[0]),0)
 def test_missing_real_front_obstacle_still_blocks_narrow_road(self):
  r,m=self.setup_risk(mapping(1.88,-1.88),12.)
  for i in range(3,9):
   a,inf=observation(m,i*.1,empty=True);out,s=r.process(intent(m),a,{},inf,m)
  self.assertEqual(s['mode'],'verified_wait_no_feasible_corridor')
  action,g=r.admit(m,dict(acc_mps2=2.,steer_rate_radps=0.))
  self.assertTrue(action is None or action['acc_mps2']<0)
 def test_stale_inflation_never_creates_new_latch(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(3):
   a,inf=observation(m,i*.1,car=100.,lateral=5.);r.process(intent(m),a,{},inf,m)
  self.assertFalse(r.tracks[0]['latched'])
  for i in range(3,20):
   a,inf=observation(m,i*.1,empty=True);r.process(intent(m),a,{},inf,m)
   self.assertFalse(any(t['latched'] for t in r.tracks))
  self.assertEqual(len(r.tracks),0)
 def test_long_gap_reacquisition_keeps_identity_and_renews_only_real_observation(self):
  r,m=self.setup_risk();identifier=r.tracks[0]['id'];last=r.tracks[0]['last']
  for i in range(3,25):
   a,inf=observation(m,i*.1,empty=True);r.process(intent(m),a,{},inf,m)
  self.assertEqual(r.tracks[0]['last'],last)
  a,inf=observation(m,2.5,car=12.);r.process(intent(m),a,{},inf,m)
  self.assertEqual(len(r.tracks),1);self.assertEqual(r.tracks[0]['id'],identifier)
  self.assertEqual(r.tracks[0]['last'],2.5);self.assertTrue(r.memory_audit['reassociated'])
 def test_duplicate_identity_is_merged_without_double_counting_hits(self):
  r,m=self.setup_risk();duplicate=copy.deepcopy(r.tracks[0]);duplicate['id']=100;duplicate['last']=-2.;r.tracks.append(duplicate)
  a,inf=observation(m,.3,car=12.);r.process(intent(m),a,{},inf,m)
  self.assertEqual(len(r.tracks),1);self.assertEqual(len(r.tracks[0]['samples']),4)
  self.assertTrue(r.memory_audit['merged'])
 def test_nearby_distinct_cars_are_not_merged(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(3):
   a,inf=observation(m,i*.1,20.);a['boxes']=np.repeat(a['boxes'],2,axis=0);a['boxes'][1,0]=3.
   a['scores']=np.array([.8,.7]);a['labels']=np.array([0,0]);r.process(intent(m),a,{},inf,m)
  self.assertEqual(len(r.tracks),2);self.assertEqual(len(r.actor_boxes()[0]),2)
 def test_duplicate_rows_cannot_confirm_track_and_clock_reset_is_local(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.);a,inf=observation(m,0,12.)
  a={k:np.repeat(v,3,axis=0) for k,v in a.items()};r.process(intent(m),a,{},inf,m)
  self.assertEqual(len(r.tracks),1);self.assertFalse(r.tracks[0]['confirmed'])
  with self.assertRaises(ValueError):r.process(intent(m),a,{},inf,m)
  a,inf=observation(m,.1,12.);r.process(intent(m),a,{},inf,m)
  a,inf=observation(m,0,empty=True);r.process(intent(m),a,{},inf,m);self.assertEqual(len(r.tracks),0)
 def test_positive_evidence_ages_while_vehicle_stops_and_then_reacquires(self):
  r=JointReference(V,mapping(1.88,-1.88));m=Motion(.1,2.7,2.)
  for i in range(3):a,inf=observation(m,i*.1,12.);r.process(intent(m,2.),a,{},inf,m)
  r.tracks[0]['latched']=True;last=r.tracks[0]['last']
  for i in range(3,25):
   a,inf=observation(m,i*.1,empty=True);out,s=r.process(intent(m,2.),a,{},inf,m)
   action,g=r.admit(m,dict(acc_mps2=2.,steer_rate_radps=0.));self.assertTrue(action is None or action['acc_mps2']<=0)
   if action:m.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
  self.assertEqual(r.tracks[0]['last'],last)
  a,inf=observation(m,2.5,12.);r.process(intent(m,2.),a,{},inf,m);self.assertEqual(r.tracks[0]['last'],2.5)
 def test_near_zero_braking_prediction_never_reverses(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,.0045)
  a,inf=observation(m,0,empty=True);r.process(intent(m),a,{},inf,m)
  states=r.stop_states(m,dict(acc_mps2=-.045,steer_rate_radps=0.),reaction_steps=2)
  self.assertTrue(np.all(states[:,3]>=-1e-12));self.assertTrue(r.joint_certificate(states,np.empty((0,9)))['dynamics_bounds_passed'])
 def test_continuous_multi_actor_dropout_and_reacquisition_with_actual_ilqr(self):
  from shapely.geometry import Polygon
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.);tracker=Tracker(V);rows=[];gaps=[]
  for i in range(240):
   now=i*.1;a,inf=observation(m,now,20.,empty=2.5<=now<3.8)
   if now<1.:
    lead,_=observation(m,now,35.+8.*now)
    a={k:np.concatenate([a[k],lead[k]],axis=0) for k in a}
   if len(a['boxes']) and i%5==0:a={k:np.concatenate([v,v[:1]],axis=0) for k,v in a.items()}
   if i%2:a={k:v[::-1].copy() for k,v in a.items()}
   out,sel=r.process(intent(m),a,{},inf,m);nominal,control=track_reference(tracker,r,out,inf,m);action,gate=r.admit(m,nominal)
   self.assertIsNotNone(action,(i,sel['mode'],gate['selected']))
   m.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
   c,s0=np.cos(m.yaw),np.sin(m.yaw);corners=np.array([[-.8,-1.5],[.8,-1.5],[.8,1.5],[-.8,1.5]])
   ego=Polygon(corners@np.array([[c,-s0],[s0,c]])+np.array([m.x,m.z]));car=Polygon([[-1,17.75],[1,17.75],[1,22.25],[-1,22.25]])
   self.assertFalse(ego.intersects(car),(i,m.state()));gaps.append(ego.distance(car))
   pose=np.array([[c,s0,0,m.z],[-s0,c,0,-m.x],[0,0,1,0],[0,0,0,1]])
   self.assertGreater(r.geometry.pose_clearance(pose),-.01)
   rows.append(dict(index=i,state=m.state(),mode=sel['mode'],gate=gate['selected'],memory=sel['memory'],stale=sel['stale_risk'],tracks=len(r.tracks)))
   if m.z>30 and abs(m.x)<1 and abs(m.yaw)<.1 and m.speed>1:break
  write(P26/'light_tracking_curvature_v15/checks/multi_track_dropout_ilqr.json',dict(rows=rows,min_body_gap_m=min(gaps),GPU_executed=False,synthetic_observation=True))
  self.assertGreater(m.z,30.,m.state());self.assertLess(abs(m.x),1.,m.state());self.assertGreater(m.speed,1.)
  self.assertTrue(any(x['memory']['reassociated'] for x in rows))

 def test_actual_v8_freeze_history_has_admissible_routes_and_rest_restart(self):
  root=P26/'light_joint_repair_v8/B';report=read(root/'run_report.json');r=JointReference(V,read(root.parent/'map/map.json'));tracker=Tracker(V);rows=[]
  for t in report['trace']:
   i=t['index'];step=root/f'uniad_worker/step_{i:03d}'
   with (step/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   with np.load(step/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state']
   if i:m.last_accel=report['trace'][i-1]['action']['acc_mps2']
   out,s=r.process(a['trajectory'],a,{},inf,m)
   row=dict(index=i,old_speed=t['motion_state'][3],old_mode=t['road_reference']['mode'],old_stale=t['road_reference']['stale_risk'],old_valid_candidates=sum(c['passed'] for c in t['road_reference']['candidates']),new_mode=s['mode'],new_stale=s['stale_risk'],selected=s['selected_candidate'],memory=s['memory'],seconds=s['seconds'],track_count=len(r.tracks))
   if i in (23,24,25,30,43,44,45,60,100,150,187):
    nominal,c=track_reference(tracker,r,out,inf,m);action,g=r.admit(m,nominal);row.update(action=action,gate=g)
   rows.append(row)
  write(P26/'light_tracking_curvature_v15/checks/v8_stall_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False,source_report=str(root/'run_report.json')))
  selected={x['index']:x for x in rows}
  for i in (23,24,25,30,43,44):
   self.assertIsNotNone(selected[i]['selected'],selected[i])
   self.assertEqual(selected[i]['gate']['selected'],'joint_verified_tracker',selected[i])
  self.assertGreater(selected[44]['action']['acc_mps2'],0.,selected[44])

if __name__=='__main__':unittest.main(verbosity=2)
