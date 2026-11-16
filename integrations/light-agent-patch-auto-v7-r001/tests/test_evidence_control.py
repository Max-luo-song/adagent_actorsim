
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

import copy,json,pickle,sys,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,mapping,info,observation,intent,prime,JointReference,Motion,Tracker,track_reference,P26,read,write

class EvidenceControlTests(unittest.TestCase):
 def test_missing_observation_never_inflates_core_or_declares_free(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,0.)
  for i in range(4):
   a,inf=observation(m,i*.1,12.);a['boxes'][0,2]=-1.5;a['boxes'][0,3:6]=[.53,.58,1.3];a['labels'][0]=8
   r.process(intent(m),a,{},inf,m)
  b,tracks=r.actor_boxes();size=b[0,3:5].copy();last=tracks[0]['last'];self.assertFalse(tracks[0]['height_supported'])
  tracks[0]['latched']=True
  a,inf=observation(m,9.,empty=True);r.process(intent(m),a,{},inf,m);b,tracks=r.actor_boxes()
  self.assertEqual(len(b),1);np.testing.assert_allclose(b[0,3:5],size)
  self.assertEqual(tracks[0]['last'],last);self.assertTrue(tracks[0]['latched'])
  self.assertLess(r.motion_evidence[0]['soft_age_margin_m'],2.)
 def test_true_pedestrian_core_still_blocks_narrow_road_when_occluded(self):
  r=JointReference(V,mapping(1.88,-1.88));m=Motion(.1,2.7,2.)
  for i in range(4):
   a,inf=observation(m,i*.1,8.);a['boxes'][0,3:6]=[.6,.6,1.7];a['labels'][0]=8;r.process(intent(m,2.),a,{},inf,m)
  self.assertTrue(r.tracks[0]['height_supported']);r.tracks[0]['latched']=True
  a,inf=observation(m,4.,empty=True);out,s=r.process(intent(m,2.),a,{},inf,m)
  self.assertGreater(len(r.actor_boxes()[0]),0)
  # A direct crossing of the observed core never becomes free with age.
  states=r.states(np.c_[np.zeros(6),np.arange(1,7)*2.]);self.assertFalse(r.actor_certificate(states,r.actor_boxes()[0])['passed'])
 def test_motion_filter_keeps_coherent_approaching_and_receding_objects(self):
  for velocity in (-6.,-2.,8.,20.,30.):
   r=JointReference(V,mapping());m=Motion(.1,2.7,0.)
   for i in range(17):
    a,inf=observation(m,i*.1,30.+velocity*i*.1);r.observed({},inf,a);r.update_tracks(a,inf)
   self.assertEqual(len(r.tracks),1);track=r.tracks[0]
   self.assertTrue(track['motion_reliable'],(velocity,track['motion_audit']))
   self.assertAlmostEqual(track['velocity'][0],velocity,places=5)
 def test_ordinary_execution_smoothness_limits_both_acceleration_directions(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,3.35);m.last_accel=-.08
  action,audit=r.smooth_action(m,dict(acc_mps2=-1.27,steer_rate_radps=0.),ceiling=-.38)
  self.assertAlmostEqual(action['acc_mps2'],-.38);self.assertLessEqual(abs(audit['jerk_mps3']),3.+1e-8)
  m.last_accel=0.;action,audit=r.smooth_action(m,dict(acc_mps2=2.,steer_rate_radps=0.),ceiling=.3)
  self.assertAlmostEqual(action['acc_mps2'],.3)
 def test_speed_saturation_does_not_reverse_or_overspeed(self):
  r=JointReference(V,mapping())
  for speed,last in [(0.,-2.),(V.max_speed_mps,2.)]:
   m=Motion(.1,2.7,speed);m.last_accel=last
   action,audit=r.smooth_action(m,dict(acc_mps2=last,steer_rate_radps=0.))
   self.assertGreaterEqual(speed+action['acc_mps2']*.1,0.)
   self.assertLessEqual(speed+action['acc_mps2']*.1,V.max_speed_mps)
 def test_rollout_records_actual_saturated_commands(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,29.9);m.last_accel=2.
  a,inf=observation(m,0,empty=True);r.observed({},inf,a)
  guide=r.guides(intent(m,30.),np.empty((0,9)),m,30.)[0]
  states,controls=r.rollout(guide,m,30.,3.)
  for j,(acc,rate) in enumerate(controls):
   m.advance(dict(acc=acc,steer_rate=rate));np.testing.assert_allclose(states[j+1],[m.x,m.z,m.yaw,m.speed,m.steer],atol=1e-9)
   self.assertLessEqual(m.speed,V.max_speed_mps+1e-9)
 def test_warm_speed_seed_cannot_override_new_neural_stop(self):
  r=JointReference(V,mapping());r.previous_target_speed=4.
  self.assertTrue(all(v<=.1 for v in r.speed_targets(.1)))
  r.previous_target_speed=2.;self.assertAlmostEqual(r.speed_targets(4.)[1],2.2)
 def test_real_urgent_obstacle_can_override_comfort_after_safety_check(self):
  r=JointReference(V,mapping(1.88,-1.88));m=Motion(.1,2.7,4.);m.last_accel=2.
  for i in range(3):a,inf=observation(m,i*.1,10.);r.process(intent(m),a,{},inf,m)
  action,g=r.admit(m,dict(acc_mps2=2.,steer_rate_radps=0.))
  self.assertIsNotNone(action);self.assertTrue(g.get('emergency_jerk_exception'),g)
  self.assertEqual(action['acc_mps2'],V.min_acc_mps2);self.assertTrue(g['certificate']['passed'])
 def test_continuous_noisy_car_and_transient_weak_object(self):
  from shapely.geometry import Polygon
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.);tracker=Tracker(V);rows=[];gaps=[]
  for i in range(300):
   now=i*.1;a,inf=observation(m,now,20.)
   # Measurement jitter is independent of the physical collision oracle.
   a['boxes'][0,1]+=.7*np.sin(i*1.7)
   if 4.5<=now<4.9:
    ghost,_=observation(m,now,35.,lateral=-.5)
    ghost['boxes'][0,2]=-1.5;ghost['boxes'][0,3:6]=[.53,.58,1.3];ghost['labels'][0]=8
    a={k:np.concatenate([a[k],ghost[k]],axis=0) for k in a}
   out,sel=r.process(intent(m),a,{},inf,m);nominal,c=track_reference(tracker,r,out,inf,m);action,g=r.admit(m,nominal)
   self.assertIsNotNone(action,(i,sel['mode'],g['selected']))
   sm=g.get('execution_smoothness',{})
   if not g.get('emergency_jerk_exception') and not sm.get('stop_saturation_relief'):self.assertLessEqual(abs(sm['jerk_mps3']),3.+1e-7)
   m.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
   ct,st=np.cos(m.yaw),np.sin(m.yaw);corners=np.array([[-.8,-1.5],[.8,-1.5],[.8,1.5],[-.8,1.5]])
   ego=Polygon(corners@np.array([[ct,-st],[st,ct]])+np.array([m.x,m.z]));car=Polygon([[-1,17.75],[1,17.75],[1,22.25],[-1,22.25]])
   self.assertFalse(ego.intersects(car),(i,m.state()));gaps.append(ego.distance(car))
   pose=np.array([[ct,st,0,m.z],[-st,ct,0,-m.x],[0,0,1,0],[0,0,0,1]])
   self.assertGreater(r.geometry.pose_clearance(pose),-.01)
   rows.append(dict(index=i,state=m.state(),gate=g['selected'],mode=sel['mode'],selected=sel['selected_candidate'],smoothness=sm,emergency=g.get('emergency_jerk_exception',False)))
   if m.z>45 and abs(m.x)<1 and abs(m.yaw)<.1 and m.speed>1:break
  write(P26/'light_agent_patch_auto_v7_r001/checks/noisy_evidence_ilqr.json',dict(rows=rows,min_body_gap_m=min(gaps),GPU_executed=False,synthetic_observation=True))
  self.assertGreater(m.z,45.,m.state());self.assertLess(abs(m.x),1.,m.state());self.assertGreater(m.speed,1.)

 def test_actual_v9_failure_history_robust_motion_and_bounded_risk(self):
  root=P26/'light_risk_memory_v9/B';report=read(root/'run_report.json');r=JointReference(V,read(root.parent/'map/map.json'));tracker=Tracker(V);rows=[];snapshots=[]
  for t in report['trace']:
   i=t['index'];step=root/f'uniad_worker/step_{i:03d}'
   with (step/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   with np.load(step/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state']
   if i:m.last_accel=report['trace'][i-1]['action']['acc_mps2']
   out,s=r.process(a['trajectory'],a,{},inf,m);boxes,tracks=r.actor_boxes()
   detail=[dict(id=t0['id'],velocity=t0['velocity'].tolist(),motion=t0['motion_audit'],size=box[3:5].tolist(),age=r.clock-t0['last'],tier=t0['evidence_tier']) for box,t0 in zip(boxes,tracks)]
   row=dict(index=i,mode=s['mode'],selected=s['selected_candidate'],valid=s['geometrically_valid_candidates'],seconds=s['seconds'],tracks=detail)
   if i in (53,72,96,97,101,102,137,145,150,154,155,170,187):
    nominal,c=track_reference(tracker,r,out,inf,m);action,g=r.admit(m,nominal);row.update(action=action,gate=g)
   rows.append(row)
  write(P26/'light_agent_patch_auto_v7_r001/checks/v9_failure_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  chosen={r0['index']:r0 for r0 in rows}
  for i in (53,72):
   t0=next(x for x in chosen[i]['tracks'] if x['id']==2)
   self.assertGreaterEqual(t0['velocity'][0],-.1,t0);self.assertIsNotNone(chosen[i]['selected'],chosen[i])
   self.assertFalse(chosen[i]['gate'].get('emergency_jerk_exception',False),chosen[i])
  for i in (145,155,170,187):self.assertIsNotNone(chosen[i]['selected'],chosen[i])
  sizes=[x['size'] for row in rows if row['index']>=101 for x in row['tracks'] if x['id']==16]
  self.assertTrue(sizes);self.assertLess(max(max(s) for s in sizes),3.)
  for row in rows:
   if 'gate' not in row or row['action'] is None:continue
   g=row['gate'];sm=g.get('execution_smoothness',{})
   if not g.get('emergency_jerk_exception') and not sm.get('stop_saturation_relief'):self.assertLessEqual(abs(sm['jerk_mps3']),3.+1e-7,row)

if __name__=='__main__':unittest.main(verbosity=2)
