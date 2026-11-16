
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

import sys,pickle,unittest,copy
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,mapping,info,observation,intent,JointReference,Motion,P26,read,write

class ProvenanceTests(unittest.TestCase):
 def test_rejected_bypass_cannot_lock_peripheral_car(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(4):
   a,inf=observation(m,i*.1,20.);b,_=observation(m,i*.1,20.,lateral=-5.5)
   a={k:np.concatenate([a[k],b[k]]) for k in a};r.process(intent(m),a,{},inf,m)
  front,side=sorted(r.tracks,key=lambda x:abs(x['world'][1]))
  self.assertTrue(front['latched']);self.assertFalse(side['latched'])
  side_id=side['id'];front_id=front['id']
  a,inf=observation(m,1.5,empty=True);r.process(intent(m),a,{},inf,m)
  self.assertNotIn(side_id,[t['id'] for t in r.tracks]);self.assertIn(front_id,[t['id'] for t in r.tracks])
  self.assertTrue(any(x['id']==side_id and not x['free_space_claim'] for x in r.memory_audit['retired']))
 def test_frozen_v10_reproduces_speculative_latch_with_identical_observations(self):
  import importlib.util
  path=LOCAL.parent/'light-evidence-control-v10/joint_safety.py';spec=importlib.util.spec_from_file_location('frozen_v10_joint',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  previous=old.JointReference(V,mapping());current=JointReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(4):
   a,inf=observation(m,i*.1,20.);b,_=observation(m,i*.1,20.,lateral=-5.5)
   a={k:np.concatenate([a[k],b[k]]) for k in a};saved={k:v.copy() for k,v in a.items()}
   previous.process(intent(m),a,{},inf,m);current.process(intent(m),a,{},inf,m)
   for k in a:np.testing.assert_array_equal(a[k],saved[k])
  old_side=max(previous.tracks,key=lambda x:abs(x['world'][1]));new_side=max(current.tracks,key=lambda x:abs(x['world'][1]))
  self.assertTrue(old_side['latched']);self.assertFalse(new_side['latched'])
  old_boxes,_=previous.actor_boxes();new_boxes,_=current.actor_boxes();np.testing.assert_allclose(old_boxes,new_boxes)
 def test_side_obstacle_on_current_neural_intent_keeps_latch(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(4):
   a,inf=observation(m,i*.1,20.,lateral=-5.5)
   raw=np.c_[np.linspace(-1.,-5.5,6),np.arange(1,7)*2.]
   r.process(raw,a,{},inf,m)
  self.assertTrue(r.tracks[0]['latched'])
  a,inf=observation(m,2.,empty=True);r.process(raw,a,{},inf,m)
  self.assertGreater(len(r.actor_boxes()[0]),0)
 def test_unselected_side_geometry_remains_collision_checked_while_current(self):
  r=JointReference(V,mapping());m=Motion(.1,2.7,4.)
  for i in range(4):
   a,inf=observation(m,i*.1,20.,lateral=-5.5);r.process(intent(m),a,{},inf,m)
  boxes,_=r.actor_boxes();self.assertEqual(len(boxes),1)
  states=r.states(np.c_[np.full(6,-5.5),np.arange(1,7)*4.])
  self.assertFalse(r.actor_certificate(states,boxes)['passed'])
 def test_real_v10_history_recovers_late_candidates_without_id_filter(self):
  p=P26/'light_evidence_control_v10/B';report=read(p/'run_report.json');r=JointReference(V,read(p.parent/'map/map.json'));rows=[]
  for t in report['trace']:
   i=t['index'];folder=p/f'uniad_worker/step_{i:03d}'
   with (folder/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   with np.load(folder/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state']
   m.last_accel=0 if i==0 else report['trace'][i-1]['action']['acc_mps2']
   out,s=r.process(a['trajectory'],a,{},inf,m)
   rows.append(dict(index=i,valid=s['admitted_candidates'],mode=s['mode'],selected=s['selected_candidate'],memory=s['memory'],tracks=[dict(id=x['id'],last=x['last'],latched=x['latched'],source=x.get('latch_source')) for x in r.tracks],seconds=s['seconds']))
  write(P26/'light_execution_horizon_v17_r2/checks/v10_failure_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  for i in (88,100,110,145,187):self.assertIsNotNone(rows[i]['selected'],rows[i])
  self.assertTrue(any(x['id']==10 for x in rows[20]['tracks']))
  self.assertFalse(any(x['id']==10 for x in rows[40]['tracks']))

if __name__=='__main__':unittest.main(verbosity=2)
