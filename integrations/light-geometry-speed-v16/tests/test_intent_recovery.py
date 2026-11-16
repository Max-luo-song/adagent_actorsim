
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

import sys,pickle,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,mapping,Motion,JointReference,P26,read,write

def c(side,score=0.):return (score,None,None,None,dict(side=side,passed=True,current_intent_cruise=True,clear_conflict_in_preview=True))
class IntentRecoveryTests(unittest.TestCase):
 def planner(self):
  r=JointReference(V,mapping());r.side=-1;return r
 def test_requires_sustained_feasibility(self):
  r=self.planner();pool=[c(-1,-3),c(0,-1)]
  for t in (0.,.1):r.clock=t;self.assertEqual(len(r.intent_pool(pool)),2)
  r.clock=.2;self.assertEqual([x[4]['side'] for x in r.intent_pool(pool)],[0])
 def test_lost_feasibility_resets_hold(self):
  r=self.planner();pool=[c(-1),c(0)];r.clock=0;r.intent_pool(pool);r.clock=.4;r.intent_pool([c(-1)]);r.clock=.6
  self.assertEqual(len(r.intent_pool(pool)),2);r.clock=.7;r.intent_pool(pool);r.clock=.8;self.assertEqual(len(r.intent_pool(pool)),1)
 def test_clock_reset_does_not_reuse_hold(self):
  r=self.planner();pool=[c(-1),c(0)];r.clock=10;r.intent_pool(pool);r.clock=0
  self.assertEqual(len(r.intent_pool(pool)),2)
 def test_native_mode_and_opposite_bypass_are_not_forced(self):
  r=self.planner();r.side=0;r.clock=0;pool=[c(-1),c(0),c(1)];self.assertEqual(r.intent_pool(pool),pool)
  r.side=-1;r.clock=1;pool=[c(-1),c(1)];self.assertEqual(r.intent_pool(pool),pool)
 def test_slow_wait_or_incomplete_preview_never_certifies_recovery(self):
  for field in ('current_intent_cruise','clear_conflict_in_preview'):
   r=self.planner();neutral=c(0);neutral[4][field]=False;pool=[c(-1),neutral]
   for t in (0.,.1,.2,.5):r.clock=t;self.assertEqual(r.intent_pool(pool),pool)
 def test_no_feasible_candidate_remains_empty(self):
  r=self.planner();r.clock=0;self.assertEqual(r.intent_pool([]),[])
 def test_real_v11_saved_history_recovers_current_intent_earlier(self):
  p=P26/'light_risk_provenance_v11/B';report=read(p/'run_report.json');r=JointReference(V,read(p.parent/'map/map.json'));rows=[]
  for t in report['trace']:
   i=t['index'];step=p/f'uniad_worker/step_{i:03d}'
   with (step/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
   with np.load(step/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
   m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state'];m.last_accel=0 if not i else report['trace'][i-1]['action']['acc_mps2']
   out,s=r.process(a['trajectory'],a,{},inf,m);old=t['road_reference']['selected_candidate'];new=s['selected_candidate']
   if new:self.assertTrue(new['certificate']['passed'] and new['prefix_certificate']['passed'])
   rows.append(dict(index=i,old_side=None if old is None else old['side'],new_side=None if new is None else new['side'],intent_recovery=s['intent_recovery'],selected=new))
  write(P26/'light_geometry_speed_v16/checks/v11_intent_recovery_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  self.assertTrue(any(x['index']>=145 and x['old_side'] in (-1,1) and x['new_side']==0 and x['intent_recovery'].get('resumed') for x in rows))
if __name__=='__main__':unittest.main(verbosity=2)
