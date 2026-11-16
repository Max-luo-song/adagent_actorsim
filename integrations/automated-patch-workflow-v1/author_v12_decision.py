"""Persist this conversation assistant's v12 decision; not an LLM inference engine."""

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

import json,sys
pass  # Path is provided by the portable boundary above
from workflow import STORE,BASE,RELEASE,asset,read,write
round=STORE/'rounds/v11_to_v12_intent_recovery'
if not (round/'evidence.json').exists():raise RuntimeError('Capture evidence first')
if (round/'failure_reason.json').exists():raise FileExistsError('Assistant decision already recorded')
write(round/'edit_instruction.json',dict(schema='scene_task_v2',task_id='v11_cutin551_intent_recovery_baseline',scene_id='551',
 description='Reproduce the frozen v11 cut-in episode, same native vehicle/program and observation window; no hand-edited controls',
 edits=[dict(operation='set_actor_trajectory',object_id='native:RigidNodes:3',program_id='scene551_native3_cutin_stop_v11')],
 simulation=dict(driver='uniad_closed_loop',profile='risk_provenance_v11_diagnostic',control_steps=188),reset_after=True,weather='clear',evaluation_mode='diagnostic'))
f=dict(schema='failure_reason_v1',author=dict(role='acting_llm_assistant',identity='Codex in current conversation',external_model_API_called=False),base_version='v11',candidate_version='v12',
 evidence=asset(round/'evidence.json'),evidence_links=[asset(BASE/'joint_safety.py'),asset(RELEASE/'B/run_report.json')],
 decision='candidate_patch',failure_class='current_intent_recovery_policy_limitation',
 observed_facts=['v11 completed 188 controls without a stop; final lateral offset about 2.58m and yaw about 0.0665rad',
 'At saved step150, at least one side0 candidate passed full and prefix certificates while selected side remained -1',
 'The same side_switch_cost is charged for recovering side0 and for selecting another bypass side'],
 supported_mechanism='The candidate selector has no explicit transition back to current intent after sustained feasibility. A bypass can retain a better weighted score even after current intent is feasible.',
 scope='This supports a policy improvement, not a finding of unsafe driving. Lateral offset alone does not prove failure; do not impose a world-position return target.',
 unresolved=['18.8s window may be too short to complete a gradual return','Current-frame neural reference changes under feedback; saved-state replay cannot prove new closed-loop improvement','Midcourse deceleration may be valid for clearance/curvature; no separate speed patch is justified yet'],
 proposed_rule='After current-intent candidates remain fully feasible for 0.5s, select among those certified candidates. Reset the hold on lost feasibility or scene/clock reset. Preserve all physical, road and actor checks.',
 required_causal_validation=['v11 saved-history counterfactual selection test','stable/transient feasibility and clock-reset tests','existing true-obstacle, mirrored-road, continuous actual-iLQR and emergency regressions'],
 acceptance_claim='candidate only; no GPU or global acceptance claimed')
write(round/'failure_reason.json',f)
helper='''    def intent_pool(self,candidates):
        # Called once per actual observation, on ALREADY certified candidates.
        if self.intent_last_clock is not None and self.clock<=self.intent_last_clock:
            self.intent_clear_since=None
        self.intent_last_clock=self.clock
        current=[c for c in candidates if c[4]['side']==0 and c[4]['passed']]
        eligible=self.side in (-1,1) and bool(current)
        if not eligible:self.intent_clear_since=None
        elif self.intent_clear_since is None:self.intent_clear_since=self.clock
        stable=eligible and self.clock-self.intent_clear_since>=POLICY['intent_recovery_hold_s']-1e-9
        self.intent_recovery_audit=dict(eligible=bool(eligible),clear_since_s=self.intent_clear_since,
            resumed=bool(stable),hold_s=POLICY['intent_recovery_hold_s'],
            fully_feasible_candidates=len(candidates),current_intent_candidates=len(current),
            source='current neural intent; no initial-frame lane anchor')
        return current if stable else candidates

'''
operations=[]
def replace(path,before,after):operations.append(dict(op='replace_exact',path=path,before=before,after=after))
replace('joint_safety.py','side_switch_cost=2.,soft_risk_weight=2.)','side_switch_cost=2.,soft_risk_weight=2.,intent_recovery_hold_s=.5)')
replace('joint_safety.py','super().__init__(vehicle,mapping);self.side=None;self.selected=None;self.previous_target_speed=None','super().__init__(vehicle,mapping);self.side=None;self.selected=None;self.previous_target_speed=None;self.intent_clear_since=None;self.intent_last_clock=None;self.intent_recovery_audit={}')
replace('joint_safety.py','    def process(self,raw,arrays,obs,info,motion):',helper+'    def process(self,raw,arrays,obs,info,motion):')
replace('joint_safety.py',"            if candidates:\n", "            candidates=self.intent_pool(candidates)\n            if candidates:\n")
replace('joint_safety.py',"adopted=raw.copy();mode='native_preserved'", "adopted=raw.copy();mode='native_preserved'\n            self.intent_clear_since=None;self.intent_last_clock=self.clock;self.intent_recovery_audit=dict(resumed=False,reason='native_reference_preserved')")
replace('joint_safety.py',"audit=dict(source='joint_intent_recovery_v12',mode=mode,", "audit=dict(source='joint_intent_recovery_v12',mode=mode,intent_recovery=self.intent_recovery_audit,")
replace('risk_memory.py',"if reset:self.tracks=[];self.side=None;self.selected=None;self.previous_target_speed=None", "if reset:self.tracks=[];self.side=None;self.selected=None;self.previous_target_speed=None;self.intent_clear_since=None;self.intent_last_clock=None")
replace('cpu_check.py'," check('risk_provenance_tests'", " check('intent_recovery_tests',[sys.executable,'-B',str(LOCAL/'tests/test_intent_recovery.py')]);check('risk_provenance_tests'")
replace('cpu_check.py',"for name in ('risk_provenance_tests'", "for name in ('intent_recovery_tests','risk_provenance_tests'")
replace('pipeline.py',"old_B_summary=read(P26/'light_evidence_control_v10/B/summary.json')","old_B_summary=read(P26/'light_risk_provenance_v11/B/summary.json')")
replace('pipeline.py',"render_pair(root,P26/'light_evidence_control_v10/B',root/'B',","render_pair(root,P26/'light_risk_provenance_v11/B',root/'B',")
replace('pipeline.py',"{'off':read(P26/'light_evidence_control_v10/B/run_report.json')","{'off':read(P26/'light_risk_provenance_v11/B/run_report.json')")
replace('compare_native.py','v11 risk provenance','v12 current-intent recovery')
replace('compare_native.py','v10 evidence and smooth control','v11 risk provenance')
replace('compare_native.py','risk_provenance_v10_v11_3CAM.mp4','intent_recovery_v11_v12_3CAM.mp4')
test='''import sys,pickle,unittest
from pathlib import Path
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,mapping,Motion,JointReference,P26,read,write

def c(side,score=0.):return (score,None,None,None,dict(side=side,passed=True))
class IntentRecoveryTests(unittest.TestCase):
 def planner(self):
  r=JointReference(V,mapping());r.side=-1;return r
 def test_requires_sustained_feasibility(self):
  r=self.planner();pool=[c(-1,-3),c(0,-1)]
  for t in (0.,.1,.4):r.clock=t;self.assertEqual(len(r.intent_pool(pool)),2)
  r.clock=.5;self.assertEqual([x[4]['side'] for x in r.intent_pool(pool)],[0])
 def test_lost_feasibility_resets_hold(self):
  r=self.planner();pool=[c(-1),c(0)];r.clock=0;r.intent_pool(pool);r.clock=.4;r.intent_pool([c(-1)]);r.clock=.6
  self.assertEqual(len(r.intent_pool(pool)),2);r.clock=1.1;self.assertEqual(len(r.intent_pool(pool)),1)
 def test_clock_reset_does_not_reuse_hold(self):
  r=self.planner();pool=[c(-1),c(0)];r.clock=10;r.intent_pool(pool);r.clock=0
  self.assertEqual(len(r.intent_pool(pool)),2)
 def test_native_mode_and_opposite_bypass_are_not_forced(self):
  r=self.planner();r.side=0;r.clock=0;pool=[c(-1),c(0),c(1)];self.assertEqual(r.intent_pool(pool),pool)
  r.side=-1;r.clock=1;pool=[c(-1),c(1)];self.assertEqual(r.intent_pool(pool),pool)
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
  write(P26/'light_intent_recovery_v12/checks/v11_intent_recovery_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  self.assertTrue(any(x['old_side'] in (-1,1) and x['new_side']==0 and x['intent_recovery'].get('resumed') for x in rows))
if __name__=='__main__':unittest.main(verbosity=2)
'''
operations.append(dict(op='create_file',path='tests/test_intent_recovery.py',content=test))
p=dict(schema='patch_instruction_v1',author=f['author'],base_version='v11',candidate_version='v12',candidate_directory='light-intent-recovery-v12',output_namespace='light_intent_recovery_v12',failure_reason=asset(round/'failure_reason.json'),
 operations=operations,acceptance=dict(causal_tests_required=True,matched_closed_loop_required=True,normal_scene_nonregression_required=True,cross_scene_nonregression_required=True),
 protected_scope=['original collision/road/vehicle/dynamics certificates','all current and retained obstacle geometry','native worker/model/renderer/scenario','v11 source and results'],
 known_limits='Recover CURRENT intent, not an initial-frame route. No full-return or score improvement claim before new GPU evidence.')
write(round/'patch_instruction.json',p)
write(round/'assistant_decision_receipt.json',dict(author=f['author'],evidence=asset(round/'evidence.json'),edit_instruction=asset(round/'edit_instruction.json'),failure_reason=asset(round/'failure_reason.json'),patch_instruction=asset(round/'patch_instruction.json'),note='Authored by the current assistant in this turn; this writer does not perform model inference'))
print(round)
