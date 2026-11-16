"""Current assistant's evidence-backed revision; no unattended LLM API claim."""
import copy
from workflow import STORE,asset,read,write,capture
old=STORE/'rounds/v11_to_v12_intent_recovery';new=STORE/'rounds/v11_to_v12_intent_recovery_r2'
replay=read(old/'apply_receipt.json');r=read(__import__('pathlib').Path(replay['output'])/'checks/v11_intent_recovery_replay.json')
events=[x for x in r['rows'] if x['intent_recovery'].get('resumed')]
gate=dict(schema='assistant_diagnostic_gate_v1',passed=any(x['index']>=145 and x['old_side'] in (-1,1) and x['new_side']==0 for x in events),
 event_frames=[x['index'] for x in events],late_event_frames=[x['index'] for x in events if x['index']>=145],
 issue='First proposal did not improve the target late-episode interval; low-speed neutral alternatives were also counted as recovery',
 replay=asset(__import__('pathlib').Path(replay['output'])/'checks/v11_intent_recovery_replay.json'),GPU_candidate_validation_allowed=False,author='acting_llm_assistant')
write(old/'diagnostic_gate.json',gate)
if gate['passed']:raise RuntimeError('Revision premise changed; inspect new evidence')
capture(new)
edit=read(old/'edit_instruction.json');edit['task_id']='v11_cutin551_intent_recovery_r2_baseline';write(new/'edit_instruction.json',edit)
f=read(old/'failure_reason.json');f.update(evidence=asset(new/'evidence.json'),previous_proposal_gate=asset(old/'diagnostic_gate.json'),
 revision_reason='First candidate improved unrelated early low-speed choices, not the target late recovery; reject it before GPU',
 proposed_rule='Recover only fully certified side0 candidates at the CURRENT neural cruise target and whose preview clears the conflict. Require the existing confirmation_hits and confirmation_span_s on distinct observations, resetting on lost feasibility or clock/scene reset.',
 acceptance_claim='candidate revision2 only; held-out and GPU evidence still required')
f['evidence_links'].append(asset(old/'diagnostic_gate.json'));write(new/'failure_reason.json',f)
p=read(old/'patch_instruction.json');p.update(candidate_directory='light-intent-recovery-v12-r2',output_namespace='light_intent_recovery_v12_r2',failure_reason=asset(new/'failure_reason.json'),revision=2)
helper='''    def intent_pool(self,candidates):
        # Recovery is the current cruise intention, not merely a slower wait.
        if self.intent_last_clock is not None and self.clock<=self.intent_last_clock:
            self.intent_clear_since=None;self.intent_clear_observations=0
        self.intent_last_clock=self.clock
        current=[c for c in candidates if c[4]['side']==0 and c[4]['passed']
            and c[4].get('current_intent_cruise',False) and c[4]['clear_conflict_in_preview']]
        eligible=self.side in (-1,1) and bool(current)
        if not eligible:self.intent_clear_since=None;self.intent_clear_observations=0
        elif self.intent_clear_since is None:self.intent_clear_since=self.clock;self.intent_clear_observations=1
        else:self.intent_clear_observations+=1
        stable=eligible and self.intent_clear_observations>=POLICY['confirmation_hits'] and self.clock-self.intent_clear_since>=POLICY['confirmation_span_s']-1e-9
        self.intent_recovery_audit=dict(eligible=bool(eligible),clear_since_s=self.intent_clear_since,
            resumed=bool(stable),hold_s=POLICY['confirmation_span_s'],observations=self.intent_clear_observations,
            fully_feasible_candidates=len(candidates),current_intent_candidates=len(current),
            source='current neural cruise intent; no initial-frame lane anchor')
        return current if stable else candidates

'''
ops=[]
for op in p['operations']:
 op=copy.deepcopy(op)
 if op['path']=='joint_safety.py' and op['before'].startswith('side_switch_cost='):continue
 if op['path']=='joint_safety.py' and op['before']=='    def process(self,raw,arrays,obs,info,motion):':op['after']=helper+op['before']
 if op['path']=='joint_safety.py' and op['before'].startswith('super().__init__'):op['after']+=';self.intent_clear_observations=0'
 if op['path']=='risk_memory.py':op['after']+=';self.intent_clear_observations=0'
 if op['op']=='create_file':
  text=op['content'].replace("dict(side=side,passed=True)","dict(side=side,passed=True,current_intent_cruise=True,clear_conflict_in_preview=True)")
  text=text.replace("for t in (0.,.1,.4):", "for t in (0.,.1):").replace('r.clock=.5;','r.clock=.2;')
  text=text.replace("r.clock=1.1;self.assertEqual(len(r.intent_pool(pool)),1)","r.clock=.7;r.intent_pool(pool);r.clock=.8;self.assertEqual(len(r.intent_pool(pool)),1)")
  text=text.replace("x['old_side'] in (-1,1) and x['new_side']==0", "x['index']>=145 and x['old_side'] in (-1,1) and x['new_side']==0")
  text=text.replace("light_intent_recovery_v12/checks/", "light_intent_recovery_v12_r2/checks/")
  extra=''' def test_slow_wait_or_incomplete_preview_never_certifies_recovery(self):
  for field in ('current_intent_cruise','clear_conflict_in_preview'):
   r=self.planner();neutral=c(0);neutral[4][field]=False;pool=[c(-1),neutral]
   for t in (0.,.1,.2,.5):r.clock=t;self.assertEqual(r.intent_pool(pool),pool)
'''
  text=text.replace(' def test_no_feasible_candidate_remains_empty(self):',extra+' def test_no_feasible_candidate_remains_empty(self):')
  op['content']=text
 ops.append(op)
ops.append(dict(op='replace_exact',path='joint_safety.py',before='item=dict(velocity_continuity_cost=',after='item=dict(current_intent_cruise=bool(abs(speed-target)<1e-8),velocity_continuity_cost='))
p['operations']=ops;write(new/'patch_instruction.json',p)
write(new/'assistant_decision_receipt.json',dict(author=f['author'],previous_decision=asset(old/'assistant_decision_receipt.json'),previous_diagnostic_gate=asset(old/'diagnostic_gate.json'),evidence=asset(new/'evidence.json'),edit_instruction=asset(new/'edit_instruction.json'),failure_reason=asset(new/'failure_reason.json'),patch_instruction=asset(new/'patch_instruction.json'),note='Current assistant revised the proposal after inspecting actual first-proposal replay; no user diagnosis and no external model API used'))
print(new)
