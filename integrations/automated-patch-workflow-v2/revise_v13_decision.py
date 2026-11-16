from workflow import STORE,P26,BASE,asset,read,write,capture
old=STORE/'rounds/v12_to_v13_reference_join';new=STORE/'rounds/v12_to_v13_reference_join_r2'
proof=P26/'light_reference_join_v13/checks/evidence_control_tests.log'
if 'FAILED (failures=1)' not in proof.read_text():raise ValueError('Review unexpected regression evidence')
write(old/'diagnostic_gate.json',dict(schema='assistant_diagnostic_gate_v1',passed=False,reason='Existing continuous noisy-car test fails: boundary-jet extrapolation can reverse lateral intent at large initial slopes',evidence=asset(proof),GPU_candidate_validation_allowed=False,source_v12_changed=False))
capture(new)
edit=read(old/'edit_instruction.json');edit['task_id']='v12_reference_join_r2_baseline';write(new/'edit_instruction.json',edit)
f=read(old/'failure_reason.json');f['evidence']=asset(new/'evidence.json');f['evidence_links']+=[asset(old/'diagnostic_gate.json'),asset(proof)];f['previous_proposal_gate']=asset(old/'diagnostic_gate.json');f['revision_reason']='Old regression exposed sign reversal under a large first-segment slope; bound only the residual path proposal without changing any safety checks.';f['proposed_rule']='Use boundary mismatch correction, with its lateral component pointwise confined between the original join and the current bounded NN reference; add the unchanged physical initial-curvature term afterward.';write(new/'failure_reason.json',f)
p=read(old/'patch_instruction.json');p.update(candidate_directory='light-reference-join-v13-r2',output_namespace='light_reference_join_v13_r2',failure_reason=asset(new/'failure_reason.json'),revision=2)
for op in p['operations']:
 if op['path']=='reference_join.py':
  op['content']=op['content'].replace("    return raw-(offset+slope*distance)*(1-blend)+.5*curvature*distance**2*(1-u)**3", "    previous=origin[0]+(raw-origin[0])*blend\n    corrected=raw-(offset+slope*distance)*(1-blend)\n    # A boundary correction must not reverse or amplify the NN residual.\n    position=np.clip(corrected,np.minimum(previous,raw),np.maximum(previous,raw))\n    return position+.5*curvature*distance**2*(1-u)**3")
 if op['path']=='tests/test_reference_join.py':
  text=op['content'].replace('light_reference_join_v13/checks/','light_reference_join_v13_r2/checks/')
  extra=''' def test_large_initial_slope_cannot_reverse_current_intent(self):
  y=self.grid();x=np.interp(y,[0,2,30],[0,3,3]);out=join_reference(x,y,[0,0],12.,0.);previous=self.old(x,y)
  self.assertTrue(np.all(out>=previous-1e-12));self.assertTrue(np.all(out<=x+1e-12))
  self.assertGreater(out[np.argmin(abs(y-5.))],0.)
 def test_residual_stays_between_previous_and_current_curve(self):
  y=self.grid()
  for sign in (-1,1):
   x=sign*(.05*np.sin(y)+.2*(1-np.exp(-np.maximum(y-1,0)**2)));out=join_reference(x,y,[0,0],12.,0.);previous=self.old(x,y)
   self.assertTrue(np.all(out>=np.minimum(previous,x)-1e-12));self.assertTrue(np.all(out<=np.maximum(previous,x)+1e-12))
'''
  text=text.replace(' def test_invalid_inputs_are_rejected(self):',extra+' def test_invalid_inputs_are_rejected(self):');op['content']=text
write(new/'patch_instruction.json',p);write(new/'assistant_decision_receipt.json',dict(author=f['author'],previous_decision=asset(old/'assistant_decision_receipt.json'),previous_failure=asset(old/'diagnostic_gate.json'),evidence=asset(new/'evidence.json'),edit_instruction=asset(new/'edit_instruction.json'),failure_reason=asset(new/'failure_reason.json'),patch_instruction=asset(new/'patch_instruction.json')))
print(new)
