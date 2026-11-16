"""Persist the current conversation assistant's next-round decision, not a model API."""

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

pass  # Path is provided by the portable boundary above
from workflow import STORE,BASE,RELEASE,RUN,asset,read,write
r=STORE/'rounds/v12_to_v13_reference_join'
if (r/'patch_instruction.json').exists():raise FileExistsError('Decision already recorded')
diagnostic=read(r/'reference_join_counterfactual.json')
write(r/'edit_instruction.json',dict(schema='scene_task_v2',task_id='v12_reference_join_baseline',scene_id='551',description='Same frozen native vehicle cut-in; v12 baseline, fresh history when explicitly reproduced',edits=[dict(operation='set_actor_trajectory',object_id='native:RigidNodes:3',program_id='scene551_native3_cutin_stop_v12')],simulation=dict(driver='uniad_closed_loop',profile='intent_recovery_v12_diagnostic',control_steps=188),reset_after=True,weather='clear',evaluation_mode='diagnostic'))
f=dict(schema='failure_reason_v1',author=dict(role='acting_llm_assistant',identity='Codex current conversation',external_model_API_called=False),base_version='v12',candidate_version='v13',evidence=asset(r/'evidence.json'),evidence_links=[asset(r/'diagnose.py'),asset(r/'reference_join_counterfactual.json'),asset(BASE/'joint_safety.py')],decision='candidate_patch',failure_class='repeated_current_reference_attenuation',
 observed_facts=['v12 completes the edited episode but ends about 2.54m left of its starting lateral coordinate','After the obstacle is passed, current neural nodes request a gradual rightward change','guides() multiplies the entire lateral reference by smooth(distance/join_length) on every observation','Single-factor saved-state probes reduce current-reference interpolation error and preserve certified actions'],
 supported_mechanism='Whole-curve multiplication suppresses higher-order shape already present in the current NN reference, although only origin value/tangent continuity needs correction. Receding replanning repeats that attenuation at each current origin.',
 proposed_rule='Preserve the bounded current NN curve; subtract only its local origin offset and first spatial slope with a compact boundary blend. Keep the existing current-steering curvature term, blend length, bounds, rollouts and certificates.',
 unresolved=['This is a confirmed reference distortion, not proof of the unique cause of the remaining lateral offset','Current NN intention and finite source time window may limit full return','No lane-center or initial-frame anchor is available or added','Reference error is diagnostic only; native scores and safety in new closed loops decide acceptance'],
 required_causal_validation=['Exact linear-reference equivalence and unchanged far reference','Left/right mirror symmetry, curvature and invalid-input cases','Real v12 history with actual iLQR, all certificates unchanged','All existing regressions, then v12 versus v13 in original and cut-in cases'],acceptance_claim='candidate only; no GPU improvement claimed')
write(r/'failure_reason.json',f)
helper='''"""Current-reference boundary correction; no world/route history or scene IDs."""
import numpy as np

def join_reference(raw,y,origin,length,curvature):
    raw=np.asarray(raw,float);y=np.asarray(y,float);origin=np.asarray(origin,float)
    if raw.ndim!=1 or raw.shape!=y.shape or len(y)<2 or origin.shape!=(2,) or not np.isfinite(raw).all() or not np.isfinite(y).all() or not np.isfinite(origin).all() or not np.isfinite([length,curvature]).all() or length<=0 or np.any(np.diff(y)<=0):raise ValueError('Invalid current-reference join')
    distance=y-origin[1];u=np.clip(distance/length,0.,1.);blend=u*u*u*(10+u*(-15+6*u))
    offset=raw[0]-origin[0];slope=(raw[1]-raw[0])/(y[1]-y[0])
    # Correct boundary mismatch, not the entire current trajectory shape.
    return raw-(offset+slope*distance)*(1-blend)+.5*curvature*distance**2*(1-u)**3
'''
test='''import sys,pickle,unittest
from pathlib import Path
import numpy as np
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from reference_join import join_reference
from test_repair import V,Motion,JointReference,P26,read,write,Tracker,track_reference
class ReferenceJoinTests(unittest.TestCase):
 def grid(self):return np.arange(0,30.001,.1)
 def old(self,x,y,length=12.,curvature=0.):
  u=np.clip(y/length,0,1);s=u**3*(10-15*u+6*u*u);return x*s+.5*curvature*y*y*(1-u)**3
 def test_linear_reference_matches_existing_join(self):
  y=self.grid()
  for slope in (-.1,0.,.1):
   for k in (-.02,0.,.02):np.testing.assert_allclose(join_reference(slope*y,y,[0,0],12.,k),self.old(slope*y,y,curvature=k),atol=1e-12)
 def test_far_reference_is_preserved_and_origin_is_current(self):
  y=self.grid();x=.03+.01*y+.2*(1-np.exp(-np.maximum(y-1,0)**2))
  out=join_reference(x,y,[0,0],12.,.02);self.assertAlmostEqual(out[0],0.);np.testing.assert_array_equal(out[y>=12],x[y>=12])
 def test_curved_shape_is_not_repeatedly_attenuated(self):
  y=self.grid();x=.2*(1-np.exp(-np.maximum(y-1,0)**2));new=join_reference(x,y,[0,0],12.,0.)
  mask=(y>2)&(y<10);self.assertLess(np.mean((new[mask]-x[mask])**2),np.mean((self.old(x,y)[mask]-x[mask])**2)*.01)
 def test_mirror_symmetry_and_input_immutability(self):
  y=self.grid();x=.01*y+.1*np.sin(y/3);saved=x.copy();a=join_reference(x,y,[0,0],12.,.02);b=join_reference(-x,y,[0,0],12.,-.02)
  np.testing.assert_allclose(a,-b,atol=1e-12);np.testing.assert_array_equal(x,saved)
 def test_zero_reference_and_current_curvature_are_consistent(self):
  y=self.grid();np.testing.assert_array_equal(join_reference(np.zeros_like(y),y,[0,0],12.,0.),np.zeros_like(y))
  np.testing.assert_allclose(join_reference(np.zeros_like(y),y,[0,0],12.,.02),self.old(np.zeros_like(y),y,curvature=.02))
 def test_invalid_inputs_are_rejected(self):
  for x,y,l in [([0,float('nan')],[0,1],2),([0,1],[0,0],2),([0,1],[0,1],0),([0],[0],2)]:
   with self.assertRaises(ValueError):join_reference(x,y,[0,0],l,0.)
 def test_real_v12_late_reference_and_execution(self):
  import importlib.util,copy
  base=LOCAL.parent/'light-intent-recovery-v12-r2';spec=importlib.util.spec_from_file_location('_baseline_v12_join',base/'joint_safety.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  p=P26/'patch_v12_validation_r2_v1/cases/cutin551/on/B';report=read(p/'run_report.json');r=module.JointReference(V,read(p.parent/'map/map.json'));rows=[]
  for t in report['trace']:
   i=t['index'];folder=p/f'uniad_worker/step_{i:03d}'
   with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
   with np.load(folder/'perception.npz') as z:arrays={k:z[k].copy() for k in z.files}
   m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state'];m.last_accel=0 if not i else report['trace'][i-1]['action']['acc_mps2']
   prior=copy.deepcopy(r) if i in (152,154,160,165,170,175,180,187) else None
   old,selection=r.process(arrays['trajectory'],arrays,{},info,m)
   if prior is None:continue
   # Same pre-observation state and inputs; only guides() implementation differs.
   prior.guides=JointReference.guides.__get__(prior,type(prior));new,selected=prior.process(arrays['trajectory'],arrays,{},info,m)
   values=[]
   for obj,out,sel in ((r,old,selection),(prior,new,selected)):
    self.assertIsNotNone(sel['selected_candidate']);self.assertTrue(sel['selected_candidate']['certificate']['passed']);self.assertTrue(sel['selected_candidate']['prefix_certificate']['passed'])
    boxes,_=obj.actor_boxes();guide=obj.guides(arrays['trajectory'],boxes,m,sel['selected_candidate']['speed_mps'])[0]
    knots=np.r_[-obj.mount[1,3],arrays['trajectory'][:,1]];xs=np.r_[-obj.mount[0,3],arrays['trajectory'][:,0]];y=np.linspace(knots[0],knots[-1],100)
    rms=float(np.sqrt(np.mean((np.interp(y,guide['y'],guide['x'])-np.interp(y,knots,xs))**2)))
    nominal,control=track_reference(Tracker(V),obj,out,info,m);action,gate=obj.admit(m,nominal);self.assertIsNotNone(action)
    values.append(dict(rms_m=rms,action=action,gate=gate['selected']))
   rows.append(dict(index=i,baseline=values[0],candidate=values[1]))
  write(P26/'light_reference_join_v13/checks/v12_reference_join_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  self.assertTrue(rows);self.assertLess(np.mean([x['candidate']['rms_m'] for x in rows]),np.mean([x['baseline']['rms_m'] for x in rows])*.5)
if __name__=='__main__':unittest.main(verbosity=2)
'''
ops=[]
def replace(path,before,after):ops.append(dict(op='replace_exact',path=path,before=before,after=after))
ops.append(dict(op='create_file',path='reference_join.py',content=helper))
replace('joint_safety.py','from observation_evidence import EVIDENCE_POLICY','from observation_evidence import EVIDENCE_POLICY\nfrom reference_join import join_reference')
replace('joint_safety.py','base=origin[0]+(base-origin[0])*smooth(u)+.5*initial_curvature*join_length**2*u*u*(1-u)**3','base=join_reference(base,y,origin,join_length,initial_curvature)')
replace('cpu_check.py'," check('intent_recovery_tests'", " check('reference_join_tests',[sys.executable,'-B',str(LOCAL/'tests/test_reference_join.py')]);check('intent_recovery_tests'")
replace('cpu_check.py',"for name in ('intent_recovery_tests'", "for name in ('reference_join_tests','intent_recovery_tests'")
for before,after in [("old_B_summary=read(P26/'light_risk_provenance_v11/B/summary.json')","old_B_summary=read(P26/'patch_v12_validation_r2_v1/cases/cutin551/on/B/summary.json')"),("render_pair(root,P26/'light_risk_provenance_v11/B',root/'B',","render_pair(root,P26/'patch_v12_validation_r2_v1/cases/cutin551/on/B',root/'B',"),("{'off':read(P26/'light_risk_provenance_v11/B/run_report.json')","{'off':read(P26/'patch_v12_validation_r2_v1/cases/cutin551/on/B/run_report.json')")]:replace('pipeline.py',before,after)
replace('compare_native.py','v12 current-intent recovery','v13 reference join')
replace('compare_native.py','v11 risk provenance','v12 current-intent recovery')
replace('compare_native.py','intent_recovery_v11_v12_3CAM.mp4','reference_join_v12_v13_3CAM.mp4')
ops.append(dict(op='create_file',path='tests/test_reference_join.py',content=test))
write(r/'patch_instruction.json',dict(schema='patch_instruction_v1',author=f['author'],base_version='v12',candidate_version='v13',candidate_directory='light-reference-join-v13',output_namespace='light_reference_join_v13',failure_reason=asset(r/'failure_reason.json'),operations=ops,acceptance=dict(causal_tests_required=True,matched_closed_loop_required=True,normal_scene_nonregression_required=True,cross_scene_nonregression_required=True),protected_scope=['all certification and execution methods','entire candidate selection process','obstacle memory','model/renderer/edit program/vehicle/map/initial state'],known_limits='No lane-center/initial-frame return goal added; native score improvement and full return remain unproven'))
write(r/'assistant_decision_receipt.json',dict(author=f['author'],evidence=asset(r/'evidence.json'),edit_instruction=asset(r/'edit_instruction.json'),failure_reason=asset(r/'failure_reason.json'),patch_instruction=asset(r/'patch_instruction.json')))
print(r)
