
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

import copy,tempfile,time,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
from common_flow import AGENT,write,read,asset
from action_validation import family,publish,configuration
from action_program import validate_phases
from test_action_program import sequence,phase
from core import AgentError,sha
import generalization as g

class SceneActionCycleTests(unittest.TestCase):
 def setup_policy(self,root):
  cases=[dict(id='original',scene=551,kind='original',ready=True,baseline=None),*[dict(id='action'+str(i),scene=551,kind='action',ready=True,baseline=None,action_fingerprint='f'+str(i),action_family='m'+str(i),withheld_from_training=True) for i in range(2)]]
  write(root/'suite.json',dict(cases=cases));return dict(generalization=dict(required=True,mode='same_scene_actions',training_scene=551,training_base_pins={},training_action_fingerprints=['training'],training_action_families=['train'],cases=cases,suite=asset(root/'suite.json'),score_tolerance=1e-9))
 def test_new_same_scene_action_policy_does_not_require_other_scene_assets(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(g.policy(self.setup_policy(Path(d)))['mode'],'same_scene_actions')
 def test_duplicate_training_or_wrong_scene_cannot_be_validation(self):
  for failure in ['duplicate','training','scene','family','original']:
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);cfg=self.setup_policy(root);cases=cfg['generalization']['cases']
    if failure=='duplicate':cases[2]['action_fingerprint']=cases[1]['action_fingerprint']
    if failure=='training':cases[1]['action_fingerprint']='training'
    if failure=='scene':cases[1]['scene']=552
    if failure=='family':cases[2]['action_family']=cases[1]['action_family']
    if failure=='original':cases.pop(0)
    write(root/'suite.json',dict(cases=cases));cfg['generalization']['suite']=asset(root/'suite.json')
    with self.assertRaises(AgentError):g.policy(cfg)
 def test_unconverged_state_cannot_dispatch_any_validation_jobs(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg=self.setup_policy(root);state=dict(active_task_id='task',champion={'code_pins':{'new':'hash'}},task_convergence={'task':{'converged':False}})
   with patch.object(g.jobs,'launch') as job:
    with self.assertRaises(AgentError):g.step(state,cfg,object(),root)
    job.assert_not_called()
 def test_parameter_tweaks_and_extra_cruise_are_not_distinct_mechanisms(self):
  a=sequence();b=copy.deepcopy(a);b[0]['duration_s']=3.4;b.insert(1,phase('cruise',1.,1.))
  self.assertEqual(family(a),family(b))
  c=[phase('decelerate',6.,0.),phase('stop_hold',1.,0.),phase('accelerate',5.,.8)]
  self.assertNotEqual(family(a),family(c))
 def test_no_confirmed_proof_cannot_publish_global_default(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write(root/'global_patch.json',{'version':'v15'});write(root/'fake.json',dict(confirmed=True,validation_scope='same_scene_cross_action'))
   with patch('action_validation.P26',root):
    for simulated,status in [(True,'confirmed'),(False,'pending_generalization')]:
     with self.assertRaises(AgentError):publish(dict(accepted_patch=asset(root/'fake.json'),simulation_only=simulated,final_patch_status=status),{'generalization':{'mode':'same_scene_actions'}},root)
   self.assertEqual(read(root/'global_patch.json'),{'version':'v15'})
 def test_rotation_requests_different_scene_and_retains_budget(self):
  import task_rotation as rotation
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);code=root/'c.py';code.write_text('# controller');write(root/'confirmed.json',{'confirmed':True})
   state=dict(active_task_id='old',task_convergence={'old':{'converged':True}},simulation_only=False,final_patch_status='confirmed',accepted_patch=asset(root/'confirmed.json'),champion=dict(code_pins={str(code):sha(code)}),task_archive=[{'task_id':'old'}],current_task_descriptor={'scene':551},usage=dict(api_requests=3,charged_tokens=53419,gpu_attempts=10),rounds_started=2,started_epoch=time.time())
   cfg=dict(scene_cycle={'different_scene_after_confirmation':True},budgets=dict(wall_seconds=100,max_rounds=12,max_api_requests=18,max_api_tokens=2000000,max_gpu_attempts=28),task_queue=dict(tasks=[dict(id='same_scene',scene=551,ready=True),dict(id='new_scene',scene=552,ready=False)]))
   usage=copy.deepcopy(state['usage']);more,_=rotation.advance(state,cfg,object(),root)
   self.assertFalse(more);self.assertEqual(state['status'],'waiting_next_task');self.assertEqual(state['usage'],usage);self.assertTrue(read(root/'next_scene_request.json')['require_different_scene'])

 def test_actual_gate_export_contract_round_trip_with_controlled_episode_fixture(self):
  # Controlled fixture exercises export/verification; no real GPU or confirmation claim.
  import os,probe_worker
  from action_validation import execute
  from governance import SHARED
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=root/'old.py';old.write_text('# old');new=root/'new.py';new.write_text('# new')
   case=dict(id='held',scene=551,kind='action',action_fingerprint='fingerprint',task={'path':'unused'})
   def prepare(folder,code):
    folder.mkdir();work=folder/'episode';work.mkdir();write(folder/'case.json',case);write(folder/'policy.json',dict(code_pins={str(code):sha(code)}))
    for file in ['scene_manifest.json','scenario/scenario.json','map/map.json']:write(work/file,{})
    plan=dict(initial_context={},source_end_frame=35,seed=1,assets=dict(uniad_weights=asset(code)),controls=25);write(work/'plan.json',plan)
    contract={k:k for k in SHARED};contract['scene']=551;write(work/'contract.json',contract)
    report=dict(status='completed',real_closed_loop=True,model_weights_unchanged=True,validation_contract=asset(work/'contract.json'),trace=[{} for _ in range(25)],observations=[{'environment':{}} for _ in range(26)])
    write(work/'report.json',report);write(work/'summary.json',dict(native_scores=dict(scores=dict(executed_path={k:{'value':.3} for k in g.SCORE_KEYS})),anomalies={'environment_collision':False},stop_recovery_unresolved=False))
    return {k:asset(work/(k+'.json')) for k in ('report','summary','contract')}
   base=root/'base';baseline=prepare(base,old)
   with patch('os.fstat',return_value=None),patch.object(probe_worker,'run',return_value=baseline):execute(base)
   candidate=root/'candidate';side=prepare(candidate,new);write(candidate/'baseline_binding.json',asset(base/'baseline.json'))
   with patch('os.fstat',return_value=None),patch.object(probe_worker,'run',return_value=side):execute(candidate)
   target=dict(case,baseline=asset(base/'baseline.json'));result=g.verify_case(asset(candidate/'proof.json'),{'code_pins':{str(new):sha(new)}},target,0.,1e-9)
   self.assertTrue(result['passed']);self.assertEqual(result['candidate']['controls'],25)

 def test_partial_output_from_failed_baseline_job_cannot_pass(self):
  from action_validation import prepare_baselines
  import action_validation as av
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write(root/'suite.json',{});write(root/'policy.json',{'code_pins':{}})
   write(root/'action_validation_baselines/original/baseline.json',{})
   cfg=dict(generalization=dict(suite=asset(root/'suite.json'),baseline_policy=asset(root/'policy.json'),cases=[{'id':'original'}]))
   state={'usage':{'gpu_attempts':0,'gpu_dispatches':0,'charged_tokens':123}}
   with patch.object(av.jobs,'inspect',return_value={'status':'failed'}),patch.object(av.jobs,'launch') as launch:
    self.assertFalse(prepare_baselines(state,cfg,root));launch.assert_not_called()
   self.assertEqual(state['status'],'action_validation_baseline_infrastructure_failure');self.assertEqual(state['usage']['charged_tokens'],123)
