
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

import sys,json,types,unittest,copy,tempfile,pickle
pass  # Path is provided by the portable boundary above
from paths import *
for folder in [HERE/'framework_snapshot/unified',HERE/'framework_snapshot/framework_backend',HERE/'framework_snapshot',ROOT/'waymo23-native-import']:sys.path.insert(0,str(folder))
sys.path.insert(0,str(HERE))
import numpy as np
import torch
from native_car_edit import VehiclePrograms,effective_source
from scenario import compile_vehicle_path
from native_scores import raw_plan,actor_boxes,export,native_trajectory_export,route_completion
class VehicleTests(unittest.TestCase):
 def fixture(self):
  with np.load(SCENE_BUNDLE/'actors.npz') as z:t=z['RigidNodes__instances_trans'].copy();q=z['RigidNodes__instances_quats'].copy();fv=z['RigidNodes__instances_fv'].copy();size=z['RigidNodes__instances_size'].copy()
  m=types.SimpleNamespace(instances_trans=torch.tensor(t),instances_quats=torch.tensor(q),instances_fv=torch.tensor(fv),instances_size=torch.tensor(size));b=types.SimpleNamespace(trainer=types.SimpleNamespace(models={'RigidNodes':m}),env=types.SimpleNamespace(models={'RigidNodes':{k:getattr(m,k) for k in ['instances_trans','instances_quats','instances_fv','instances_size']}}),torch=torch,dataset=types.SimpleNamespace(test_timesteps=[3,4,5,7]),clear_observation_history=lambda:None)
  obj=VehiclePrograms(b,types.SimpleNamespace(fps=10.,frames=199));command=dict(object_id='native:RigidNodes:3',operation='set_actor_trajectory',trajectory=_hugsim_expand(json.loads((HERE/'scenario.json').read_text()))['trajectory']);return obj,m,b,command
 def test_original_pose_other_actors_identity_and_restore(self):
  obj,m,b,command=self.fixture();t=m.instances_trans.clone();q=m.instances_quats.clone();identity=id(m.instances_trans);obj.apply(command);self.assertEqual(id(m.instances_trans),identity);torch.testing.assert_close(m.instances_trans[0,3],t[0,3],rtol=0,atol=0);torch.testing.assert_close(m.instances_trans[:,:3],t[:,:3],rtol=0,atol=0);self.assertLess(abs(float(m.instances_trans[34,3,1])-.3),.02);obj.restore();torch.testing.assert_close(m.instances_trans,t,rtol=0,atol=0);torch.testing.assert_close(m.instances_quats,q,rtol=0,atol=0)
 def test_heldout_renderer_pose_matches_proxy_and_restores_neighbors(self):
  obj,m,b,command=self.fixture();obj.apply(command);before=m.instances_trans.clone();quats=m.instances_quats.clone()
  with obj.render_context(5):torch.testing.assert_close((m.instances_trans[4,3]+m.instances_trans[6,3])*.5,m.instances_trans[5,3],rtol=0,atol=0)
  torch.testing.assert_close(m.instances_trans,before,rtol=0,atol=0);torch.testing.assert_close(m.instances_quats,quats,rtol=0,atol=0)
 def test_sh_rotation_is_target_only_and_restored_on_failure(self):
  obj,m,b,command=self.fixture();m.point_ids=torch.tensor([3,2,3]);m._features_rest=torch.arange(27,dtype=torch.float32).reshape(3,3,3);original=m._features_rest.clone();obj.apply(command)
  with obj.render_context(0):torch.testing.assert_close(m._features_rest,original,rtol=0,atol=0)
  with self.assertRaisesRegex(RuntimeError,'render failure'):
   with obj.render_context(20):
    self.assertFalse(torch.equal(m._features_rest[[0,2]],original[[0,2]]));torch.testing.assert_close(m._features_rest[1],original[1],rtol=0,atol=0);raise RuntimeError('render failure')
  torch.testing.assert_close(m._features_rest,original,rtol=0,atol=0)
 def test_session_edit_checks_revision_and_started_phase(self):
  from native_car_edit import make_types
  _,Session=make_types();obj,m,b,command=self.fixture();b._last=None;session=object.__new__(Session);session.backend=b;session.vehicles=obj;session.revision=0;session.generation='cpu_fixture';session._editor=types.SimpleNamespace(require_current=lambda:None,started=False,journal=[])
  command.update(schema='scene_edit_v1',mode='diagnostic_preview',expected_revision=0,expected_generation='cpu_fixture');receipt=session.edit(command);self.assertEqual(receipt['edit_revision'],1);self.assertEqual(len(session._editor.journal),1);before=m.instances_trans.clone()
  with self.assertRaisesRegex(ValueError,'Stale'):session.edit(command)
  torch.testing.assert_close(m.instances_trans,before,rtol=0,atol=0);session._editor.started=True;command['expected_revision']=1
  with self.assertRaisesRegex(RuntimeError,'pre-run'):session.edit(command)
  session._editor.started=False;restore={k:v for k,v in command.items() if k!='trajectory'};restore['operation']='restore_actor_trajectory';session.edit(restore);self.assertEqual(session.revision,2);self.assertFalse(obj.programs)
 def test_edit_failure_rolls_back(self):
  obj,m,b,command=self.fixture();before=m.instances_trans.clone();b.clear_observation_history=lambda:(_ for _ in ()).throw(RuntimeError('fixture failure'))
  with self.assertRaises(RuntimeError):obj.apply(command)
  torch.testing.assert_close(m.instances_trans,before,rtol=0,atol=0);self.assertFalse(obj.programs)
 def test_missing_visibility_rejected(self):
  obj,m,b,command=self.fixture();m.instances_fv[30,3]=False
  with self.assertRaises(ValueError):obj.apply(command)
 def test_high_speed_trajectory_rejected(self):
  obj,m,b,command=self.fixture();bad=copy.deepcopy(command['trajectory']);bad['waypoints'][-1]['position_world_m'][0]+=1000.
  with self.assertRaises(ValueError):compile_vehicle_path(bad,m.instances_trans[:,3].numpy(),m.instances_quats[:,3].numpy())
 def test_actor_export_has_original_count_and_target_motion(self):
  obj,m,b,command=self.fixture();obj.apply(command);tracks=obj.export();self.assertEqual(tracks['translations'].shape,(199,15,3));np.testing.assert_array_equal(tracks['translations'][:,3],m.instances_trans[:,3].numpy());boxes,names=actor_boxes(tracks,25);self.assertEqual(len(boxes),len(names));self.assertTrue(all(n=='car' for n in names))
 def test_source_scenario_motion_finite_and_ends_in_lane(self):
  obj,m,b,command=self.fixture();t,q,a=compile_vehicle_path(command['trajectory'],m.instances_trans[:,3].numpy(),m.instances_quats[:,3].numpy());self.assertLess(a['max_acceleration_mps2'],12.);self.assertLess(a['max_yaw_rate_radps'],.8);self.assertAlmostEqual(t[-1,1],.3,places=5);self.assertTrue(np.isfinite(q).all())
class ScoreTests(unittest.TestCase):
 def test_stock_export_preserves_native_five_node_behavior(self):
  traj=native_trajectory_export(np.c_[np.arange(1,7),np.zeros(6)],[2.,3.,0.,2.,4.,1.5,0.]);self.assertEqual(len(traj),5);np.testing.assert_array_equal(np.asarray(traj)[:,0],np.arange(3.,8.))
 def test_raw_plan_validates_and_keeps_native_frame(self):
  info=dict(timestamp=0.,model_lidar_frame={'virtual_to_ego':np.eye(4)},explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3));v=types.SimpleNamespace(half_width_m=1.,half_length_m=2.,body_height_m=1.5);a=raw_plan(np.c_[np.zeros(6),np.arange(1,7)],info,v);self.assertEqual(len(a['planned_traj']['traj']),5);np.testing.assert_array_equal(np.asarray(a['planned_traj']['traj'])[:,0],np.arange(1.,6.))
  with self.assertRaises(ValueError):raw_plan(np.full((6,2),np.nan),info,v)
 def test_native_route_completion_is_finite(self):self.assertTrue(np.isfinite(route_completion([0.,0.])))
 def test_no_confirmed_control_never_fabricates_scores(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);(p/'run_report.json').write_text(json.dumps(dict(trace=[])));np.savez(p/'edited_actor_tracks.npz',translations=np.zeros((199,1,3)),quaternions_wxyz=np.tile([1.,0.,0.,0.],(199,1,1)),visible=np.ones((199,1),bool),size_lwh=np.ones((1,3)))
   with self.assertRaises(ValueError):export(p,None)
 def test_native_aggregation_uses_full_episode_rc(self):
  from native_scores import native_aggregate
  details={0.:dict(nc=1.,dac=1.,ttc=0.,c=1.,pdms=2./7.),.1:dict(nc=1.,dac=1.,ttc=1.,c=1.,pdms=1.)};data=[dict(frames=[dict(rc=.1),dict(rc=.2),dict(rc=.2)])];value=native_aggregate(data,details);self.assertAlmostEqual(value['pdms'],(2./7.+1.)/2.);self.assertEqual(value['rc'],.2);self.assertAlmostEqual(value['hdscore'],value['pdms']*.2)
 def test_independent_patch_comparison_rejects_changed_scenario_and_incomplete_success(self):
  from compare_runs import compare
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);a=root/'a';b=root/'b'
   for run in (a,b):(run/'native_eval').mkdir(parents=True)
   fields={k:dict(lower=.5,upper=.5,value=.5) for k in ('nc','dac','ttc','c','pdms','rc','hdscore')};base=dict(status='computed_native_metric_bounds',comparison_key={'scenario':'same'},episode_complete=True,scores={'adopted_plan':fields});candidate=copy.deepcopy(base);candidate['scores']['adopted_plan']['hdscore'].update(lower=.8,upper=.8,value=.8)
   def write(run,data):(run/'native_eval/native_scores.json').write_text(json.dumps(data))
   write(a,base);write(b,candidate);self.assertTrue(compare(a,b)['robust_hdscore_improvement']);candidate['episode_complete']=False;write(b,candidate);self.assertFalse(compare(a,b)['robust_hdscore_improvement']);candidate['comparison_key']['scenario']='changed';write(b,candidate)
   with self.assertRaises(ValueError):compare(a,b)
 def test_paired_export_preserves_future_motion_and_does_not_invent_progress(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'run_report.json').write_text(json.dumps(dict(trace=[{},{}],status='completed',checkpoint_before={'sha256':'cpu_fixture'},uniad_seed=20261021)));t=np.zeros((199,1,3));t[:,0,0]=np.arange(199);np.savez(root/'edited_actor_tracks.npz',translations=t,quaternions_wxyz=np.tile([1.,0.,0.,0.],(199,1,1)),visible=np.ones((199,1),bool),size_lwh=np.array([[4.,2.,1.5]]));v=types.SimpleNamespace(half_width_m=1.,half_length_m=2.,body_height_m=1.5)
   for i in range(2):
    step=root/'uniad_worker'/('step_%03d'%i);step.mkdir(parents=True);info=dict(timestamp=i/10.,model_lidar_frame={'virtual_to_ego':np.eye(4)},explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.array([0.,float(i),0.]));plan=np.c_[np.zeros(6),np.arange(1.,7.)];np.savez(step/'perception.npz',trajectory=plan)
    with (step/'sensor_metadata.pkl').open('wb') as f:pickle.dump(info,f)
    adopted=raw_plan(plan,info,v);adopted['planned_traj']['traj']=np.c_[np.arange(1.,7.)+i,np.zeros(6),np.zeros(6)].tolist();(step/'native_adopted_plan.json').write_text(json.dumps(adopted))
   unconfirmed=root/'uniad_worker/step_002';unconfirmed.mkdir();(unconfirmed/'native_adopted_plan.json').write_text('{}');data,manifest=export(root,v);self.assertEqual(manifest['pairs'],2);self.assertEqual(manifest['discarded_unconfirmed_attempt_steps'],[2]);self.assertFalse(manifest['episode_complete']);frames=data['raw_uniad'][0]['frames'];self.assertEqual(len(frames),27);self.assertFalse(frames[-1]['is_key_frame']);self.assertEqual(frames[-1]['obj_boxes'][0][0],26.);self.assertEqual(frames[-1]['rc'],max(x['rc'] for x in frames[:2]));self.assertEqual(len(frames[0]['planned_traj']['traj']),5)
if __name__=='__main__':unittest.main(verbosity=2)
