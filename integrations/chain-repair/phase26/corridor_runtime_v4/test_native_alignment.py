
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import ast,contextlib,copy,io,json,pickle,tempfile,types,unittest,numpy as np
from native_contract import CONTRACT,calculate_trajectory_kinematics,comfort_terms,combine
from native_objective import NativeAlignedObjective
from native_plan_export import save_plan
from native_evaluation import evaluate
from runtime_configuration import load_vehicle_config
from postprocess import summarize
class NativeAlignmentTests(unittest.TestCase):
 def test_native_formula_gates_and_exact_5_to_2_weights(self):
  self.assertAlmostEqual(combine(1.,1.,.4,.8,.7)['hdscore'],((5*.4+2*.8)/7)*.7)
  for values in ((0.,1.,1.,1.,1.),(1.,0.,1.,1.,1.),(1.,1.,1.,1.,0.)):self.assertEqual(combine(*values)['hdscore'],0.)
  self.assertEqual(CONTRACT['active_weights'],{'ttc':5,'c':2});self.assertEqual(CONTRACT['inactive_declared_weights'],['ep'])
 def test_comfort_matches_unmodified_native_method(self):
  source=(ROOT/CONTRACT['source']).read_text();tree=ast.parse(source);klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScoreCalculator');method=copy.deepcopy(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_calculate_is_comfortable'));standalone=ast.ClassDef(name='Original',bases=[],keywords=[],body=[method],decorator_list=[]);namespace=dict(np=np,boundaries=CONTRACT['comfort_boundaries'],calculate_trajectory_kinematics=calculate_trajectory_kinematics);exec(compile(ast.fix_missing_locations(ast.Module(body=[standalone],type_ignores=[])),'original-native-comfort','exec'),namespace)
  t=np.arange(13)*.5
  cases=[np.c_[2*t,np.zeros(13),np.zeros(13)],np.c_[3*t*t,np.zeros(13),np.zeros(13)],np.c_[t,np.zeros(13),2*t],np.c_[t,np.zeros(13),np.linspace(3.,3.5,13)],np.zeros((3,3))]
  for traj in cases:
   with contextlib.redirect_stdout(io.StringIO()):expected=namespace['Original']()._calculate_is_comfortable(traj,.5)
   self.assertEqual(comfort_terms(traj,.5)['native_flag'],expected)
 def test_native_kinematic_function_is_byte_equivalent_ast(self):
  source=ast.parse((ROOT/CONTRACT['source']).read_text());original=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_trajectory_kinematics');copytree=ast.parse((HERE/'native_contract.py').read_text());local=next(n for n in copytree.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_trajectory_kinematics');self.assertEqual(ast.dump(original),ast.dump(local));self.assertIn('zero array',CONTRACT['native_lat_acceleration_implementation'])
 def objective(self,road_loss=0.,obstacle=100.):
  v=load_vehicle_config();plan=np.c_[np.zeros(6),np.arange(1,7)];road=types.SimpleNamespace(has_road=True,peak_violation=lambda states,**kwargs:0.,cost=lambda states,stride:road_loss*len(states)/stride);calls=[]
  def clearance(states,times):calls.append(np.asarray(states).copy());return obstacle-states[:,1]-v.half_length_m-v.clearance_margin_m
  temporal=types.SimpleNamespace(clearance=clearance);guard=types.SimpleNamespace(clearance=lambda states,times:np.full(len(states),np.inf),soft_cost=lambda *a:0.);clear=lambda states:clearance(states,np.arange(len(states))*.1)
  obj=NativeAlignedObjective(plan,v,road,temporal,guard,clear);obj.epistemic=types.SimpleNamespace(cost=lambda *a:0.);return obj,calls
 def states(self,speed=2.):
  t=np.arange(61)*.1;s=np.zeros((61,5));s[:,1]=speed*t;s[:,3]=speed;return s,s[:,:2].copy()
 def test_ttc_uses_native_half_and_one_second_projections(self):
  obj,calls=self.objective(obstacle=6.);states,lidar=self.states();states[:,1]=np.minimum(states[:,1],4.);lidar=states[:,:2].copy();m=obj.components(states,lidar,np.zeros((6,2)))
  self.assertEqual(m['nc_proxy'],1.);self.assertLess(m['ttc_proxy'],1.);self.assertAlmostEqual(calls[-2][0,1],1.);self.assertAlmostEqual(calls[-1][0,1],2.);self.assertEqual(len(calls[-1]),5)
 def test_bad_road_cannot_be_cancelled_by_fast_progress(self):
  clean,_=self.objective(road_loss=0.);bad,_=self.objective(road_loss=1.);slow,sl=self.states(1.4);fast,fl=self.states(2.);controls=np.zeros((6,2));a=clean.components(slow,sl,controls);b=bad.components(fast,fl,controls)
  self.assertGreater(a['local_hd_proxy'],b['local_hd_proxy']);self.assertGreater(b['local_progress_proxy'],a['local_progress_proxy'])
 def test_progress_is_signed_not_accumulated_loop_length(self):
  obj,_=self.objective();states,lidar=self.states();t=np.arange(61)*.1;lidar[:,1]=np.sin(np.pi*t);self.assertAlmostEqual(obj.progress(lidar),0.,places=10)
 def fixture(self):
  source=CR/'phase31/runs/occupancy_closed551_20261105T140958Z_48dc4599/closed_loop';f=source/'uniad_worker/step_046'
  with (f/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
  del obs
  selected=_hugsim_expand(json.loads((f/'optimization.json').read_text()))['selected'];report=_hugsim_expand(json.loads((source/'run_report.json').read_text()));return info,selected,report
 def test_adopted_plan_coordinate_mapping_and_native_comfort(self):
  info,selected,report=self.fixture();v=load_vehicle_config()
  with tempfile.TemporaryDirectory() as temp:data=save_plan(temp,selected,True,info,v)
  pose=np.asarray(report['observations'][46]['closed_ego_to_world']);np.testing.assert_allclose(data['ego_box'][:2],pose[:2,3],atol=1e-6);self.assertAlmostEqual(data['ego_box'][6],np.arctan2(pose[1,0],pose[0,0]),places=7)
  nodes=np.asarray(selected['states'])[::5];export=np.vstack([np.asarray(data['ego_box'])[[0,1,6]],data['planned_traj']['traj']]);self.assertEqual(comfort_terms(nodes[:,:3],.5)['native_flag'],comfort_terms(export,.5)['native_flag']);self.assertFalse(data['scene_truth_used_for_control'])
 def test_missing_native_scene_data_never_creates_a_full_score(self):
  info,selected,report=self.fixture();v=load_vehicle_config()
  with tempfile.TemporaryDirectory() as temp:
   run=Path(temp);folder=run/'uniad_worker/step_000';folder.mkdir(parents=True);save_plan(folder,selected,True,info,v);(run/'run_report.json').write_text(json.dumps(dict(status='completed',trace=[{}],observations=[{}],trajectory_score={'score':99},passed=True)))
   result=evaluate(run);self.assertFalse(result['full_native_score_available']);self.assertIsNotNone(result['scores']['c'])
   for field in ('nc','dac','ttc','pdms','rc','hdscore'):self.assertIsNone(result['scores'][field])
   summary=summarize(run);self.assertNotIn('passed',summary);self.assertNotIn('route_rejoined',summary);self.assertNotIn('trajectory_score',summary);self.assertFalse(summary['case_specific_success_used'])
 def test_exported_braking_plan_is_finite_and_no_scene_truth_is_read(self):
  info,selected,report=self.fixture();v=load_vehicle_config()
  with tempfile.TemporaryDirectory() as temp:data=save_plan(temp,None,False,info,v)
  self.assertEqual(data['plan_kind'],'controlled_braking_fallback');self.assertTrue(np.isfinite(data['planned_traj']['traj']).all());self.assertEqual(len(data['planned_traj']['traj']),12)
 def test_unverified_native_bundle_cannot_create_scores(self):
  with tempfile.TemporaryDirectory() as temp:
   run=Path(temp);(run/'run_report.json').write_text(json.dumps(dict(trace=[],observations=[])));bundle=run/'native_eval';bundle.mkdir()
   for name in ('data.pkl','ground.ply','scene.ply'):(bundle/name).write_text('unverified')
   (bundle/'native_export_manifest.json').write_text(json.dumps(dict(native_source_sha256=CONTRACT['source_sha256'])))
   result=evaluate(run);self.assertEqual(result['status'],'invalid_native_export_manifest');self.assertFalse(result['full_native_score_available']);self.assertIsNone(result['scores']['hdscore'])
if __name__=='__main__':unittest.main(verbosity=2)
