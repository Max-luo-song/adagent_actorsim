
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

import ast,copy,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shapely.geometry import Point,Polygon,mapping
from scene_contracts import ROOT,policy,sha
from boundaries import polygon_from_mask,fit_ground,camera_projection
from runtime_adapter import rewrite

class BoundaryTests(unittest.TestCase):
 def test_unknown_hole_and_disconnected_gap_remain_outside(self):
  mask=np.ones((12,12),bool);mask[4:8,4:8]=False;mask[:,9]=False
  poly=polygon_from_mask(mask,[0,0],.5)
  self.assertFalse(poly.covers(Point(2.5,2.5)));self.assertFalse(poly.covers(Point(4.5,1)));self.assertTrue(poly.covers(Point(.5,.5)))
  self.assertAlmostEqual(poly.area,mask.sum()*.25)
 def test_no_support_does_not_invent_a_width(self):
  with self.assertRaises(ValueError):polygon_from_mask(np.zeros((10,10)),[0,0],.5)
 def test_ground_fit_uses_geometry_with_unknown_flow_labels(self):
  rng=np.random.default_rng(0);p=np.zeros((1000,14));p[:,3:5]=rng.uniform([-5,-5],[20,5],(1000,2));p[:,5]=.01*p[:,3]-.02*p[:,4]-1.8;p[:,9]=-1;p[:,10]=1
  fit,residual=fit_ground(p);np.testing.assert_allclose(fit,[.01,-.02,-1.8],atol=1e-9);self.assertLess(residual,1e-9)
 def test_projection_uses_capture_pose_and_rejects_points_behind_camera(self):
  c={'opencv_to_waymo':np.eye(3),'camera_waymo_to_ego':np.eye(4),'intrinsics_9':[100,100,50,50,0,0,0,0,0]};pose=np.eye(4);pose[0,3]=1
  pix,z,valid=camera_projection(np.array([[1,0,10],[2,0,10],[1,0,-10]]),pose,c,(100,100))
  np.testing.assert_equal(pix[:2],[[50,50],[60,50]]);np.testing.assert_equal(valid,[True,True,False])
 def test_distortion_projection_matches_existing_semantic_projector(self):
  # Independent Brown-Conrady example: radial scale at x/z=.1.
  c={'opencv_to_waymo':np.eye(3),'camera_waymo_to_ego':np.eye(4),'intrinsics_9':[1000,1000,100,100,.2,0,0,0,0]};pix,_,valid=camera_projection(np.array([[1,0,10]]),np.eye(4),c,(1000,1000));self.assertTrue(valid[0]);np.testing.assert_equal(pix[0],[200,100])

class RuntimeContractTests(unittest.TestCase):
 def test_all_registered_runtime_versions_accept_only_known_host_substitutions(self):
  for version in ('v13','v15','v17'):
   row=policy(version);file=Path(row['code'])/'runtime.py';before=sha(file);tree,edits=rewrite(file.read_text());compile(tree,str(file),'exec');self.assertEqual(sha(file),before);self.assertIn('actor_count',[v for _,v in edits])
 def test_unexpected_runtime_schema_fails_closed(self):
  source=(ROOT/'integrations/light-tracking-curvature-v15-r2/runtime.py').read_text().replace("report['rigid_actor_count']!=15","report['rigid_actor_count']!=99")
  with self.assertRaises(ValueError):rewrite(source)
 def test_frozen_tracking_and_admission_calls_are_preserved(self):
  source=(ROOT/'integrations/light-tracking-curvature-v15-r2/runtime.py').read_text();tree,_=rewrite(source)
  def calls(tree):
   return [ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('process','admit','advance','control')]
  self.assertEqual(calls(ast.parse(source)),calls(tree))
 def test_no_static_scene_or_navigation_value_left_in_adapted_run(self):
  source=(ROOT/'integrations/light-tracking-curvature-v15-r2/runtime.py').read_text();tree,_=rewrite(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_arm')
  for node in ast.walk(fn):
   if isinstance(node,ast.keyword) and node.arg in ('scene','nav_command','command'):self.assertIsInstance(node.value,ast.Subscript)
 def test_non_run_functions_are_unchanged(self):
  source=(ROOT/'integrations/light-tracking-curvature-v15-r2/runtime.py').read_text();tree,_=rewrite(source)
  before={n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name!='run_arm'};after={n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='run_arm'};self.assertEqual(before,after)

class GeometryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path[:0]=[str(ROOT/'integrations/light-tracking-curvature-v15-r2'),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
  import common;common.configure()
  import map_reference
  from map_adapter import install
  cls.Geometry,cls.Reference=install(map_reference)
  cls.vehicle=SimpleNamespace(half_length_m=1.5,half_width_m=.8,clearance_margin_m=.2,dt=.1)
 def data(self,poly):return {'geometry':mapping(poly),'online_static_map_allowed':True,'unknown_is_drivable':False,'uncertainty_m':.3}
 def test_body_covering_unknown_hole_is_rejected_even_when_corners_are_on_road(self):
  poly=Polygon([[-10,-10],[30,-10],[30,10],[-10,10]],holes=[[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]]]);g=self.Geometry(self.data(poly),self.vehicle);self.assertLess(g.pose_clearance(np.eye(4)),0)
 def test_model_world_round_trip_and_road_sides(self):
  ref=self.Reference(self.vehicle,self.data(Polygon([[-10,-5],[40,-5],[40,5],[-10,5]])));info={'model_lidar_frame':{'virtual_to_ego':np.eye(4)},'explicit_l2g_r_mat':np.eye(3),'explicit_l2g_t':np.zeros(3)}
  packet=ref.observed(None,info,None);self.assertEqual(packet['status'],'fresh');edges={e['side']:e for e in packet['edges']};self.assertAlmostEqual(edges['left']['b'],-5.);self.assertAlmostEqual(edges['right']['b'],5.)
  p=np.array([[1.,2.],[-2.,5.]]);np.testing.assert_allclose(ref.geometry.model(ref.geometry.world(p)),p)
 def test_straight_map_full_body_coverage_preserves_unknown_exterior(self):
  ref=self.Reference(self.vehicle,self.data(Polygon([[-10,-5],[40,-5],[40,5],[-10,5]])));info={'model_lidar_frame':{'virtual_to_ego':np.eye(4)},'explicit_l2g_r_mat':np.eye(3),'explicit_l2g_t':np.zeros(3)};ref.observed(None,info,None)
  self.assertTrue(ref.coverage(np.array([[0,0,0,0,0]])));self.assertFalse(ref.coverage(np.array([[10,0,0,0,0]])))
class BudgetTests(unittest.TestCase):
 def fixture(self,root,exhausted=False,active=False):
  from scene_contracts import write
  import time
  write(root/'state.json',{'usage':{'gpu_attempts':17,'gpu_dispatches':17,'charged_tokens':203406},'started_epoch':time.time()-100 if exhausted else time.time(),'active_round':{'id':'active'} if active else None})
  write(root/'config.json',{'budgets':{'wall_seconds':60,'max_gpu_attempts':34}})
 def test_exhausted_shared_budget_never_reserves_a_GPU_attempt(self):
  import budget
  from unittest.mock import patch
  from scene_contracts import read
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,True)
   with patch.object(budget,'campaign',return_value=root):
    with self.assertRaises(RuntimeError):
     with budget.reserve(root/'episode'):self.fail('must not execute')
   self.assertEqual(read(root/'state.json')['usage']['gpu_attempts'],17)
 def test_active_parent_round_cannot_be_bypassed(self):
  import budget
  from unittest.mock import patch
  from scene_contracts import read
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,active=True)
   with patch.object(budget,'campaign',return_value=root):
    with self.assertRaises(RuntimeError):
     with budget.reserve(root/'episode'):self.fail('must not execute')
   self.assertEqual(read(root/'state.json')['usage']['gpu_attempts'],17)
 def test_dispatch_increments_existing_usage_without_resetting_tokens_or_clock(self):
  import budget
  from unittest.mock import patch
  from scene_contracts import read
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root);before=read(root/'state.json')
   with patch.object(budget,'campaign',return_value=root):
    with budget.reserve(root/'episode') as remaining:self.assertGreater(remaining,0)
   after=read(root/'state.json');self.assertEqual(after['usage']['gpu_attempts'],18);self.assertEqual(after['usage']['charged_tokens'],203406);self.assertEqual(after['started_epoch'],before['started_epoch'])

class ExtraAssetTests(unittest.TestCase):
 def test_replacement_sources_are_frozen_by_identity_not_controller_scores(self):
  from extra_assets import context,EXTRA
  from scene_contracts import ASSETS,read
  for scene in EXTRA:
   raw,checkpoint,count=context(scene);data=read(ASSETS/str(scene)/'source_manifest.json');original=read(ASSETS/str(scene)/'raw_metadata/raw_metadata.json')
   self.assertEqual(data['scene'],scene);self.assertEqual(data['raw_data'],str(raw));self.assertEqual(data['frames'],count);self.assertEqual(original['scene_index'],scene);self.assertTrue(original['raw_processed_pose_identity_all_frames']);self.assertTrue(original['raw_processed_image_identity_all_images'])

class ExplicitAllocationTests(unittest.TestCase):
 def test_explicit_wall_extension_preserves_usage_start_and_freeze_integrity(self):
  import budget
  from unittest.mock import patch
  from scene_contracts import read,write,sha
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);BudgetTests().fixture(root,exhausted=True);write(root/'host_freeze.json',{'code':{'frozen.py':'unchanged'},'config_sha256':sha(root/'config.json')});before=read(root/'state.json')
   with patch.object(budget,'campaign',return_value=root),patch.object(budget,'P26',root/'unrelated'):
    event=budget.extend_window(60)
   self.assertGreater(event['added_wall_seconds'],0);self.assertEqual(read(root/'state.json'),before);self.assertEqual(read(root/'host_freeze.json')['config_sha256'],sha(root/'config.json'));self.assertEqual(read(root/'host_freeze.json')['code'],{'frozen.py':'unchanged'})

if __name__=='__main__':unittest.main()
