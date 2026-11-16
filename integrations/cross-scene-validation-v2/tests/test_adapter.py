
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

import sys,ast,types,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
from shapely.geometry import Polygon,MultiPolygon,mapping
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL))
from bindings import ROOT,OLD,MAP,CANON,load,runtime_tree
from pipeline import choose_start
from support_map import install

class AdapterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  common=load('common',OLD/'common.py');common.configure();sys.path[:0]=[str(CANON),str(MAP),str(OLD)]
  import map_reference
  module=types.SimpleNamespace(StaticGeometry=map_reference.StaticGeometry,MapRoadReference=type('Stub',(object,),{}));cls.adapter=install(module)
  cls.v=types.SimpleNamespace(half_length_m=1.5,half_width_m=.8,clearance_margin_m=.2)
 def geometry(self,poly):return self.adapter.StaticGeometry({'geometry':mapping(poly),'authority':'reconstructed_support_proxy','holes_preserved':True,'uncertainty_m':.3},self.v)
 def test_hole_is_not_free_even_when_outer_polygon_covers_ego(self):
  poly=Polygon([[-12,-12],[12,-12],[12,12],[-12,12]],holes=[[[-3,-3],[3,-3],[3,3],[-3,3]]]);g=self.geometry(poly)
  self.assertLess(g.pose_clearance(np.eye(4)),0);p=np.eye(4);p[0,3]=7;self.assertGreater(g.pose_clearance(p),0)
  iy=np.argmin(abs(g.ys));ix=np.argmin(abs(g.xs));self.assertLess(g.sdf[iy,ix],0)
 def test_disconnected_roads_do_not_create_a_bridge(self):
  a=Polygon([[-12,-4],[-4,-4],[-4,4],[-12,4]]);b=Polygon([[4,-4],[12,-4],[12,4],[4,4]]);g=self.geometry(MultiPolygon([a,b]));self.assertLess(g.pose_clearance(np.eye(4)),0)
 def test_missing_authority_rejected(self):
  with self.assertRaises(ValueError):self.adapter.StaticGeometry({'geometry':mapping(Polygon([[0,0],[9,0],[9,9],[0,9]])),'holes_preserved':True,'authority':'surveyed_white_line','uncertainty_m':.3},self.v)
 def test_start_selection_requires_full_body_and_scoring_horizon(self):
  poly=Polygon([[5,-5],[20,-5],[20,5],[5,5]]);poses=np.tile(np.eye(4),(70,1,1));poses[:,0,3]=np.arange(70)*.2
  v={'half_length_m':1.5,'half_width_m':.8,'clearance_margin_m':.2};i=choose_start(poly,poses,v);self.assertEqual(i,36)
  self.assertIsNone(choose_start(poly,poses[:50],v))
 def test_scene_metadata_change_keeps_other_runtime_functions_identical(self):
  source=(CANON/'runtime.py').read_text();tree,changes=runtime_tree(source,552,0,11);original=ast.parse(source)
  old={n.name:ast.dump(n) for n in original.body if isinstance(n,ast.FunctionDef) and n.name!='run_arm'};new={n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='run_arm'}
  self.assertEqual(old,new);self.assertEqual(set(changes),{'scene','command','nav_command','actor_count'})
 def test_runtime_actor_count_is_scene_data_not_fixed_fifteen(self):
  t,_=runtime_tree((CANON/'runtime.py').read_text(),172,1,28);text=ast.unparse(t);self.assertIn("report['rigid_actor_count'] != 28",text);self.assertNotIn("report['rigid_actor_count'] != 15",text);self.assertIn('command=1',text);self.assertIn('verify_original_actor_arrays(root, tracks)',text)
 def test_changed_runtime_import_contract_is_rejected(self):
  from pipeline import read
  with self.assertRaises(ValueError):runtime_tree((CANON/'runtime.py').read_text().replace('scene=551','scene=999'),172,1,28)
if __name__=='__main__':unittest.main()
