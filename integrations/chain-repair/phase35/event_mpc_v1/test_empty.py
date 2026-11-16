
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
import unittest,json,pickle,copy,ast,numpy as np
from empty_native_temporal import TemporalOccupancy
from mpc_reuse_v1.native_temporal import TemporalOccupancy as Previous
from runtime_configuration import load_vehicle_config
import empty_controller as controller
from evidence_guard import PlanningGuard
SOURCE=CR/'phase31/runs/depth_consistency551_20261105T051937Z_fc796426/closed_loop'
class EmptyPredictionTests(unittest.TestCase):
 def setUp(self):
  self.v=load_vehicle_config();self.m=np.zeros((1,5,200,200),dtype=np.int64);self.states=np.array([[0.,0.,0.,0.,0.],[0.,2.,0.,1.,0.]])
 def test_valid_empty_has_only_grid_boundary(self):
  occ=TemporalOccupancy(self.m,self.v)
  np.testing.assert_allclose(occ.clearance(self.states,[0.,1.]),[50.-self.v.half_length_m-self.v.clearance_margin_m,48.-self.v.half_length_m-self.v.clearance_margin_m])
  self.assertLess(occ.clearance(np.array([[0.,49.,0.,1.,0.]]),[0.])[0],0)
 def test_depth_and_unknown_still_block_empty_raster(self):
  for field in ('observed','unknown'):
   for cell in (False,True):
    with self.subTest(field=field,cell=cell):
     kw={field+'_points':[[0.,0.]],field+'_point_radius':np.sqrt(3)*.25}
     if cell:kw[field+'_cell_rotation']=np.eye(3)
     self.assertLess(TemporalOccupancy(self.m,self.v,**kw).clearance(self.states[:1],[0.])[0],0.)
 def test_missing_malformed_and_nonfinite_rejected(self):
  for m in (None,np.zeros((5,200,200)),np.full((1,5,200,200),np.nan)):
   with self.assertRaises(ValueError):TemporalOccupancy(m,self.v)
   with self.assertRaises(ValueError):controller.optimize(np.zeros((6,2)),m,{},vehicle=self.v)
 def test_all_saved_nonempty_clearances_identical_and_empty_accepted(self):
  counts=[0,0]
  for f in sorted((SOURCE/'uniad_worker').glob('step_*/perception.npz')):
   with np.load(f) as z:m=z['occupancy'].reshape(1,5,200,200)
   before=m.copy();new=TemporalOccupancy(m,self.v);a=new.clearance(self.states,[0.,1.])
   if np.any(m):np.testing.assert_array_equal(a,Previous(m,self.v).clearance(self.states,[0.,1.]));counts[0]+=1
   else:counts[1]+=1;self.assertTrue(np.isfinite(a).all())
   np.testing.assert_array_equal(m,before)
  self.assertEqual(counts,[96,76])
 def test_confirmed_proposal_still_blocks_empty_input(self):
  f=SOURCE/'uniad_worker/step_040'
  with (f/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
  rel=_hugsim_expand(json.loads((f/'guard_memory_report.json').read_text()));rel=copy.deepcopy(rel)
  off=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
  rel['obstacles']=[dict(candidate_index=0,state='network_retained',raw_score=.9,label=0,box=[0.,0.,-1.,2.,4.,1.5,0.,0.,0.],observations=1,duration_seconds=0.)]
  controller.PlanningGuard=PlanningGuard
  result=controller.optimize(np.zeros((6,2)),self.m,info,reliability=rel,vehicle=self.v)
  self.assertIsNone(result['selected']);self.assertEqual(result['reason'],'fixed_initial_occupancy_conflict');self.assertLess(result['initial_conflict']['hard_proposal_clearance_m'],0.)
 def test_safety_constraints_and_rollout_ast_unchanged(self):
  old=ast.parse((CR/'phase31/occupancy_closed_v2/empty_controller.py').read_text());new=ast.parse((HERE/'empty_controller.py').read_text())
  for name in ('constraints','clear','rollout','integrate_state'):
   before=next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name==name)
   after=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name==name)
   if name=='clear':
    self.assertEqual([ast.unparse(d) for d in after.decorator_list],['memo.wrap']);after.decorator_list=[]
   self.assertEqual(ast.dump(before),ast.dump(after),name)
if __name__=='__main__':unittest.main(verbosity=2)
