
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
import unittest,pickle,json,copy,types
import numpy as np
from white_edge import WhiteEdge,detect,restrict_road,SCHEMA,geometry,project
from runtime_configuration import load_vehicle_config
from motion_policy import braking_plan
class WhiteEdgeTests(unittest.TestCase):
 def setUp(self):self.v=load_vehicle_config()
 def packet(self,side='right',a=0.,b=2.):
  return dict(schema=SCHEMA,status='fresh',timestamp=0.,ground_truth_used=False,edges=[dict(side=side,a=a,b=b,valid_y_range=[-10,50],observed_y_range=[6,45],uncertainty_m=.15)])
 def saved(self,n):
  folder=CR/f'phase43/runs/cutin551_20261109T045728Z_11636/uniad_worker/step_{n:03d}'
  with (folder/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
  return folder,obs,info
 def test_actual_camera_edge_and_crossed_old_plan(self):
  f,obs,info=self.saved(49);p=detect(obs,info);self.assertEqual(len(p['edges']),1);self.assertEqual(p['edges'][0]['side'],'right');e=WhiteEdge(p,self.v)
  states=np.array(_hugsim_expand(json.loads((f/'optimization.json').read_text()))['selected']['states']);self.assertLess(e.clearance(states).min(),-.5)
  f,obs,info=self.saved(0);p=detect(obs,info);offset=np.array(info['model_lidar_frame']['virtual_to_ego'])[:2,3];self.assertGreater(WhiteEdge(p,self.v).clearance(np.array([[-offset[0],-offset[1],0,8,0]])).min(),0)
 def test_body_corner_crosses_even_if_center_inside(self):
  e=WhiteEdge(self.packet(),self.v);s=np.array([[1.8,10.,0.,2.,0.]]);self.assertLess(e.clearance(s)[0],0)
 def test_rotated_body_support_and_left_edge(self):
  e=WhiteEdge(self.packet(a=.1,b=3.),self.v);s=np.array([[.2,15,.3,2.,0.]])
  # Analytic support for the whole rectangle against a*x? boundary x=.1*y+3.
  a=.1;u=np.array([np.cos(.3),-np.sin(.3)]);f=np.array([np.sin(.3),np.cos(.3)]);n=np.array([1.,-a]);expected=(3.-n@s[0,:2]-abs(n@u)*self.v.half_width_m-abs(n@f)*self.v.half_length_m)/np.linalg.norm(n)-self.v.clearance_margin_m-.15
  self.assertAlmostEqual(e.clearance(s)[0],expected,places=9)
  left=WhiteEdge(self.packet(side='left',b=-2.),self.v);self.assertLess(left.clearance(np.array([[-1.8,10.,0.,0.,0.]]))[0],0)
 def test_finite_observation_band_clips_complete_body(self):
  p=self.packet();p['edges'][0]['valid_y_range']=[9.9,10.1];e=WhiteEdge(p,self.v)
  self.assertLess(e.clearance(np.array([[2.,10.,0.,1.,0.]]))[0],0);self.assertTrue(np.isinf(e.clearance(np.array([[2.,30.,0.,1.,0.]]))[0]))
 def test_empty_or_nonfinite_contract_never_fabricates_edge(self):
  p=dict(schema=SCHEMA,status='unavailable',ground_truth_used=False,edges=[]);self.assertTrue(np.isinf(WhiteEdge(p,self.v).clearance(np.zeros((1,5)))[0]))
  p=self.packet();p['edges'][0]['a']=float('nan')
  with self.assertRaises(ValueError):WhiteEdge(p,self.v)
 def test_raw_model_and_temporal_predictions_preserved(self):
  f,obs,info=self.saved(0);report=_hugsim_expand(json.loads((f/'guard_memory_report.json').read_text()));raw=report['predicted_drivable_area'];before=copy.deepcopy(raw);p=detect(obs,info);result=restrict_road(raw,p,self.v)
  self.assertEqual(raw,before);self.assertEqual(result['mask'],before['mask']);self.assertEqual(result['model_temporal_drivable_mask'],before['temporal_drivable_mask']);self.assertGreater(result['white_edge_clipped_cells'],0)
  self.assertFalse(np.any(np.asarray(result['temporal_drivable_mask'])&~np.asarray(before['temporal_drivable_mask'])))
 def test_white_car_and_dash_or_pavement_divider_rejected(self):
  _,obs,info=self.saved(0);rgb=obs['rgb']['CAM_FRONT'];h,w=rgb.shape[:2];K,M,z=geometry(info,'CAM_FRONT',rgb.shape)
  base=np.full_like(rgb,[80,78,82]);uv,_=project(np.c_[np.full(200,1.85),np.linspace(8,45,200),np.full(200,z)],K,M)
  import cv2
  for mode in ('car','dashed','divider'):
   im=base.copy()
   if mode=='car':cv2.rectangle(im,(360,390),(480,470),(230,230,230),-1)
   else:
    # White divider has no pavement->shoulder transition; dashed line has gaps.
    if mode=='dashed':
     for j in range(0,180,40):cv2.polylines(im,[np.rint(uv[j:j+16]).astype(np.int32)],False,(230,230,230),5)
    else:cv2.polylines(im,[np.rint(uv).astype(np.int32)],False,(230,230,230),5)
   self.assertEqual(detect({'rgb':{'CAM_FRONT':im}},info)['status'],'unavailable',mode)
 def test_braking_cannot_trade_edge_for_other_infeasible_clearance(self):
  road=types.SimpleNamespace(mask=np.ones((2,2),bool),clearance=lambda s:np.ones(len(s)),white_edge=WhiteEdge(self.packet(b=1.85),self.v))
  plan=braking_plan(np.array([0.,0.,0.,4.,0.]),np.array([[0.,2.],[0.,4.]]),self.v,lambda s:np.full(len(s),-10.+s[-1,0]),road)
  self.assertTrue(plan['white_edge_feasible']);self.assertGreaterEqual(plan['white_edge_clearance_m'],-1e-5)
 def test_current_constraints_include_edge_with_soft_road(self):
  import ast
  root=ast.parse((HERE/'empty_controller.py').read_text());fn=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name=='optimize')
  for name in ('clear','robust_clear'):
   f=next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name==name);f.decorator_list=[]
   edge=WhiteEdge(self.packet(b=1.85),self.v);scope=dict(np=np,vehicle=self.v,temporal_occupancy=types.SimpleNamespace(clearance=lambda s,t:np.full(len(s),100.)),guard=types.SimpleNamespace(clearance=lambda s,t:np.full(len(s),100.),full_clearance=lambda s,t:np.full(len(s),100.)),_forecast_recovery=False,sweep_allowance=lambda s,v:np.zeros(len(s)),road=None,road_hard=False,white_edge=edge)
   exec(compile(ast.Module(body=[f],type_ignores=[]),'edge-wiring','exec'),scope);self.assertLess(scope[name](np.array([[1.8,10.,0.,4.,0.]]))[0],0)
if __name__=='__main__':unittest.main(verbosity=2)
