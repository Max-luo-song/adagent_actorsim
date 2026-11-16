
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

import sys,pickle,unittest
pass  # Path is provided by the portable boundary above
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
 def test_large_initial_slope_cannot_reverse_current_intent(self):
  y=self.grid();x=np.interp(y,[0,2,30],[0,3,3]);out=join_reference(x,y,[0,0],12.,0.);previous=self.old(x,y)
  self.assertTrue(np.all(out>=previous-1e-12));self.assertTrue(np.all(out<=x+1e-12))
  self.assertGreater(out[np.argmin(abs(y-5.))],0.)
 def test_residual_stays_between_previous_and_current_curve(self):
  y=self.grid()
  for sign in (-1,1):
   x=sign*(.05*np.sin(y)+.2*(1-np.exp(-np.maximum(y-1,0)**2)));out=join_reference(x,y,[0,0],12.,0.);previous=self.old(x,y)
   self.assertTrue(np.all(out>=np.minimum(previous,x)-1e-12));self.assertTrue(np.all(out<=np.maximum(previous,x)+1e-12))
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
  write(P26/'light_tracking_curvature_v15/checks/v12_reference_join_replay.json',dict(rows=rows,GPU_executed=False,counterfactual_closed_loop=False))
  self.assertTrue(rows);self.assertLess(np.mean([x['candidate']['rms_m'] for x in rows]),np.mean([x['baseline']['rms_m'] for x in rows])*.5)
if __name__=='__main__':unittest.main(verbosity=2)
