
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

import copy,json,pickle,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
from resume_store import *
from run_resume import install

class Fixture:
 def __init__(self,out,source=None,commit=None,scene='generation-A:1',bad=None,fail=None):
  self.out=Path(out);self.out.mkdir();self.store=ReplayStore(out,'frozen',source,commit);self.scene=scene;self.solved=[];self.history=[];self.bad=bad;self.fail=fail
  self.h=types.SimpleNamespace();self.base=types.SimpleNamespace();self.driver=types.SimpleNamespace(h=self.h,base=self.base)
  def solver(reference,occupancy,info,*args,**kwargs):
   i=int(info['timestamp']*10);self.solved.append(i)
   if i==self.fail:raise RuntimeError('Simulated interrupted optimizer')
   return dict(selected=dict(controls=np.array([[i+.1,0.]])),scene=info['scene_token'])
  self.base.boundary_optimize=solver
  def outer(reference,occupancy,info,*args,**kwargs):
   self.history.append(int(info['timestamp']*10))
   # A missing-head/edge step still reconstructs outer memory without MPC.
   result=self.base.boundary_optimize(reference,occupancy,info,*args,**kwargs) if int(info['timestamp']*10)%2==0 else dict(selected=None,reason='bounded_braking')
   return dict(result,history=list(self.history))
  self.h.optimize=outer
  self.h.atomic_json=lambda p,d:atomic_json(p,d)
  def packet(rendered):
   i=rendered['i'];info=dict(timestamp=i/10.,scene_token=self.scene,ego_velo=4.,model_lidar_frame={'virtual_to_ego':np.eye(4)})
   obs={'rgb':{'CAM_FRONT':np.full((2,2,3),i,np.uint8)},'depth':{'CAM_FRONT':np.full((2,2),1.,np.float32)}}
   if self.bad=='pixels' and i==0:obs['rgb']['CAM_FRONT'][0,0,0]+=1
   return obs,info
  self.packet=packet
  def loop(ctx,out,session,report,count):
   trace=[]
   for i in range(count):
    obs,info=ctx['model_packet']({'i':i});f=self.out/'uniad_worker'/f'step_{i:03d}';f.mkdir(parents=True,exist_ok=True);save_pickle(f/'observation.pkl',(obs,info));(f/'perception.npz').write_bytes(b'fixture-neural-arrays')
    ref=np.array([[0.,1.]],np.float64)
    if self.bad=='reference' and i==0:ref[0,0]=1e-10
    kwargs=dict(warm=np.zeros((6,2)),reliability=dict(road=True))
    if self.bad=='warm' and i==0:kwargs['warm'][0,0]=.01
    result=self.h.optimize(ref,np.zeros((1,5,2,2),np.uint8),info,**kwargs)
    for name,obj in [('guard_memory_report.json',kwargs['reliability']),('optimization.json',{'ok':True}),('control_decision.json',{'state_after':'normal'})]:atomic_json(f/name,obj)
    trace.append(dict(step=i,timestamp=i/10.,action={'acc':0.,'steer_rate':0.},state_before=[0.,i,0.,4.,0.],state_after=[0.,i+1,0.,4.,0.],control_decision={'history':result['history']},neural_plan_raw=ref.tolist(),physics_integration_verified=True,predicted_road_swept_clearance_m=1.))
    self.h.atomic_json(self.out/'run_report.json',dict(trace=copy.deepcopy(trace)))
   return trace
  self.h._control_loop=loop;install(self.driver,self.store)
 def run(self,count):return self.h._control_loop(dict(model_packet=self.packet,spec=types.SimpleNamespace(fps=10.),setup={'start':0}),self.out,None,{},count)

class ResumeTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def source(self,n=2):
  f=Fixture(self.root/'attempt_001');f.run(n);return f,load_commit(f.out,'frozen')
 def test_restart_rebuilds_memory_skips_old_solves_and_switches_live(self):
  old,doc=self.source(3);before={str(f.relative_to(old.out)):digest(f) for f in old.out.rglob('*') if f.is_file()};fresh=Fixture(self.root/'attempt_002',old.out,doc,'generation-B:1');fresh.run(5)
  self.assertEqual(old.solved,[0,2]);self.assertEqual(fresh.solved,[4]);self.assertEqual(fresh.history,list(range(5)));self.assertEqual(load_commit(fresh.out,'frozen')['completed_controls'],5)
  self.assertEqual(before,{str(f.relative_to(old.out)):digest(f) for f in old.out.rglob('*') if f.is_file()})
 def test_half_step_crash_keeps_previous_commit(self):
  f=Fixture(self.root/'attempt_001',fail=2)
  with self.assertRaises(RuntimeError):f.run(4)
  doc=load_commit(f.out,'frozen');self.assertEqual(doc['completed_controls'],2);self.assertTrue((f.out/'uniad_worker/step_002/resume_outer_input.pkl').exists());self.assertNotIn('uniad_worker/step_002/resume_outer_input.pkl',doc['files'])
 def test_restarts_more_than_once(self):
  a,doc=self.source(2);b=Fixture(self.root/'attempt_002',a.out,doc,'generation-B:1');b.run(3);doc=load_commit(b.out,'frozen');c=Fixture(self.root/'attempt_003',b.out,doc,'generation-C:1');c.run(5);self.assertEqual(c.solved,[4]);self.assertEqual(c.history,list(range(5)))
 def test_source_selects_most_advanced_commit_not_latest_empty_attempt(self):
  a,doc=self.source(2);(self.root/'attempt_999').mkdir();(self.root/'attempt_999/resume_commit.json.tmp.crash').write_text('{');p,d=select_source(self.root,'frozen');self.assertEqual(p,a.out);self.assertEqual(d['completed_controls'],2)
 def test_pixel_change_fails_before_any_solver(self):
  a,doc=self.source();b=Fixture(self.root/'attempt_002',a.out,doc,'generation-B:1',bad='pixels')
  with self.assertRaisesRegex(RuntimeError,'observation/pose'):b.run(3)
  self.assertEqual(b.solved,[])
 def test_reference_change_even_sub_nanometre_is_not_rounded(self):
  a,doc=self.source();b=Fixture(self.root/'attempt_002',a.out,doc,'generation-B:1',bad='reference')
  with self.assertRaisesRegex(RuntimeError,'optimizer outer input'):b.run(3)
 def test_changed_warm_start_not_reused(self):
  a,doc=self.source();b=Fixture(self.root/'attempt_002',a.out,doc,'generation-B:1',bad='warm')
  with self.assertRaisesRegex(RuntimeError,'optimizer outer input'):b.run(3)
 def test_corrupted_committed_file_rejected_before_unpickle(self):
  a,doc=self.source();p=a.out/'uniad_worker/step_000/resume_outer_input.pkl';p.write_bytes(b'corrupted')
  with self.assertRaisesRegex(RuntimeError,'Committed file changed'):load_commit(a.out,'frozen')
 def test_configuration_change_cannot_resume(self):
  a,doc=self.source()
  with self.assertRaisesRegex(RuntimeError,'version/config mismatch'):load_commit(a.out,'other')
 def test_dtypes_and_schema_flags_exact(self):
  a=np.array([1.],np.float64);b=a.astype(np.float32);self.assertNotEqual(canonical(a,self.root,'g:1'),canonical(b,self.root,'g:1'));self.assertNotEqual(canonical({'valid':True},self.root,'g:1'),canonical({'valid':1},self.root,'g:1'))
 def test_only_explicit_run_and_generation_identity_is_normalized(self):
  a={'path':str(self.root/'old/a'),'scene':'abc:1','geometry':np.arange(3)};b={'path':str(self.root/'new/a'),'scene':'xyz:1','geometry':np.arange(3)}
  require_same(a,b,run=self.root/'old',source=self.root/'new',scene='abc:1',source_scene='xyz:1',label='identity')
  b['geometry'][0]=9
  with self.assertRaises(RuntimeError):require_same(a,b,run=self.root/'old',source=self.root/'new',scene='abc:1',source_scene='xyz:1',label='geometry')
 def test_commit_trace_corruption_rejected(self):
  a,doc=self.source();doc['trace'][0]['action']['acc']=1.;atomic_json(a.out/'resume_commit.json',doc)
  with self.assertRaisesRegex(RuntimeError,'trace checksum'):load_commit(a.out,'frozen')
 def test_commit_cannot_skip_a_control(self):
  a=Fixture(self.root/'attempt_001');a.store.index=1
  with self.assertRaisesRegex(RuntimeError,'Nonsequential'):a.store.commit([{},{}])
if __name__=='__main__':unittest.main(verbosity=2)
