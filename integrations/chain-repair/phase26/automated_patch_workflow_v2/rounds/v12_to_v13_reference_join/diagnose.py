
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

import ast,copy,sys,pickle,json,time,types
pass  # Path is provided by the portable boundary above
import numpy as np
ROOT=Path(_hugsim_expand('${HUGSIM_ROOT}'));LOCAL=ROOT/'integrations/light-intent-recovery-v12-r2'
sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,Motion,JointReference,P26,read,write,Tracker,track_reference
import joint_safety
run=P26/'patch_v12_validation_r2_v1/cases/cutin551/on/B';report=read(run/'run_report.json');mapping=read(run.parents[0]/'map/map.json')
source=(LOCAL/'joint_safety.py').read_text();old='base=origin[0]+(base-origin[0])*smooth(u)+.5*initial_curvature*join_length**2*u*u*(1-u)**3'
new='''distance=y-origin[1]
        raw_slope=0. if np.any(np.diff(knots)<=0) else (xs[1]-xs[0])/(knots[1]-knots[0])
        base=base-raw_slope*distance*(1-smooth(u))+.5*initial_curvature*distance**2*(1-u)**3'''
assert source.count(old)==1
scope=dict(vars(joint_safety));tree=ast.parse(source.replace(old,new));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointReference');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='guides');exec(compile(ast.Module(body=[node],type_ignores=[]),str(LOCAL/'joint_safety.py'),'exec'),scope);guides=scope['guides']
r=JointReference(V,mapping);rows=[]
for t in report['trace']:
 i=t['index'];step=run/f'uniad_worker/step_{i:03d}'
 with (step/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(step/'perception.npz') as z:arrays={k:z[k].copy() for k in z.files}
 m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state'];m.last_accel=0 if not i else report['trace'][i-1]['action']['acc_mps2']
 prior=copy.deepcopy(r) if i in (150,152,154,160,165,170,175,180,187) else None
 out,s=r.process(arrays['trajectory'],arrays,{},info,m)
 if prior is None:continue
 alternative=prior;alternative.guides=types.MethodType(guides,alternative);new_out,n=alternative.process(arrays['trajectory'],arrays,{},info,m)
 def evaluate(planner,out,selection):
  box,_=planner.actor_boxes();target=selection['selected_candidate']['speed_mps'] if selection['selected_candidate'] else 4.
  guide=planner.guides(arrays['trajectory'],box,m,target)[0];knots=np.r_[-planner.mount[1,3],arrays['trajectory'][:,1]];values=np.r_[-planner.mount[0,3],arrays['trajectory'][:,0]]
  y=np.linspace(knots[0],knots[-1],100);truth=np.interp(y,knots,values);proposed=np.interp(y,guide['y'],guide['x'])
  nominal,c=track_reference(Tracker(V),planner,out,info,m);action,gate=planner.admit(m,nominal)
  return dict(side=None if selection['selected_candidate'] is None else selection['selected_candidate']['side'],guide_error_rms_m=float(np.sqrt(np.mean((truth-proposed)**2))),guide_x_at_5m=float(np.interp(5.,guide['y'],guide['x'])),action=action,gate=gate['selected'],valid=selection['admitted_candidates'],reference_certificate=selection['reference_certificate'])
 rows.append(dict(index=i,old=evaluate(r,out,s),new=evaluate(alternative,new_out,n)))
 print(i,rows[-1],flush=True)
write(Path(__file__).parent/'reference_join_counterfactual.json',dict(source_report=str(run/'run_report.json'),rows=rows,GPU_executed=False,counterfactual_closed_loop=False,only_change='current-reference boundary interpolation'))
