
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

import sys,json,pickle,copy,importlib.util,hashlib
pass  # Path is provided by the portable boundary above
import numpy as np
R=Path(_hugsim_expand('${HUGSIM_ROOT}'));C=R/'integrations/light-agent-smoke-20261112t101312';sys.path[:0]=[str(C/'tests'),str(C)]
from test_repair import V,Motion,JointReference,P26,read,write,Tracker,track_reference
spec=importlib.util.spec_from_file_location('_frozen_v13_replay',R/'integrations/light-reference-join-v13-r2/joint_safety.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
rows=[]
for case,arm in [('original551','A'),('cutin551','B')]:
 p=P26/f'patch_v13_validation_r2_v1/cases/{case}/on/{arm}';report=read(p/'run_report.json');r=base.JointReference(V,read(p.parent/'map/map.json'))
 for t in report['trace']:
  i=t['index'];folder=p/f'uniad_worker/step_{i:03d}'
  with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
  with np.load(folder/'perception.npz') as z:arrays={k:z[k].copy() for k in z.files}
  m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state'];m.last_accel=0 if i==0 else report['trace'][i-1]['action']['acc_mps2']
  probe=i in (0,20,40,60,70,80,90,97,100,110,120,130,132,135,140,150,160,170,180,187)
  candidate=copy.deepcopy(r) if probe else None
  old,selection=r.process(arrays['trajectory'],arrays,{},info,m)
  if not probe:continue
  candidate.__class__=JointReference;new,chosen=candidate.process(arrays['trajectory'],arrays,{},info,m)
  vals=[]
  for obj,out,sel in [(r,old,selection),(candidate,new,chosen)]:
   nom,_=track_reference(Tracker(V),obj,out,info,m);action,gate=obj.admit(m,nom)
   s=sel['selected_candidate'];boxes,_=obj.actor_boxes();target=s['speed_mps'] if s else m.speed
   guide=obj.guides(arrays['trajectory'],boxes,m,target)[0];knots=np.r_[-obj.mount[1,3],arrays['trajectory'][:,1]];xs=np.r_[-obj.mount[0,3],arrays['trajectory'][:,0]];y=np.linspace(knots[0],knots[-1],100)
   selected_states=sel.get('selected_rollout_states')
   lateral=None if selected_states is None else float(max(row[3]**2*abs(np.tan(row[4]))/V.wheelbase_m for row in selected_states[:31]))
   vals.append(dict(max_prefix_lateral_acceleration_mps2=lateral,mode=sel['mode'],selected=s,action=action,execution_gate=gate['selected'],raw_tracking_rms_m=float(np.sqrt(np.mean((np.interp(y,guide['y'],guide['x'])-np.interp(y,knots,xs))**2))),selected_prefix_passed=s['prefix_certificate']['passed'] if s else None))
  row=dict(case=case,index=i,baseline=vals[0],candidate=vals[1]);rows.append(row);print(case,i,vals[0]['mode'],vals[1]['mode'],flush=True)
fail=[dict(case=x['case'],index=x['index']) for x in rows if x['baseline']['action'] is not None and x['candidate']['action'] is None or x['baseline']['selected'] is not None and x['candidate']['selected'] is None]
lateral_regressions=[dict(case=x['case'],index=x['index']) for x in rows if x['baseline']['max_prefix_lateral_acceleration_mps2'] is not None and x['candidate']['max_prefix_lateral_acceleration_mps2'] is not None and x['candidate']['max_prefix_lateral_acceleration_mps2']>max(2.,x['baseline']['max_prefix_lateral_acceleration_mps2'])+1e-7]
fail+=lateral_regressions
result=dict(lateral_acceleration_regressions=lateral_regressions,schema='matched_observation_v13_v16_replay',rows=rows,passed=not fail,new_admission_or_candidate_losses=fail,GPU_executed=False,counterfactual_closed_loop=False,scope='Same v13 pre-observation history and recorded sensor inputs; the candidate keeps v13 geometry selection and requires immediate and 3-second execution benefit. Does not predict counterfactual pixels or scores.')
write(P26/'light_agent_smoke_20261112t101312/checks/v13_matched_replay.json',result)
raise SystemExit(1 if fail else 0)
