
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

import sys,json,pickle,copy
pass  # Path is provided by the portable boundary above
import numpy as np
R=Path(_hugsim_expand('${HUGSIM_ROOT}'));C=R/'integrations/light-geometry-speed-v16';sys.path[:0]=[str(C/'tests'),str(C)]
from test_repair import V,Motion,JointReference,P26,read,write
D=Path(__file__).resolve().parent;P=P26/'patch_v16_validation_v1/cases/cutin551/on/B';report=read(P/'run_report.json');r=JointReference(V,read(P.parent/'map/map.json'));method=r.retime_selected;captured={};rows=[]
def capture(states,controls,motion,target):
 captured.update(states=states.copy(),controls=controls.copy());return method(states,controls,motion,target)
r.retime_selected=capture
for t in report['trace']:
 i=t['index'];folder=P/f'uniad_worker/step_{i:03d}'
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(folder/'perception.npz') as z:arrays={k:z[k].copy() for k in z.files}
 m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state'];m.last_accel=0 if i==0 else report['trace'][i-1]['action']['acc_mps2']
 captured.clear();_,a=r.process(arrays['trajectory'],arrays,{},info,m);rt=a['spatial_retiming']
 if not rt['accepted']:continue
 original=r.states(arrays['trajectory']);boxes,tracks=r.actor_boxes();s=captured['states'];u=captured['controls'];ns=r.selected_states;nu=r.selected_controls
 old=r.retiming_cost(s[:31],u[:30],original,3.,boxes,tracks);new=r.retiming_cost(ns[:31],nu[:30],original,3.,boxes,tracks)
 row=dict(index=i,full_preview_gain=rt['baseline_cost']-rt['proposal_cost'],executed_prefix_gain=old-new,prefix_state_max_change=float(np.max(np.abs(ns[:31]-s[:31]))),prefix_speed_max_change_mps=float(np.max(np.abs(ns[:31,3]-s[:31,3]))),first_control_change=(nu[0]-u[0]).tolist());rows.append(row)
 print(i,'preview_gain',round(row['full_preview_gain'],5),'prefix_gain',round(old-new,9),'prefix_delta',round(row['prefix_state_max_change'],9),flush=True)
write(D/'prefix_diagnostic.json',dict(rows=rows,accepted=len(rows),prefix_worse=sum(x['executed_prefix_gain']< -1e-9 for x in rows),prefix_no_material_gain=sum(x['executed_prefix_gain']<=1e-6 for x in rows),same_recorded_v16_observations=True,counterfactual_closed_loop=False,GPU_executed=False))
