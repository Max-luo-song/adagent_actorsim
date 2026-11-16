
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

import sys,json,pickle
pass  # Path is provided by the portable boundary above
import numpy as np
ROOT=Path(_hugsim_expand('${HUGSIM_ROOT}'))
LOCAL=ROOT/'integrations/light-risk-provenance-v11'
sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from test_repair import V,JointReference,Motion,Tracker,track_reference,P26,read,write
p=P26/'light_evidence_control_v10/B';report=read(p/'run_report.json');cached=read(P26/'light_risk_provenance_v11/checks/v10_failure_replay.json')['rows'];r=JointReference(V,read(p.parent/'map/map.json'));rows=[];tracker=Tracker(V)
for t in report['trace']:
 i=t['index'];folder=p/f'uniad_worker/step_{i:03d}'
 with (folder/'sensor_metadata.pkl').open('rb') as f:inf=pickle.load(f)
 with np.load(folder/'perception.npz') as z:a={k:z[k].copy() for k in z.files}
 if i in (100,110,145,187):
  m=Motion(.1,2.7,t['motion_state'][3]);m.x,m.z,m.yaw,m.speed,m.steer=t['motion_state'];m.last_accel=0 if i==0 else report['trace'][i-1]['action']['acc_mps2']
  out,s=r.process(a['trajectory'],a,{},inf,m);nominal,c=track_reference(tracker,r,out,inf,m);action,gate=r.admit(m,nominal)
  assert action is not None,(i,s['mode'],gate)
  if i>=110:assert action['acc_mps2']>0,(i,action,gate)
  rows.append(dict(index=i,saved_speed=m.speed,action=action,gate=gate['selected'],certificate=gate['certificate'],smoothness=gate['execution_smoothness']))
 else:r.observed({},inf,a);r.update_tracks(a,inf)
 current={x['id']:x for x in cached[i]['tracks']}
 for track in r.tracks:
  track['latched']=current.get(track['id'],{}).get('latched',False)
 selected=cached[i]['selected']
 if selected:r.previous_target_speed=selected['speed_mps'];r.side=selected['side']
write(P26/'light_risk_provenance_v11/checks/restart_probe.json',dict(passed=True,rows=rows,real_ilqr=True,saved_state_replay=True,GPU_executed=False,counterfactual_closed_loop=False))
print(json.dumps(rows))
