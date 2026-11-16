"""Recover exact optimizer arguments by replaying driver state over saved results."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import copy,json,pickle,sys
from types import SimpleNamespace
import numpy as np
from paths import *
from run_cutin import configure
SOURCE=CR/'phase43/runs/cutin551_20261109T040010Z_1411461'
def capture_all():
 driver=configure();h=driver.h;out=HERE/'validation/inputs';out.mkdir(parents=True,exist_ok=True)
 from occupancy_protocol import unpack
 from forecast_memory import frame_health
 from drivable_area import from_saved
 from runtime_configuration import load_vehicle_config
 report=_hugsim_expand(json.loads((SOURCE/'run_report.json').read_text()));vehicle=load_vehicle_config();captured={};index=0
 def intercept(reference,occupancy,info,*args,**kwargs):
  captured.update(reference=copy.deepcopy(reference),occupancy=copy.deepcopy(occupancy),info=copy.deepcopy(info),kwargs=copy.deepcopy(kwargs));return copy.deepcopy(original)
 driver.base.boundary_optimize=intercept
 for index in range(len(report['trace'])):
  folder=SOURCE/'uniad_worker'/('step_%03d'%index);original=_hugsim_expand(json.loads((folder/'optimization.json').read_text()))
  with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
  with np.load(folder/'perception.npz') as z:
   plan=z['trajectory'].copy();occupancy,contract=unpack(z);road=from_saved(z);candidates=[z['candidate_'+k].copy() for k in ('boxes','scores','labels')]
   if 'geometry_candidate_boxes' in z:candidates[0]=z['geometry_candidate_boxes'].copy()
  info['occupancy_contract']=contract;info['perception_health']=frame_health(info['timestamp'],_hugsim_expand(json.loads((folder/'occupancy_head_audit.json').read_text())),plan,candidates,road is not None)
  reliability=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));previous=report['trace'][index-1] if index else None;warm=None
  if previous is not None and previous['control_decision']['state_after'] in ('normal','caution','recovering','degraded'):
   selected=_hugsim_expand(json.loads((SOURCE/'uniad_worker'/('step_%03d'%(index-1))/'optimization.json').read_text())).get('selected');warm=None if selected is None else np.asarray(selected['controls'])
  state=report['trace'][index]['control_decision']['state_before']
  if state not in ('normal','caution'):raise RuntimeError('Nontrivial recovery state needs explicit snapshot reconstruction: '+state)
  h._active_recovery_controller=SimpleNamespace(state=state)
  captured.clear();h.optimize(plan,occupancy,info,warm=warm,reliability=reliability,vehicle=vehicle,_reuse_verified_warm=False)
  captured['source_step']=index;captured['source_result']=original
  with (out/('step_%03d.pkl'%index)).open('wb') as f:pickle.dump(captured,f,protocol=4)
 print('Captured exact current optimizer inputs:',len(report['trace']),flush=True)
if __name__=='__main__':capture_all()
