"""Actual healthy skipped-head packet, cold planning context, CPU only."""

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
import json,pickle,time,numpy as np
from run_driver import configure
from forecast_memory import frame_health
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
from motion_policy import RecoveryController
from drivable_area import DrivableArea
configure();source=CR/'phase31/runs/occupancy_closed551_20261105T140958Z_48dc4599/closed_loop/uniad_worker';v=load_vehicle_config();chosen=None
for folder in sorted(source.glob('step_[0-9][0-9][0-9]')):
 audit_path=folder/'occupancy_head_audit.json';guard_path=folder/'guard_memory_report.json'
 if not audit_path.exists() or not guard_path.exists():continue
 audit=_hugsim_expand(json.loads(audit_path.read_text()))
 if audit.get('motion_query_count')!=0:continue
 report=_hugsim_expand(json.loads(guard_path.read_text()));depth=report.get('observed_depth_geometry',{})
 if not depth.get('depth_available') or not depth.get('ground_available') or not np.any(DrivableArea(report['predicted_drivable_area'],v).mask):continue
 with np.load(folder/'perception.npz') as data:
  occupancy,contract=unpack(data);plan=data['trajectory'].copy();candidates=[data['candidate_'+key].copy() for key in ['boxes','scores','labels']]
 if occupancy is not None:continue
 with (folder/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
 del obs
 info['occupancy_contract']=contract;info['perception_health']=frame_health(info['timestamp'],audit,plan,candidates,True);chosen=folder;break
if chosen is None:raise RuntimeError('No healthy saved agentless packet')
controller=RecoveryController();h._active_recovery_controller=controller;start=time.monotonic();result=h.optimize(plan,None,info,reliability=report,vehicle=v,warm=np.zeros((6,2)),maxiter=20);wall=time.monotonic()-start;action,decision=controller.update(result,info['ego_velo'],info['timestamp'],v.dt,v.emergency_braking_mps2)
assert result['planning_input_source']=='fresh_agentless_foreground'
assert result['occupancy_input_audit']['status']=='unavailable'
assert not result['perception_context_audit']['raster_fabricated']
summary={'source_step':chosen.name,'wall_seconds':wall,'planning_input_source':result['planning_input_source'],'raw_current_status_preserved':result['occupancy_input_audit']['status'],'selected':result['selected'] is not None,'reason':result.get('reason'),'minimum_constraint':(result.get('selected') or {}).get('minimum_constraint'),'action':action,'decision':decision,'current_raster_fabricated':False,'current_geometry_and_candidates_retained':True,'new_gpu_executed':False,'new_closed_loop_executed':False};(HERE/'agentless_replay_result.json').write_text(json.dumps(result,indent=2)+'\n');(HERE/'agentless_replay_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
