"""Replay current sensor-derived motion evidence; never alter raw boxes or scores."""

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
import argparse,copy,importlib.util,json,pickle,time
import numpy as np
from motion_observer import MotionObserver
from runtime_configuration import load_vehicle_config
from speed_policy import PlanningGuard
from time_local_motion import tube_boxes,paired_rectangle_gap
from planning_guard import rectangle_separation
from occupancy_protocol import unpack
parser=argparse.ArgumentParser();parser.add_argument('--optimize',type=int,nargs='*',default=[]);args=parser.parse_args()
source=HERE.parent/'bridge/closed_loop_integration/runs/20261107T064934Z/uniad_worker';output=HERE/'envelope_evidence';output.mkdir(exist_ok=True);observer=MotionObserver();v=load_vehicle_config();rows=[];reports={}
legacy_path=HERE.parent/'memory_lifecycle_v1/time_local_motion.py';spec=importlib.util.spec_from_file_location('legacy_tube',legacy_path);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
import motion_observer
new_qualify=motion_observer.qualify;old_observer=MotionObserver();legacy_reproductions=0
for index in range(12):
 folder=source/f'step_{index:03d}';report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()))
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 motion_observer.qualify=legacy.qualify;old_certificates,_=old_observer.observe(report,info)
 motion_observer.qualify=new_qualify;certificates,_=observer.observe(report,info)
 changed=copy.deepcopy(report)
 for obstacle in changed['obstacles']:
  original=obstacle.pop('geometry_time_local_motion',None);candidate=obstacle['candidate_index']
  if original:
   reference=old_certificates[candidate]
   np.testing.assert_allclose([reference['speed_lower_mps'],reference['speed_upper_mps']],[original['speed_lower_mps'],original['speed_upper_mps']],rtol=0,atol=1e-10);legacy_reproductions+=1
  if candidate in certificates:obstacle['geometry_time_local_motion']=certificates[candidate]
 reports[index]=(changed,info)
 (output/f'step_{index:03d}_guard.json').write_text(json.dumps(changed,indent=2)+'\n')
 if index not in [10,11]:continue
 saved=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));states=np.array(saved['braking_fallback']['states']);times=np.arange(len(states))*v.dt;old_guard=PlanningGuard(report,v,states,times);new_guard=PlanningGuard(changed,v,states,times)
 row=dict(step=index,objects=[],old_full_proposal_gap_m=float(old_guard.full_clearance(states,times).min()),new_full_proposal_gap_m=float(new_guard.full_clearance(states,times).min()),same_saved_fallback=True,new_closed_loop_executed=False)
 for k,entry in enumerate(r for r in new_guard.rows if r['hard_constraint']):
  if k not in old_guard.local_motion or k not in new_guard.local_motion:continue
  before=old_guard.local_motion[k];after=new_guard.local_motion[k]
  # Independent NumPy matrix-SAT check against production paired SAT.
  def gaps(cert):
   boxes=tube_boxes(new_guard.hard_boxes[k],cert,times,new_guard.hard_radii[k]);paired=paired_rectangle_gap(states,boxes,v.half_width_m,v.half_length_m)
   independent=np.diag(rectangle_separation(states,boxes,np.zeros(len(times)),v.half_width_m,v.half_length_m));np.testing.assert_allclose(paired,independent,atol=1e-10,rtol=0)
   return paired-new_guard.hard_margins[k]
  old_gap=gaps(before);new_gap=gaps(after)
  np.testing.assert_array_equal(old_guard.hard_boxes,new_guard.hard_boxes);np.testing.assert_array_equal(old_guard.hard_margins,new_guard.hard_margins)
  row['objects'].append(dict(candidate_index=entry['candidate_index'],old_interval=[before['speed_lower_mps'],before['speed_upper_mps']],new_interval=[after['speed_lower_mps'],after['speed_upper_mps']],sources=after['longitudinal_velocity_sources_mps'],padding=after['velocity_interval_padding_mps'],old_gap_m=float(old_gap[1:].min()),new_gap_m=float(new_gap[1:].min()),acceleration_unchanged=before['actor_acceleration_budget_mps2']==after['actor_acceleration_budget_mps2']==10.,independent_sat_matches=True))
 rows.append(row)
summary=dict(passed=legacy_reproductions>0,legacy_motion_certificates_reproduced=legacy_reproductions,rows=rows,scope='Current recorded fallback/proposal geometry only; not complete planning acceptance.',new_gpu_executed=False)
(output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if args.optimize:
 import sys
 sys.path.insert(0,str(HERE.parent/'bridge/closed_loop_integration'))
 import run_driver
 run_driver.configure()
 from geometry_acceleration import accelerated
 accelerated.install()
 from proposal_geometry_audit import install
 install(run_driver.optimized_controller.PlanningGuard)
 from empty_controller import optimize
 for index in args.optimize:
  folder=source/f'step_{index:03d}';report,info=reports[index]
  with np.load(folder/'perception.npz') as z:plan=z['trajectory'].copy();occ,contract=unpack(z)
  info['occupancy_contract']=contract;previous=_hugsim_expand(json.loads((source/f'step_{index-1:03d}/optimization.json').read_text())).get('selected') if index else None;warm=None if previous is None else np.array(previous['controls'])
  start=time.monotonic();result=optimize(plan,occ,info,warm=warm,maxiter=220,reliability=report,vehicle=v,_warm_elapsed_s=.1);wall=time.monotonic()-start
  (output/f'step_{index:03d}_result.json').write_text(json.dumps(result,indent=2)+'\n');selected=result.get('selected');brief=dict(step=index,wall_seconds=wall,selected=selected is not None,forecast_recovery=result.get('forecast_recovery'),reason=result.get('reason'),minimum_constraint=None if selected is None else selected['minimum_constraint'],first_control=None if selected is None else selected['controls'][0],new_gpu_executed=False)
  (output/f'step_{index:03d}_summary.json').write_text(json.dumps(brief,indent=2)+'\n');print(json.dumps(brief),flush=True)
