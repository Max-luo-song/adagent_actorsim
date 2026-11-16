
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
import argparse,json,pickle,time,numpy as np
from runtime_configuration import load_vehicle_config
from run_driver import configure
from occupancy_protocol import unpack
from motion_memory import ContractObstacleMemory
configure();parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=0);parser.add_argument('--memory-only',action='store_true');args=parser.parse_args();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop';spec=h.SceneSpec.read(source/'scene_manifest.json');memory=ContractObstacleMemory.from_source(spec.source);v=load_vehicle_config();rows=[]
with (source/'uniad_worker/step_000/sensor_metadata.pkl').open('rb') as stream:entry_info=pickle.load(stream)
entry_speed=float(entry_info['ego_velo'])
for index in range(args.step+1):
 folder=source/'uniad_worker'/('step_%03d'%index)
 with (folder/'sensor_metadata.pkl').open('rb') as stream:info=pickle.load(stream)
 with np.load(folder/'perception.npz') as saved:
  plan=saved['trajectory'].copy();occ,contract=unpack(saved);boxes=saved['geometry_candidate_boxes'].copy() if 'geometry_candidate_boxes' in saved else saved['candidate_boxes'].copy();scores=saved['candidate_scores'].copy();labels=saved['candidate_labels'].copy()
 raw=_hugsim_expand(json.loads((folder/'scene_reliability.json').read_text()));fresh=memory.update(raw,boxes,scores,labels,info,current_occupancy=np.asarray(occ).reshape(5,200,200)[0] if occ is not None else None);original=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));report=dict(original,**{k:fresh[k] for k in ['obstacles','obstacle_memory','global_evidence_receipt','geometry_rejections']});report.pop('velocity_evidence_gate_v2_applied',None);rows.append(dict(step=index,old_confirmed=original['obstacle_memory']['confirmed_tracks'],new_confirmed=fresh['obstacle_memory']['confirmed_tracks']))
memory_out=HERE/'memory_replays';memory_out.mkdir(exist_ok=True);(memory_out/('step_%03d_guard.json'%args.step)).write_text(json.dumps(report,indent=2)+'\n')
if args.memory_only:
 (HERE/'memory_replay_summary.json').write_text(json.dumps(dict(rows=rows,scope='Driver memory replay from immutable raw worker candidates; no new inference, no ground-truth target identities.',new_gpu_executed=False),indent=2)+'\n');print(json.dumps(rows,indent=2));raise SystemExit(0)
info['occupancy_contract']=contract;report['requested_cruise_speed_mps']=entry_speed;warm=None
if args.step:
 previous=_hugsim_expand(json.loads((source/'uniad_worker'/('step_%03d'%(args.step-1))/'optimization.json').read_text())).get('selected');warm=np.asarray(previous['controls']) if previous else None
start=time.perf_counter();result=h.optimize(plan,occ,info,warm=warm,reliability=report,vehicle=v,maxiter=120);wall=time.perf_counter()-start
out=HERE/'behavior_replays';out.mkdir(exist_ok=True);(out/('step_%03d_result.json'%args.step)).write_text(json.dumps(result,indent=2)+'\n');(out/('step_%03d_guard.json'%args.step)).write_text(json.dumps(report,indent=2)+'\n');selected=result.get('selected');old=_hugsim_expand(json.loads((folder/'optimization.json').read_text())).get('selected');summary=dict(step=args.step,selected=selected is not None,old_first_action=(old or {}).get('controls',[None])[0],new_first_action=(selected or {}).get('controls',[None])[0],minimum_constraint=(selected or {}).get('minimum_constraint'),wall_seconds=wall,restoration=result.get('guard',{}).get('feasibility_restoration'),memory_counts=rows,scope='Fresh CPU optimizer on saved old closed-loop observation and newly replayed perception memory. Not new closed-loop success.',new_gpu_executed=False);(out/('step_%03d_summary.json'%args.step)).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
