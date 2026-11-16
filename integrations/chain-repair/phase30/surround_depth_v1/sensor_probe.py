"""GPU sensor validation on saved ego poses. No reference truth enters memory."""

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

import copy,hashlib,json,pickle,sys,time,os
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from surround_depth_v1 import sensors
from surround_depth_v1.depth_memory import SurroundDepthMemory
from surround_depth_v1.native_guard import PlanningGuard
import run_harness
from uniad_bridge import to_uniad,ORDER
from native_actor_evaluation import NativeActorEvaluation
from planning_guard import sample_half_intervals,sweep_allowance
CR=HERE.parents[1]
S=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
def main():
 cfg=_hugsim_expand(json.loads((HERE/'probe.json').read_text()));output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
 request=_hugsim_expand(json.loads((HERE/'request.json').read_text()));h=run_harness.harness;arm=h._load_arm_request(request['arm_request']);setup=h._setup(arm,output)
 sensors.install();sensors.OUTPUT=output
 report=_hugsim_expand(json.loads((Path(cfg['source_run'])/'run_report.json').read_text()));memory=SurroundDepthMemory(setup['vehicle'])
 # This evaluator is used AFTER sensor-only tracking, exclusively for probe scoring.
 actors=NativeActorEvaluation.from_archive(dict(np.load(CR/'phase5/native552_actor_geometry.npz')))
 rows=[];restoration_gate=False;began=time.monotonic()
 from scene_api import SceneSession
 with SceneSession() as session:
  session.load_scene(output/'scene_manifest.json');session.edit(_hugsim_expand(json.loads(Path(arm['insert_request']).read_text())))
  for index in range(cfg['first_step'],cfg['last_step']+1):
   if time.monotonic()-began>cfg['maximum_wall_seconds']:raise TimeoutError('Sensor probe wall cap')
   old=Path(cfg['source_run'])/f'uniad_worker/step_{index:03d}';record=report['observations'][index];pose=np.asarray(record['closed_ego_to_world']);timestamp=record['timestamp'];frame=record['source_frame']
   path=old/'sensor_metadata.pkl'
   with (path if path.exists() else old/'observation.pkl').open('rb') as f:stored=pickle.load(f)
   old_info=stored if path.exists() else stored[1]
   packet=session.render(pose,timestamp)
   if index==cfg['first_step']:
    bare=sensors._ORIGINAL(session.backend,pose,timestamp)
    for name in ORDER:
     np.testing.assert_array_equal(packet['cameras'][name]['rgb'],bare['cameras'][name]['rgb'])
     np.testing.assert_array_equal(packet['cameras'][name]['depth'],bare['cameras'][name]['depth'])
    assert set(session.backend.env.cam_names.values())==set(ORDER)
    restoration_gate=True
   obs,info=to_uniad(packet,setup['mount'],speed_mps=old_info['ego_velo'],steering_rad=old_info['ego_steer'],command=old_info['command'],scene_token='surround-sensor-probe',dt=setup['vehicle'].dt)
   assert tuple(obs['rgb'])==ORDER and tuple(info['cam_params'])==ORDER
   digest=hashlib.sha256(pickle.dumps((obs,info),protocol=4)).hexdigest()
   xy,audit=memory.update(obs,info)
   assert digest==hashlib.sha256(pickle.dumps((obs,info),protocol=4)).hexdigest(),'Depth pipeline modified NN packet'
   dynamic=audit['dynamic_depth_envelopes']
   group=actors.groups[cfg['evaluation_only_actor_group']];k=cfg['evaluation_only_actor_index'];trans=group[0];center=trans[frame,k]@S.T;velocity=((trans[frame+1,k]-trans[frame,k])/.1)@S.T
   matches=[]
   for row in dynamic:
    error=float(np.linalg.norm(np.asarray(row['velocity_world_xy'])-velocity[:2]));distance=float(np.linalg.norm(np.asarray(row['center_world'])[:2]-center[:2]));radius=row['radius_m']+np.linalg.norm(group[2][k,:2])/2
    if distance<=radius and error<=cfg['maximum_velocity_error_mps']:matches.append(row)
   old_result=_hugsim_expand(json.loads((old/'optimization.json').read_text()));selected=old_result.get('selected');warning=None
   if matches and selected:
    guard=PlanningGuard.__new__(PlanningGuard);guard.vehicle=setup['vehicle'];guard.dynamic=matches
    states=np.asarray(selected['states']);times=np.arange(len(states))*setup['vehicle'].dt
    warning=float((guard.dynamic_clearance(states,times,sample_half_intervals(times))-sweep_allowance(states,setup['vehicle'])).min())
   rows.append(dict(step=index,source_frame=frame,supplemental_render_seconds=sensors.LATEST['render_wall_seconds'],moving_tracks=sum(r['track_state']=='confirmed_motion' for r in dynamic),provisional_tracks=sum(r['track_state']=='provisional' for r in dynamic),reference_target_motion_matches=len(matches),matched_target_old_plan_minimum_gap_m=warning,neural_packet_unchanged=True))
   (output/'progress.json').write_text(json.dumps(rows[-1],indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
   (output/f'depth_audit_step{index:03d}.json').write_text(json.dumps(audit,default=lambda x:np.asarray(x).tolist(),indent=2)+'\n')
 warnings=[r for r in rows if r['step']<=cfg['earliest_plan_warning_required_by_step'] and r['matched_target_old_plan_minimum_gap_m'] is not None and r['matched_target_old_plan_minimum_gap_m']< -1e-5]
 passed=restoration_gate and len(warnings)>=cfg['required_near_term_warnings'] and all(r['neural_packet_unchanged'] for r in rows)
 result=dict(launch_manifest_sha256=os.environ.get('SURROUND_LAUNCH_MANIFEST_SHA256'),passed=bool(passed),front_rgb_depth_restoration_exact=restoration_gate,nn_packet_unchanged=True,early_motion_warnings=len(warnings),first_early_warning_step=warnings[0]['step'] if warnings else None,rows=rows,wall_seconds=time.monotonic()-began,scope='Saved ego-pose GPU sensor probe; actor reference only used for post-hoc scoring; not a new closed-loop result',formal_evaluation_admitted=False)
 (output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('SENSOR PROBE PASSED:',passed,flush=True)
 if not passed:raise SystemExit('Sensor/motion gate failed; do not run closed-loop')
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  folder=Path(_hugsim_expand(json.loads((HERE/'probe.json').read_text()))['output'])
  if folder.exists():
   (folder/'error.json').write_text(json.dumps(dict(passed=False,error=repr(exc),closed_loop_started=False),indent=2)+'\n')
  raise
