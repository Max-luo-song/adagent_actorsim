"""Read-only B collision diagnosis; archived-data geometry and CPU tracker probes."""

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

import copy,hashlib,json,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
from shapely.geometry import Polygon
HERE=Path(__file__).resolve().parent;EXPERIMENT=HERE.parents[1];B=EXPERIMENT/'B';ROOT=next(p for p in HERE.parents if (p/'code/HUGSIM').is_dir())
sys.path.insert(0,str(ROOT/'integrations/light-cutin-diagnostic'))
from tracker import reference,Tracker
from common import configure
report=_hugsim_expand(json.loads((B/'run_report.json').read_text()));association=_hugsim_expand(json.loads((B/'target_detection_association_reference_only.json').read_text()))['rows'];trace=report['trace'];geom=report['vehicle_geometry']

def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def polygon_from_box(box):
 x,y,z,w,l,height,yaw=box;c,s=np.cos(yaw),np.sin(yaw);p=np.array([[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2]])
 return Polygon(p@np.array([[c,s],[-s,c]])+np.array([x,y]))

def reference_overlaps(row,raw,info):
 s,_=reference(raw,info['model_lidar_frame']['virtual_to_ego']);pose=np.asarray(row['ego_to_world']);target=polygon_from_box(row['environment']['target_world']['box']);hl,hw=geom['half_length_m'],geom['half_width_m'];times=[];gaps=[]
 for j,state in enumerate(s):
  yaw=state[2];forward=np.array([np.cos(yaw),np.sin(yaw)]);right=np.array([-np.sin(yaw),np.cos(yaw)]);body=np.array([state[:2]+a*hl*forward+b*hw*right for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])
  body=np.c_[body[:,0],-body[:,1],np.zeros(4)];world=body@pose[:3,:3].T+pose[:3,3];p=Polygon(world[:,:2]);gaps.append(p.distance(target))
  if p.intersects(target):times.append(j*.1)
 return dict(sampled_future_overlap=bool(times),first_overlap_after_s=times[0] if times else None,minimum_distance_m=float(min(gaps)),scope='0.1s samples of reference footprints against already-stopped target only; not a separately executed raw-policy episode')

rows=[]
for i,row in enumerate(trace):
 with (B/f'uniad_worker/step_{i:03d}/sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 raw=np.asarray(row['raw_uniad']);adopted=np.asarray(row['adopted']);rref,_=reference(raw,info['model_lidar_frame']['virtual_to_ego']);aref,_=reference(adopted,info['model_lidar_frame']['virtual_to_ego']);a=association[i]
 def detection(key):
  d=a.get(key)
  if d is None:return None
  return dict(index=d['index'],score=d['score'],center_distance_m=d['xy_distance_m'],near_target=d['xy_distance_m']<=2.,velocity_right_forward_mps=d['box'][7:9],reference_only_not_certified_track_identity=True)
 out=dict(index=i,time_s=row['timestamp_s'],target_forward_m=row['environment']['target_relative_forward_left_up_m'][0],target_actual_speed_mps=row['environment']['target_world']['backward_difference_speed_mps'],actual_speed_mps=row['motion_state'][3],raw_reference_first_speed_mps=float(rref[1,3]),adopted_reference_first_speed_mps=float(aref[1,3]),acc_mps2=row['action']['acc_mps2'],raw_lateral_end_m=float(raw[-1,0]),adopted_lateral_end_m=float(adopted[-1,0]),forward_nodes_max_difference_m=float(abs(raw[:,1]-adopted[:,1]).max()),retained=detection('retained'),candidate=detection('candidate'),motion_query=detection('motion_query'),map_gate_override=bool(row['road_gate'].get('overridden',False)))
 if row['timestamp_s']>=5.3 and row['environment']['target_world']['backward_difference_speed_mps']<1e-4:
  out['raw_plan_target_overlap']=reference_overlaps(row,raw,info);out['adopted_plan_target_overlap']=reference_overlaps(row,adopted,info)
 rows.append(out)

# Use the actual unchanged iLQR at selected saved current states, without NN/GPU.
configure()
from scene_api import SceneSpec
from live_env import Motion
v=SceneSpec.read(EXPERIMENT/'scene_manifest.json').vehicle
controls=[]
for i in (115,120,125,130,135,140):
 row=trace[i]
 with (B/f'uniad_worker/step_{i:03d}/sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 motion=Motion(v.dt,v.wheelbase_m,row['motion_state'][3]);motion.x,motion.z,motion.yaw,motion.speed,motion.steer=row['motion_state']
 tracker=Tracker(v);adopted,ad=tracker.control(np.asarray(row['adopted']),info,copy.copy(motion));raw,rd=tracker.control(np.asarray(row['raw_uniad']),info,copy.copy(motion));delta=max(abs(adopted[k]-row['action'][k]) for k in adopted)
 if delta>1e-7:raise RuntimeError('Cannot reproduce actual adopted tracker action at '+str(i)+': '+str(delta))
 target=row['environment']['target_world']['box'];bumper_gap=row['environment']['target_relative_forward_left_up_m'][0]-v.half_length_m-target[4]/2;speed=motion.speed
 controls.append(dict(index=i,time_s=row['timestamp_s'],adopted_action_reproduced_max_error=delta,adopted_action=adopted,raw_reference_one_step_counterfactual=raw,single_step_only=True,target_bumper_gap_m=bumper_gap,constant_2mps2_stop_distance_m=speed**2/4.,scope='One-step reference/controller comparison; not a new closed loop or full-scene safety certificate'))

close=[r for r in rows if r['retained'] and r['retained']['near_target']];terminal=report['terminal_collision'];relevant={k:report['worker_ready'].get(k) for k in ('original_collision_postprocessor','thresholds','planning_bev','cameras')}
result=dict(experiment=str(EXPERIMENT),run_report_sha256=h(B/'run_report.json'),association_sha256=h(B/'target_detection_association_reference_only.json'),termination_reason=report['termination_reason'],terminal_collision=terminal,controls=len(trace),observations=len(report['observations']),worker_ready=relevant,forward_nodes_preserved_all_frames=all(r['forward_nodes_max_difference_m']==0 for r in rows),near_retained_detection_frames=len(close),near_retained_intervals_note='Reference-only nearest-center association; do not infer persistent neural track identity',tracker_control_replay=controls,rows=rows,GPU_executed=False,new_closed_loop_executed=False,patch_adopted=False)
(HERE/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('near retained count',len(close));print('terminal collision',terminal);print('worker_ready',relevant);print('raw reference first future overlap frame',next((r['index'] for r in rows if r.get('raw_plan_target_overlap',{}).get('sampled_future_overlap')),None));print('control probes',controls,flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
fig=plt.figure(figsize=(15,10));gs=fig.add_gridspec(3,2,height_ratios=[1.3,1,1])
for col,i in enumerate((120,140)):
 ax=fig.add_subplot(gs[0,col]);ax.imshow(Image.open(B/f'steps/{i:03d}/CAM_FRONT.png'));ax.set_axis_off();ax.set_title(f'B at {i/10:.1f}s: '+('nearby retained car detection, score 0.546' if i==120 else 'near-target retained detection lost'))
t=np.array([r['time_s'] for r in rows]);ax=fig.add_subplot(gs[1,0]);ax.plot(t,[r['actual_speed_mps'] for r in rows],label='Executed ego speed');ax.plot(t,[r['adopted_reference_first_speed_mps'] for r in rows],label='Adopted first reference speed');ax.plot(t,[r['target_actual_speed_mps'] for r in rows],label='Target actual speed (post hoc)',alpha=.7);ax.set_ylim(-1,24);ax.set_ylabel('m/s');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=fig.add_subplot(gs[1,1]);ax.plot(t,[r['target_forward_m'] for r in rows],label='Target center forward');ax.axhline(geom['half_length_m']+trace[-1]['environment']['target_world']['box'][4]/2,color='red',ls='--',label='Approx. body overlap distance');ax.set_ylabel('m');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=fig.add_subplot(gs[2,0]);scores=[r['retained']['score'] if r['retained'] and r['retained']['near_target'] else np.nan for r in rows];ax.scatter(t,scores,s=8,label='Retained detection within 2m of target');ax.set_ylim(0,1);ax.set_ylabel('Score');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=fig.add_subplot(gs[2,1]);ax.plot(t,[r['raw_lateral_end_m'] for r in rows],label='Raw UniAD 3s endpoint');ax.plot(t,[r['adopted_lateral_end_m'] for r in rows],label='After static-map alignment');ax.set_ylabel('Rightward endpoint (m)');ax.legend(fontsize=8);ax.grid(alpha=.2)
for ax in fig.axes[2:]:ax.axvline(5.3,color='gray',ls=':',lw=1);ax.axvline(15.12,color='red',ls=':',lw=1);ax.set_xlabel('Local simulation time (s)')
fig.suptitle('B collision: intermittent detection + insufficient braking/avoidance response');fig.text(.5,.01,'Nearest-center detection matching is post hoc. Blueprints/reference overlap are not independent closed-loop executions. No GPU rerun.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.03,1,.96));fig.savefig(HERE/'collision_evidence.png',dpi=140);plt.close(fig)
