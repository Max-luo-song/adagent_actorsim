"""Post-hoc audit of recorded A; no controller, source data or scoring changes."""

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

import ast,json,os,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.spatial import cKDTree

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'integrations/light-cutin-diagnostic').is_dir())
sys.path.insert(0,str(ROOT/'integrations/light-cutin-diagnostic'))
from common import BUNDLE,SCORER,read,write,asset,configure
configure()
OUT=Path(__file__).resolve().parent;RUN=OUT.parents[1];A=RUN/'A'
r=read(A/'run_report.json');summary=read(A/'summary.json')
vehicle=read(ROOT/'waymo23-native-import/configs/vehicle.json')
rows=[]
for t in r['trace']:
    desired=np.asarray(t['control']['reference_states'])[1,:2]
    speed=t['motion_state'][3];dt=vehicle['dt']
    low=max(0,speed+vehicle['min_acc_mps2']*dt)*dt
    high=min(vehicle['max_speed_mps'],speed+vehicle['max_acc_mps2']*dt)*dt
    # Native Motion updates translation with pre-step heading. Steering cannot
    # change the current step's body-relative lateral displacement.
    achievable=np.array([np.clip(desired[0],low,high),0.])
    bound=float(np.linalg.norm(achievable-desired))
    rows.append(dict(index=t['index'],timestamp_s=t['timestamp_s'],speed_mps=speed,
        reference_speed_mps=t['control']['target_speed_mps'],reference_first_step_xy=desired.tolist(),
        min_reachable_forward_m=low,max_reachable_forward_m=high,
        minimum_possible_one_step_error_m=bound,actual_error_m=t['tracking_error']['position_m'],
        excess_over_physical_lower_bound_m=t['tracking_error']['position_m']-bound,
        applied_acc_mps2=t['action']['acc_mps2'],steer_rate_radps=t['action']['steer_rate_radps']))

tree=ast.parse(SCORER.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScoreCalculator')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_calculate_drivable_area_compliance')
scope={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(SCORER),'exec'),scope)
native_dac=scope[method.name]
bg=np.load(BUNDLE/'background_sr_world.npy');road=np.load(BUNDLE/'road_sr_world.npy')
with np.load(BUNDLE/'point_semantics/background_semantics.npz') as z:labels=z['class_id']
clouds={'pessimistic':np.concatenate([road,bg[labels==0]])[:,:2],
        'optimistic':np.concatenate([road,bg[(labels==0)|(labels==255)]])[:,:2]}
trees={k:cKDTree(v) for k,v in clouds.items()}
def coverage(pose):
    x,y=pose[:2,3];yaw=float(np.arctan2(pose[1,0],pose[0,0]));result={}
    for k,points in clouds.items():
        local=points[trees[k].query_ball_point([x,y],4.)]
        value=native_dac(None,local,np.array([[x,y,yaw]]),2*vehicle['half_width_m'],2*vehicle['half_length_m'])
        result[k]=dict(native_current_footprint_dac=value,nearest_ground_xy_distance_m=float(trees[k].query([x,y])[0]),ground_points_within_4m=len(local))
    return result
raw=Path(read(BUNDLE/'source_manifest.json')['raw_data']);origin=np.loadtxt(raw/'ego_pose/000.txt')
coverage_rows=[]
for o in r['observations'][:12]:
    i=o['index'];source=np.linalg.inv(origin)@np.loadtxt(raw/f'ego_pose/{i:03d}.txt')
    coverage_rows.append(dict(index=i,timestamp_s=o['timestamp'],executed=coverage(np.asarray(o['pose'])),
        original_recorded_ego=coverage(source),original_recorded_source_xy=source[:2,3].tolist()))
details=read(A/'native_eval/executed_path_pessimistic_native_details.json')['details']
result=dict(status='original_A_anomalies_diagnosed_B_remains_blocked',run=asset(A/'run_report.json'),
    original_admission=asset(RUN/'baseline_gpu_admission.json'),original_summary=asset(A/'summary.json'),
    original_native_scoring=asset(A/'native_eval/native_scores.json'),analysis_code=asset(__file__),
    fixed_controller_changed=False,model_changed=False,thresholds_or_initial_state_changed=False,
    actual_controls=summary['controls'],actual_simulation_s=summary['elapsed_simulation_s'],actual_wall_s=summary['wall_seconds'],
    actual_collision=summary['anomalies']['environment_collision'],max_route_distance_m=max(o['environment']['route_distance_m'] for o in r['observations']),
    speed_reference_feasibility=rows,initial_road_point_coverage=coverage_rows,
    dac_bad_keyframes={k:v for k,v in details.items() if v['dac']<1},
    executed_native_scores=summary['native_scores']['scores']['executed_path'],
    mechanism='Initial 16m/s state is incompatible with the much slower model reference under fixed -2m/s2 actuation; one-step longitudinal residual is physically unavoidable',
    road_caveat='DAC includes the current footprint and reconstructed ground coverage; missing support at original t0 is not proof of a physical road departure',
    unresolved=['Why frozen UniAD reference speed is low on scene551: domain/input-contract causality not yet established',
        'Independent surveyed road-boundary truth is unavailable; reconstructed DAC is only a proxy',
        'No scene-B response or patch-effectiveness evidence exists'],
    decision='Keep failed admission, preserve A, do not launch B or manufacture a B-specific patch; any revised initial condition/scenario must use a separately preregistered experiment')
write(OUT/'report.json',result)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8))
time=[v['timestamp_s'] for v in rows]
axes[0,0].plot(time,[v['speed_mps'] for v in rows],label='Actual speed')
axes[0,0].plot(time,[v['reference_speed_mps'] for v in rows],label='UniAD first-segment reference')
axes[0,0].set(xlabel='Simulation time (s)',ylabel='Speed (m/s)');axes[0,0].legend()
axes[0,1].plot(time,[v['actual_error_m'] for v in rows],label='Recorded error')
axes[0,1].plot(time,[v['minimum_possible_one_step_error_m'] for v in rows],'--',label='Actuator/kinematic lower bound')
axes[0,1].axhline(1.,color='r',linestyle=':',label='Frozen admission threshold')
axes[0,1].set(xlabel='Simulation time (s)',ylabel='One-step position error (m)');axes[0,1].legend()
p=np.asarray([o['pose'] for o in r['observations']]);route=np.load(BUNDLE/'native_dense_route.npy')
axes[1,0].plot(route[:,2,3],-route[:,0,3],label='Original route (evaluation only)')
axes[1,0].plot(p[:,0,3],p[:,1,3],label='A actual execution');axes[1,0].set_xlim(-5,75);axes[1,0].set_ylim(-5,5)
axes[1,0].set(xlabel='Source forward x (m)',ylabel='Source left y (m)');axes[1,0].legend()
pts=clouds['pessimistic'];pts=pts[(pts[:,0]>-5)&(pts[:,0]<22)&(abs(pts[:,1])<6)]
axes[1,1].scatter(pts[::max(1,len(pts)//20000),0],pts[::max(1,len(pts)//20000),1],s=1,label='Native known ground')
axes[1,1].plot(p[:12,0,3],p[:12,1,3],'r.-',label='A initial execution');axes[1,1].scatter([0],[0],s=60,marker='x',label='Shared original start')
axes[1,1].set(xlabel='Source forward x (m)',ylabel='Source left y (m)');axes[1,1].legend()
fig.suptitle('Recorded baseline A diagnosis; controller and admission criteria unchanged');fig.tight_layout()
fig.savefig(OUT/'baseline_diagnosis.png',dpi=160);fig.savefig(OUT/'baseline_diagnosis.svg');plt.close(fig)

from diagnostics import artifacts
presentation=artifacts(RUN,'A','A')  # Existing supported renderer, both rows disclose A.
write(OUT/'artifacts.json',dict(report=asset(OUT/'report.json'),figure=asset(OUT/'baseline_diagnosis.png'),
    single_arm_video=asset(presentation['video']),video_rows='Both rows are the same recorded baseline A; no B has run',
    actual_observations_reused=True,gpu_rerun=False))
print(json.dumps(dict(report=str(OUT/'report.json'),figure=str(OUT/'baseline_diagnosis.png'),video=presentation['video'],
    first_step=rows[0],initial_ground=coverage_rows[0]),indent=2))
