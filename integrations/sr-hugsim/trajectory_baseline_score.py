"""Fixed, post-hoc diagnostic score. Never imported by perception/control.

Not an official benchmark or safety certificate. Compare equal scenarios,
assets, durations, time policies, vehicle configs and frozen scoring versions.
"""
import numpy as np
from closed172_common import body_vertices,asset_reference_vertices
from waymo_adapter import SR_TO_HUGSIM as T

SCORE_SPEC = dict(version='diagnostic_trajectory_baseline_v1', maximum=100,
    weights=dict(no_reference_collision=35.,clearance=25.,passing_progress=20.,comfort=10.,smoothness=10.),
    clearance='25 * clip(minimum reference gap / vehicle clearance margin, 0, 1)',
    passing_progress='20 * clip(rear-footprint forward progress / distance to asset far plane, 0, 1)',
    comfort='10 * (1 - mean(abs(acc)/acc_limit + abs(steer_rate)/rate_limit)/2)',
    smoothness='10 * (1 - mean(abs(delta acc)/acc_range + abs(delta steer_rate)/(2*rate_limit))/2)',
    completion='multiply total by executed/requested controls; reference collision forces total=0',
    pass_definition='entire ego footprint beyond asset far plane along initial ego forward axis',
    scope='2D inserted-asset proxy only; not all-object collision or drivable-area validation')


def score_report(report, asset, vehicle):
    observations=report.get('observations',[]);trace=report.get('trace',[])
    if not observations: return dict(spec=SCORE_SPEC,score=None,reason='No completed observation',comparable=False)
    poses=[T@np.asarray(o['closed_ego_to_world'],dtype=float)@T.T for o in observations]
    forward=poses[0][[0,2],2];forward=forward/np.linalg.norm(forward)
    target=asset_reference_vertices(asset)[:,[0,2]];far=float(np.max(target@forward))
    rears=np.array([np.min(body_vertices(pose,vehicle)[:,[0,2]]@forward) for pose in poses])
    required=max(far-rears[0],1e-8)
    progress=float(np.clip((np.max(rears)-rears[0])/required,0,1))
    passed_indices=np.flatnonzero(rears>far)
    gap=report.get('minimum_inserted_reference_gap_m')
    if gap is None:gap=min(o['inserted_reference_gap_m'] for o in observations)
    collision='overlap' in str(report.get('error','')).lower() or gap<=0
    controls=np.array([[row['action']['acc'],row['action']['steer_rate']] for row in trace],dtype=float).reshape(-1,2)
    acc_limit=max(abs(vehicle.min_acc_mps2),abs(vehicle.max_acc_mps2))
    rate_limit=np.deg2rad(vehicle.max_steer_rate_degrees)
    acc_range=vehicle.max_acc_mps2-vehicle.min_acc_mps2
    comfort_loss=float(np.mean(np.abs(controls)/[acc_limit,rate_limit])) if len(controls) else 1.
    smooth_loss=float(np.mean(np.abs(np.diff(controls,axis=0))/[acc_range,2*rate_limit])) if len(controls)>1 else 0.
    parts=dict(no_reference_collision=0. if collision else 35.,
        clearance=25*float(np.clip(gap/vehicle.clearance_margin_m,0,1)),
        passing_progress=20*progress,comfort=10*float(np.clip(1-comfort_loss,0,1)),
        smoothness=10*float(np.clip(1-smooth_loss,0,1)))
    completion=min(1.,len(trace)/max(1,report['steps_requested']))
    total=0. if collision else sum(parts.values())*completion
    elapsed=[o['timestamp']-observations[0]['timestamp'] for o in observations]
    speeds=np.array([o['closed_speed_mps'] for o in observations])
    xyz=np.array([pose[:3,3] for pose in poses])
    passed_index=int(passed_indices[0]) if len(passed_indices) else None
    after=observations[passed_index:] if passed_index is not None else []
    return dict(spec=SCORE_SPEC,score=round(total,3),components=parts,completion_fraction=completion,
        metrics=dict(reference_collision=bool(collision),minimum_reference_gap_m=float(gap),
            required_clearance_m=vehicle.clearance_margin_m,clearance_satisfied=bool(gap>=vehicle.clearance_margin_m),
            passed_asset_far_plane=passed_index is not None,pass_time_s=elapsed[passed_index] if passed_index is not None else None,
            seconds_observed_after_pass=elapsed[-1]-elapsed[passed_index] if passed_index is not None else 0.,
            mean_speed_after_pass_mps=float(np.mean([o['closed_speed_mps'] for o in after])) if after else None,
            initial_speed_mps=float(speeds[0]),final_speed_mps=float(speeds[-1]),
            distance_travelled_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
            rms_acceleration_mps2=float(np.sqrt(np.mean(controls[:,0]**2))) if len(controls) else None,
            rms_jerk_mps3=float(np.sqrt(np.mean((np.diff(controls[:,0])/vehicle.dt)**2))) if len(controls)>1 else None),
        baseline_conditions=dict(scene=report['scene'],start_frame=report['start_frame'],
            steps_requested=report['steps_requested'],dt=report['dt'],guard_policy=report['guard_policy'],
            time_policy=report.get('time_policy','source_time_only'),training=False),
        ground_truth_used_for_control=False,formal_evaluation_admitted=False)
