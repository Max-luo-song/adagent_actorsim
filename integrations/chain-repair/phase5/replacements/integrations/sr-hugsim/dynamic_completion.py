"""Post-hoc termination for a bounded dynamic diagnostic; never planner input."""
import numpy as np
from closed172_common import body_vertices, asset_reference_vertices
from waymo_adapter import SR_TO_HUGSIM as T


def evaluate_completion(report, asset, vehicle, confirmation_s=.5):
    observations=report.get('observations',[])
    result=dict(success=False,reason='not_yet_passed',confirmation_seconds=confirmation_s,
                ground_truth_used_for_control=False)
    if not observations:return result
    if any(row.get('background_frozen',False) for row in observations):
        return dict(result,reason='source_time_frozen')
    poses=[T@np.asarray(row['closed_ego_to_world'])@T.T for row in observations]
    forward=poses[0][[0,2],2];forward=forward/np.linalg.norm(forward)
    far=float(np.max(asset_reference_vertices(asset)[:,[0,2]]@forward))
    passed=[float(np.min(body_vertices(pose,vehicle)[:,[0,2]]@forward))>far for pose in poses]
    last=len(passed)-1
    if not passed[last]:return result
    first=last
    while first and passed[first-1]:first-=1
    tail=observations[first:]
    duration=float(tail[-1]['timestamp']-tail[0]['timestamp'])
    gap=min(row['inserted_reference_gap_m'] for row in observations)
    native=report.get('minimum_native_reference_gap_m')
    safe=gap>=vehicle.clearance_margin_m and native is not None and native>=vehicle.clearance_margin_m
    if report.get('actor_reference_groups'):
        actor_gap=report.get('minimum_all_actor_reference_gap_m')
        safe=safe and actor_gap is not None and actor_gap>=vehicle.clearance_margin_m
    moving=all(row['closed_speed_mps']>=.5 for row in tail)
    positions=np.array([pose[[0,2],3] for pose in poses])
    lateral=np.array([forward[1],-forward[0]])
    lateral_displacement=float(np.max(abs((positions-positions[0])@lateral)))
    maneuver=lateral_displacement>=.5
    success=duration>=confirmation_s-1e-8 and safe and moving and maneuver
    return dict(result,success=bool(success),reason='passed_and_continued_safely' if success else 'awaiting_safe_continuation',
                continuous_pass_duration_s=duration,clearance_satisfied=bool(safe),continued_motion=bool(moving),lateral_displacement_m=lateral_displacement,
                lateral_maneuver_observed=bool(maneuver))
