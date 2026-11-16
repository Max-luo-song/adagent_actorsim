"""Feed physically generated 0.1s states to unchanged HUGSIM iLQR."""
import time
import numpy as np

def control(tracker,planner,raw,info,motion):
    if planner.selected_states is None:return tracker.control(raw,info,motion)
    model=planner.selected_states[:31].copy();mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'])
    states=np.c_[model[:,1]+mount[1,3],model[:,0]+mount[0,3],model[:,2:]]
    if states.shape!=(31,5) or not np.isfinite(states).all():raise ValueError('Invalid dense tracking reference')
    initial=np.array([0.,0.,0.,motion.speed,motion.steer]);began=time.perf_counter();solutions=tracker.solver.solve(initial,states)
    if not solutions:raise RuntimeError('Existing iLQR returned no solution')
    controls=np.asarray(solutions[-1].input_trajectory);action=tracker.helpers['clip_action'](*controls[0],motion,tracker.vehicle)
    return action,dict(controller='fixed_existing_ilqr_v1',reference_states=states.tolist(),raw_control=controls[0].tolist(),action=action,solve_seconds=time.perf_counter()-began,solver_iterations=len(solutions),target_speed_mps=float(states[1,3]),raw_model_dt_s=.5,solver_dt_s=tracker.vehicle.dt,obstacle_constraints=False,mpc_search=False,reference_source='joint_repair_dense_physical_rollout',navigation_anchor_override=False)
