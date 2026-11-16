"""Fixed existing HUGSIM iLQR core; one control per fresh UniAD observation."""
import ast,dataclasses,math,time
import numpy as np
from common import ROOT,P26,configure,sha

PARAMETERS=dict(reference_dt_s=.5,execution_dt_s=.1,horizon_s=3.,max_ilqr_iterations=40,
    max_solve_time=None,state_cost=[5.,5.,10.,1.,.1],input_cost=[.2,.5],
    source_factory='phase26/bridge/agent_json/ego_execution.make_solver',
    selection='existing dt-matched light iLQR passes analytic direction/tracking gates; no Pure Pursuit/PID introduced')


def load_helpers():
    configure()
    from scene_api import validate_action
    from sim.ilqr.lqr_solver import ILQRSolver,ILQRSolverParameters,ILQRWarmStartParameters
    path=P26/'bridge/agent_json/ego_execution.py';tree=ast.parse(path.read_text())
    names={'make_solver','clip_action','motion_state','motion_pose'}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in body}!=names:raise ValueError('Existing light-tracker helpers changed')
    scope=dict(np=np,math=math,validate_action=validate_action)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)
    return scope


def reference(raw,virtual_to_ego,dt=.1):
    raw=np.asarray(raw,dtype=np.float64);mount=np.asarray(virtual_to_ego,dtype=np.float64)
    if raw.shape!=(6,2) or not np.isfinite(raw).all() or mount.shape!=(4,4) or not np.allclose(mount[:3,:3],np.eye(3)):
        raise ValueError('Expected six finite right/forward nodes and an unrotated declared virtual mount')
    if not np.isclose(dt,.1):raise ValueError('Frozen tracker clock is 0.1s')
    # Model predictions are in virtual-LiDAR coordinates. Vehicle center is -mount there.
    positions=np.vstack([np.zeros(2),(raw+mount[:2,3])[:,[1,0]]])
    knots=np.arange(7)*.5;times=np.arange(31)*dt
    xy=np.stack([np.interp(times,knots,positions[:,i]) for i in [0,1]],axis=1)
    # Same sampling origin/time for position, velocity, heading and steering feedforward.
    velocity=np.gradient(xy,dt,axis=0);speed=np.linalg.norm(velocity,axis=1)
    heading=np.zeros(len(xy));last=0.
    for i,v in enumerate(velocity):
        if np.linalg.norm(v)>1e-7:last=float(np.arctan2(v[1],v[0]))
        heading[i]=last
    heading=np.unwrap(heading);yawrate=np.gradient(heading,dt)
    states=np.c_[xy,heading,speed,np.zeros(len(xy))]
    return states,yawrate


class Tracker:
    def __init__(self,vehicle):
        self.vehicle=vehicle;self.helpers=load_helpers();self.solver=self.helpers['make_solver'](vehicle)
        # Remove wall-time-dependent stopping; retain the existing 40-iteration bound.
        self.solver._solver_params=dataclasses.replace(self.solver._solver_params,max_solve_time=None)
    def control(self,raw,info,motion):
        states,yawrate=reference(raw,info['model_lidar_frame']['virtual_to_ego'],self.vehicle.dt)
        states[:,4]=np.arctan2(self.vehicle.wheelbase_m*yawrate,np.maximum(states[:,3],1e-8))
        # Current body frame resets naturally at every actual observation, not by changing Motion.
        initial=np.array([0.,0.,0.,motion.speed,motion.steer]);began=time.perf_counter()
        solutions=self.solver.solve(initial,states)
        if not solutions:raise RuntimeError('Light tracker returned no solution; no action executed')
        controls=np.asarray(solutions[-1].input_trajectory)
        action=self.helpers['clip_action'](*controls[0],motion,self.vehicle)
        return action,dict(controller='fixed_existing_ilqr_v1',reference_states=states.tolist(),
            raw_control=controls[0].tolist(),action=action,solve_seconds=time.perf_counter()-began,
            solver_iterations=len(solutions),target_speed_mps=float(states[1,3]),
            raw_model_dt_s=.5,solver_dt_s=self.vehicle.dt,limits_clipped=not np.allclose(controls[0],list(action.values()),rtol=0,atol=1e-10),
            obstacle_constraints=False,mpc_search=False,recovery_override=False,navigation_anchor_override=False,
            reference_source='current_uniad_after_declared_external_processing')


def native_legacy_reference(raw):
    """Execute the current stock conversion body with a spy, without changing its source."""
    path=ROOT/'code/HUGSIM/sim/utils/sim_utils.py';tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='traj2control')
    captured={}
    def spy(states,initial):captured.update(states=states,initial=initial);return 0.,0.
    scope=dict(np=np,plan2control=spy)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    scope['traj2control'](raw,dict(ego_velo=5.,ego_steer=0.));return captured['states']


def synthetic_audit():
    configure()
    from runtime_configuration import load_vehicle_config
    from live_env import Motion
    vehicle=load_vehicle_config();mount=np.eye(4);mount[:2,3]=[0.,.985793]
    records={};allpassed=True
    for kind in ['straight','right_turn','left_turn','brake_stop']:
        tracker=Tracker(vehicle);motion=Motion(vehicle.dt,vehicle.wheelbase_m,5.);errors=[];steers=[];latency=[];limits=[]
        def analytic(t):
            t=np.asarray(t)
            if kind=='brake_stop':return np.c_[5*np.minimum(t,5)-.5*np.minimum(t,5)**2,np.zeros_like(t)]
            if kind=='straight':return np.c_[5*t,np.zeros_like(t)]
            sign=1 if kind=='right_turn' else -1;radius=30.;angle=5*t/radius
            return np.c_[radius*np.sin(angle),sign*radius*(1-np.cos(angle))]
        for index in range(100):
            now=index*vehicle.dt;future=analytic(now+np.arange(1,7)*.5)
            delta=future-[motion.z,motion.x];c,s=np.cos(motion.yaw),np.sin(motion.yaw)
            body=delta@np.array([[c,-s],[s,c]])
            raw=body[:,[1,0]]-mount[:2,3]
            action,audit=tracker.control(raw,dict(model_lidar_frame=dict(virtual_to_ego=mount)),motion)
            before=np.array([motion.x,motion.z,motion.yaw,motion.speed,motion.steer])
            motion.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
            expected=analytic([now+vehicle.dt])[0];errors.append(float(np.linalg.norm([motion.z-expected[0],motion.x-expected[1]])))
            steers.append(motion.steer);latency.append(audit['solve_seconds']);limits.append(audit['limits_clipped'])
        passed=max(errors)<1.0 and (kind!='brake_stop' or motion.speed<.2)
        if kind=='right_turn':passed &= motion.x>0 and motion.yaw>0
        if kind=='left_turn':passed &= motion.x<0 and motion.yaw<0
        records[kind]=dict(passed=bool(passed),max_tracking_error_m=max(errors),final_tracking_error_m=errors[-1],
            final_speed_mps=motion.speed,mean_solve_seconds=float(np.mean(latency)),max_solve_seconds=max(latency),
            final_right_positive_yaw_rad=motion.yaw,limit_clips=sum(limits))
        allpassed &= bool(passed)
    straight=np.c_[np.zeros(6),np.arange(1,7)*2.5]
    old=native_legacy_reference(straight)
    return dict(passed=allpassed,cases=records,parameters=PARAMETERS,
        stock_traj2control_straight_first_heading_rad=float(old[1,2]),expected_heading_rad=0.,
        stock_conversion_reused=False,reason='stock heading atan2(forward,right) disagrees with its forward/right state layout; existing corrected helpers retained',
        original_fast_control_harness_reused=False,removed=['heavy obstacle MPC','multistart avoidance search','guard/recovery state machine','NavigationAnchor','CommittedRoute','reference-route control override'],
        retained=['unchanged UniAD weights/code','original iLQR core','vehicle limits','native Motion integration','environment collision kernels','termination and native scorer'],
        gpu_closed_loop_executed=False)
