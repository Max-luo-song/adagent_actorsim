"""Opt-in, offline pedestrian motion conditioning. No weights are trained.

C1 Hermite root paths, quaternion-space filtering and distance-based source
animation sampling. All interpolation is in 3D before Gaussian rasterization.
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline,PchipInterpolator
from scipy.spatial.transform import Rotation,RotationSpline


def spline(times,values):
    values=np.asarray(values);slopes=np.diff(values,axis=0)/np.diff(times).reshape((-1,)+(1,)*(values.ndim-1))
    tangent=np.empty_like(values);tangent[0]=slopes[0];tangent[-1]=slopes[-1]
    if len(times)>2:
        # Weighted neighbouring secants: bounded, shared derivatives at knots.
        dt=np.diff(times);shape=(-1,)+(1,)*(values.ndim-1)
        tangent[1:-1]=(slopes[:-1]*dt[1:].reshape(shape)+slopes[1:]*dt[:-1].reshape(shape))/(dt[:-1]+dt[1:]).reshape(shape)
    return CubicHermiteSpline(times,values,tangent)


def normalize(q):
    norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or np.any(norm<1e-8):raise ValueError('source animation contains invalid quaternion')
    return q/norm


def qmix(a,b,fraction):
    a=normalize(a);b=normalize(b);dot=(a*b).sum(-1,keepdims=True);b=np.where(dot<0,-b,b);dot=np.abs(dot).clip(0,1)
    theta=np.arccos(dot);sin=np.sin(theta);den=np.maximum(sin,1e-8)
    result=np.sin((1-fraction)*theta)/den*a+np.sin(fraction*theta)/den*b
    return normalize(np.where(sin<1e-6,(1-fraction)*a+fraction*b,result))


def filter_quaternions(values,valid,weights=(.2,.6,.2)):
    values=normalize(values);result=values.copy();radius=len(weights)//2
    for frame in np.flatnonzero(valid):
        indices=np.arange(max(0,frame-radius),min(len(values),frame+radius+1));indices=indices[valid[indices]]
        # Do not smooth across an invalid source-animation gap.
        indices=np.array([i for i in indices if valid[min(i,frame):max(i,frame)+1].all()])
        w=np.array([weights[i-frame+radius] for i in indices]);w/=w.sum()
        q=values[indices];q=np.where((q*values[frame]).sum(-1,keepdims=True)<0,-q,q)
        result[frame]=normalize(np.sum(q*w.reshape((-1,)+(1,)*(q.ndim-1)),axis=0))
    return result


def prepare_motion(plan,spec,trans,valid,root_quats,joints,test_frames):
    from actor_trajectories import source_center
    trans=np.asarray(trans);valid=np.asarray(valid,dtype=bool)
    roots=np.asarray(root_quats).reshape(len(valid),4).copy();bones=np.asarray(joints).copy()
    roots[~valid]=np.array([1.,0.,0.,0.]);bones[~valid]=np.array([1.,0.,0.,0.])
    roots=normalize(roots);bones=normalize(bones)
    roots=roots.copy();bones=bones.copy()
    for frame in test_frames:
        if 1<frame<len(valid)-1 and valid[frame-1:frame+2].all():
            roots[frame]=qmix(root_quats[frame-1].reshape(4),root_quats[frame+1].reshape(4),.5)
            bones[frame]=qmix(joints[frame-1],joints[frame+1],.5)
    raw_bones=bones.copy()
    roots=filter_quaternions(roots,valid,(1/9,2/9,3/9,2/9,1/9));bones=filter_quaternions(bones,valid)
    start=plan['first_frame'];end=start
    while end+1<len(valid) and valid[end+1]:end+=1
    if end<=start:raise ValueError('walking mode requires continuous source animation after start')
    centers=np.array([source_center(trans,valid,f,test_frames) for f in range(start,end+1)])
    steps=np.diff(centers,axis=0);steps-=np.outer(steps@plan['up'],plan['up'])
    distance=np.r_[0,np.cumsum(np.linalg.norm(steps,axis=1))]
    times=np.linspace(plan['times'][0],plan['times'][-1],max(2,int(round((plan['times'][-1]-plan['times'][0])*120))+1))
    positions=plan['position_spline'](times);steps=np.diff(positions,axis=0);steps-=np.outer(steps@plan['up'],plan['up'])
    target_distance=np.r_[0,np.cumsum(np.linalg.norm(steps,axis=1))]
    if distance[-1]+1e-6<target_distance[-1]:raise ValueError('insufficient continuous source walk distance for gait retiming')
    if target_distance[-1]>1e-3 and distance[-1]<1e-3:raise ValueError('stationary source body cannot supply a walking gait')
    keep=np.r_[True,np.diff(distance)>1e-7]
    source_frames=np.arange(start,end+1)[keep];source_distance=distance[keep]
    if len(source_distance)<2:raise ValueError('source walk needs measurable distance')
    phase=PchipInterpolator(source_distance,source_frames)(target_distance)
    phase_curve=PchipInterpolator(times,phase)
    extrema=phase_curve.derivative(2).roots(extrapolate=False)
    extrema=extrema[np.isfinite(extrema)&(extrema>=times[0])&(extrema<=times[-1])]
    rates=phase_curve(np.r_[times,extrema],1)/spec.fps
    if np.max(rates)>2.5 or np.min(rates)<.25:raise ValueError('gait retiming outside diagnostic 0.25..2.5x limits')
    direction=centers[min(5,len(centers)-1)]-centers[0];direction-=plan['up']*(direction@plan['up'])
    if np.linalg.norm(direction)<.02:raise ValueError('source walking heading is ambiguous')
    direction/=np.linalg.norm(direction)
    joint_curves=[RotationSpline(np.arange(start,end+1),Rotation.from_quat(bones[start:end+1,j][:,[1,2,3,0]])) for j in range(bones.shape[1])]
    interior=plan['times'][1:-1];epsilon=1e-6
    before=max((float(np.linalg.norm((plan['positions'][i+1]-plan['positions'][i])/(plan['times'][i+1]-plan['times'][i])-(plan['positions'][i]-plan['positions'][i-1])/(plan['times'][i]-plan['times'][i-1]))) for i in range(1,len(plan['times'])-1)),default=0.)
    after=max((float(np.linalg.norm(plan['position_spline'](t-epsilon,1)-plan['position_spline'](t+epsilon,1))) for t in interior),default=0.)
    def roughness(quaternions):
        q=quaternions[start:int(np.ceil(phase[-1]))+1].copy()
        for f in range(1,len(q)):
            q[f]=np.where((q[f]*q[f-1]).sum(-1,keepdims=True)<0,-q[f],q[f])
        return float(np.sqrt(np.mean(np.diff(q,n=2,axis=0)**2))) if len(q)>2 else 0.
    return dict(joint_roughness_before=roughness(raw_bones),joint_roughness_after=roughness(bones),roots=roots,bones=bones,raw_bones=raw_bones,joint_curves=joint_curves,phase_curve=phase_curve,source_direction=direction,reference_root=roots[start],
        velocity_jump_before_mps=before,velocity_jump_after_mps=after,
        times=times,phase=phase,min_playback_rate=float(rates.min()),max_playback_rate=float(rates.max()),
        last_animation_frame=float(phase[-1]),target_distance_m=float(target_distance[-1]))


def motion_sample(entry,timestamp):
    plan=entry['plan'];motion=entry['motion'];t=np.clip(timestamp,plan['times'][0],plan['times'][-1])
    phase=float(motion['phase_curve'](t))
    joints=np.array([curve(phase).as_quat()[[3,0,1,2]] for curve in motion['joint_curves']])
    if plan['heading_mode']=='path_tangent':
        direction=plan['position_spline'](t,1);up=plan['up'];direction-=up*(direction@up)
        if np.linalg.norm(direction)<1e-6:raise ValueError('cannot orient walker at zero path tangent')
        direction/=np.linalg.norm(direction);source=motion['source_direction']
        yaw=float(np.arctan2(up@np.cross(source,direction),source@direction))
    else:yaw=float(plan['yaw_spline'](t))
    return dict(joints=joints,root_quat=motion['reference_root'],yaw=yaw,animation_frame=phase)
