"""Pure external reference processing; no simulator, I/O, scene IDs or future truth."""
import numpy as np

PARAMETERS=dict(score_threshold=.4,vehicle_labels=[0,1,2,3,4],confirmation_observations=3,
    association_distance_m=2.5,memory_max_age_s=.6,lateral_margin_m=.3,
    standstill_gap_m=7.,range_uncertainty_m=.5,reaction_s=.3,
    planning_dt_s=.5,plan_horizon_s=3.,near_limit_m=45.)

def apply(plan,perception,ego,state):
    plan=np.asarray(plan,dtype=float);mount=np.asarray(ego['virtual_to_ego'],dtype=float)
    rotation=np.asarray(ego['lidar_to_world_rotation'],dtype=float);translation=np.asarray(ego['lidar_to_world_translation'],dtype=float)
    dt=float(ego['dt_s']);speed=float(ego['speed_mps']);half_length=float(ego['half_length_m']);half_width=float(ego['half_width_m'])
    braking=max(1e-6,-float(ego['min_acc_mps2']));p=PARAMETERS
    if plan.shape!=(6,2) or not np.isfinite(plan).all() or not 0<dt<=.5 or speed<0:raise ValueError('Invalid patch reference/ego contract')
    state=dict(state or {});old=[dict(t,age_s=t['age_s']+dt) for t in state.get('tracks',[]) if t['age_s']+dt<=p['memory_max_age_s']+1e-9]
    boxes=np.asarray(perception.get('boxes',np.empty((0,9))),dtype=float)
    scores=np.asarray(perception.get('scores',[]),dtype=float);labels=np.asarray(perception.get('labels',[]))
    if boxes.ndim!=2 or boxes.shape[1]<7 or len(scores)!=len(boxes) or len(labels)!=len(boxes):raise ValueError('Unaligned current neural detections')
    current=[];used=set()
    for i,box in enumerate(boxes):
        if not np.isfinite(box).all() or not np.isfinite(scores[i]) or scores[i]<p['score_threshold'] or int(labels[i]) not in p['vehicle_labels']:continue
        width,length,height=box[3:6]
        if not(.8<=width<=4. and 2.<=length<=20. and .5<=height<=5.):continue
        center=box[:3];world=center@rotation.T+translation
        distances=[np.linalg.norm(world-np.asarray(t['world_center'])) if j not in used else np.inf for j,t in enumerate(old)]
        j=int(np.argmin(distances)) if distances else -1
        matched=j>=0 and distances[j]<=p['association_distance_m']
        hits=old[j]['hits']+1 if matched else 1
        if matched:used.add(j)
        c,s=abs(np.cos(box[6])),abs(np.sin(box[6]))
        hx,hy=(c*width+s*length)/2,(s*width+c*length)/2
        offsets=np.array([[-hx,-hy,0],[hx,-hy,0],[hx,hy,0],[-hx,hy,0]])
        # Rigid world axes/camera calibration remain runtime inputs, not hidden truth.
        current.append(dict(world_center=world.tolist(),world_corners=(world+offsets@rotation.T).tolist(),
            hits=hits,age_s=0.,score=float(scores[i])))
    tracks=current+[t for j,t in enumerate(old) if j not in used]
    eligible=[]
    for t in tracks:
        if t['hits']<p['confirmation_observations']:continue
        local=(np.asarray(t['world_center'])-translation)@rotation
        corners=(np.asarray(t['world_corners'])-translation)@rotation
        half_x=float(np.max(np.abs(corners[:,0]-local[0])));half_y=float(np.max(np.abs(corners[:,1]-local[1])))
        right=float(local[0]+mount[0,3]);forward=float(local[1]+mount[1,3])
        if forward<=0 or forward>p['near_limit_m'] or abs(right)>half_width+half_x+p['lateral_margin_m']:continue
        # Worst-case stationary front object; no trust in highly uncertain box velocity.
        safe_travel=forward-half_y-half_length-p['standstill_gap_m']-p['range_uncertainty_m']-p['reaction_s']*speed-t['age_s']
        eligible.append((safe_travel,t,forward,right))
    audit=dict(source='bounded_front_vehicle_braking_reference_v1',triggered=False,
        inputs='current/past retained neural boxes/scores/labels; actual ego pose, speed, geometry, mount, dt',
        no_future_actor_or_scene_time_inputs=True,memory_track_count=len(tracks),parameters=p,
        rejections=[],candidate_count=1,selected='unchanged_raw_uniad',reference_changed=False)
    result=plan.copy()
    if eligible:
        distance,track,forward,right=min(eligible,key=lambda v:v[0]);original_forward=plan[:,1]+mount[1,3]
        danger=float(np.max(original_forward))>distance or speed*speed/(2*braking)>distance
        if danger:
            required=speed*speed/(2*max(distance,.01));deceleration=min(braking,required)
            if speed<1e-4:deceleration=0.
            times=np.arange(1,7)*p['planning_dt_s']
            until=np.minimum(times,speed/max(deceleration,1e-12))
            reference=speed*until-.5*deceleration*until*until
            # Do not invent instant stopping where actuator-limited stopping is impossible.
            reference=np.minimum(original_forward,reference)
            result[:,1]=reference-mount[1,3]
            # Retime the already map-processed path. Repeated stop nodes must have
            # identical XY; otherwise numerical lateral motion invents stop yaw.
            if np.any(np.diff(np.r_[0.,original_forward])<0):raise ValueError('Front braking patch requires a forward-monotone reference')
            result[:,0]=np.interp(reference,np.r_[0.,original_forward],np.r_[-mount[0,3],plan[:,0]])
            changed=not np.array_equal(result,plan)
            audit.update(triggered=True,reference_changed=changed,selected='front_vehicle_braking_reference',
                front_center_forward_m=forward,front_center_right_m=right,
                safe_travel_m=distance,required_deceleration_mps2=required,reference_deceleration_mps2=deceleration,
                stopping_feasible_with_declared_brake=required<=braking,
                detection_age_s=track['age_s'],confirmation_hits=track['hits'],conservative_stationary_assumption=True)
    return dict(reference=result,state=dict(tracks=tracks),audit=audit)
