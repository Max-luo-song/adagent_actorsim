"""Keep the initial neural navigation curve in world space across local turns.

Bounded diagnostic route prior, not a ground-truth lane centerline. Reset on
scene/command/time discontinuity; never override an explicit neural stop.
"""
import numpy as np
class NavigationAnchor:
    def __init__(self,stop_distance_m=2.):self.stop_distance_m=float(stop_distance_m);self.points=None;self.arc=None;self.scene=None;self.command=None;self.time=None;self.progress=0.
    def update(self,plan,info):
        plan=np.asarray(plan,dtype=float).reshape(6,2);r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t'])
        if r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(plan).all():raise ValueError('Invalid navigation pose/reference')
        scene=info['scene_token'];command=info.get('command');now=float(info['timestamp'])
        reset=self.points is None or scene!=self.scene or command!=self.command or now<=self.time
        if reset:
            local=np.vstack([np.zeros(2),plan]);length=np.linalg.norm(np.diff(local,axis=0),axis=1)
            good=np.r_[True,length>1e-5];local=local[good]
            if len(local)<2:return plan.copy(),dict(source='neural_stop_no_anchor',ground_truth_used=False)
            direction=local[-1]-local[-2];direction/=np.linalg.norm(direction)
            # Unbounded route direction represented by a long finite ray; it
            # is only an objective. Full hard geometry is checked separately.
            local=np.vstack([local,local[-1]+direction*100.])
            self.points=np.c_[local,np.zeros(len(local))]@r.T+t;self.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))];self.progress=0.
        self.scene=scene;self.command=command;self.time=now
        segments=np.diff(self.points,axis=0);u=np.clip(np.sum((t-self.points[:-1])*segments,axis=1)/np.sum(segments*segments,axis=1),0,1)
        closest=self.points[:-1]+u[:,None]*segments;idx=int(np.argmin(np.linalg.norm(closest-t,axis=1)))
        self.progress=max(self.progress,float(self.arc[idx]+u[idx]*(self.arc[idx+1]-self.arc[idx])))
        requested=np.cumsum(np.linalg.norm(np.diff(np.vstack([np.zeros(2),plan]),axis=0),axis=1))
        distances=self.progress+requested
        world=np.column_stack([np.interp(distances,self.arc,self.points[:,j]) for j in range(3)])
        anchored=((world-t)@r)[:,:2]
        neural_stop=np.linalg.norm(plan[-1])<self.stop_distance_m
        return (plan.copy() if neural_stop else anchored),dict(source='initial_neural_world_route',reset=bool(reset),command=command,progress_m=self.progress,neural_stop_preserved=bool(neural_stop),ground_truth_used=False,physical_lane_center_certified=False)
    def errors(self,info):
        if self.points is None:return None
        t=np.asarray(info['explicit_l2g_t'])[:2];r=np.asarray(info['explicit_l2g_r_mat']);segments=np.diff(self.points[:,:2],axis=0)
        u=np.clip(np.sum((t-self.points[:-1,:2])*segments,axis=1)/np.sum(segments*segments,axis=1),0,1)
        closest=self.points[:-1,:2]+u[:,None]*segments;distance=np.linalg.norm(closest-t,axis=1);idx=int(np.argmin(distance))
        forward=r[:2,1];forward=forward/np.linalg.norm(forward);tangent=segments[idx]/np.linalg.norm(segments[idx])
        return dict(cross_track_m=float(distance[idx]),heading_error_rad=float(np.arccos(np.clip(forward@tangent,-1,1))),ground_truth_used=False)

def rejoin_completion(completion,observations,vehicle,confirmation_s=.5):
    result=dict(completion);now=observations[-1]['timestamp'];tail=[x for x in observations if x['timestamp']>=now-confirmation_s-1e-8]
    rejoined=bool(tail and now-tail[0]['timestamp']>=confirmation_s-1e-8 and all(x.get('navigation_rejoin') is not None and x['navigation_rejoin']['cross_track_m']<=vehicle.half_width_m and x['navigation_rejoin']['heading_error_rad']<=np.deg2rad(5.) and x['closed_speed_mps']>=.5 for x in tail))
    result.update(original_pass_success=bool(completion['success']),route_rejoined=rejoined,rejoin_lateral_tolerance_m=vehicle.half_width_m,rejoin_heading_tolerance_rad=float(np.deg2rad(5.)),rejoin_confirmation_s=confirmation_s,success=bool(completion['success'] and rejoined))
    if completion['success'] and not rejoined:result['reason']='passed_but_route_rejoin_unconfirmed'
    return result
