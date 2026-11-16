"""Keep the entire perceived detour in world space until evidence invalidates it.

This is a navigation objective, never an obstacle exemption. Every new MPC
solve retains all fresh hard constraints; no selected feasible solve causes
invalidation and a fresh route proposal on the next observation.
"""
import numpy as np
class CommittedRoute:
    def __init__(self):self.world=None;self.arc=None;self.progress=0.;self.scene=None;self.command=None
    def invalidate(self):self.world=None;self.arc=None;self.progress=0.
    def commit(self,optimization,info):
        if self.world is not None or optimization.get('selected') is None:return False
        guard=optimization.get('guard',{});route=guard.get('road_route_proposal',{});nav=guard.get('navigation_reference',{})
        if not route.get('path_xy'):return False
        offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
        path=np.asarray(route['path_xy'],dtype=float)+offset
        reference=np.asarray(nav.get('original_neural_reference',[]))
        if not len(reference):return False
        if np.linalg.norm(path[-1]-reference[-1])>1e-5:path=np.vstack([path,reference[-1]])
        keep=np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-5];path=path[keep]
        if len(path)<2:return False
        # Continue only the stable navigation tangent after the merge target.
        direction=reference[-1]-reference[-2];norm=np.linalg.norm(direction)
        if norm>1e-5:path=np.vstack([path,path[-1]+direction/norm*100.])
        r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t'])
        self.world=np.c_[path,np.zeros(len(path))]@r.T+t
        self.initial_offset_world=np.r_[offset,0.]@r.T
        self.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.world,axis=0),axis=1))]
        self.scene=info['scene_token'];self.command=info.get('command');self.progress=0.
        return True
    def reference(self,neural,info,vehicle,horizon=6.):
        if self.world is None:return None,dict(active=False)
        if info['scene_token']!=self.scene or info.get('command')!=self.command:self.invalidate();return None,dict(active=False,reason='navigation_context_changed')
        neural=np.asarray(neural,dtype=float).reshape(6,2)
        if np.linalg.norm(neural[-1])<vehicle.half_length_m:return None,dict(active=False,reason='neural_stop_preserved')
        r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t']);segments=np.diff(self.world,axis=0)
        u=np.clip(np.sum((t-self.world[:-1])*segments,axis=1)/np.sum(segments*segments,axis=1),0,1)
        nearest=self.world[:-1]+u[:,None]*segments
        idx=int(np.argmin(np.linalg.norm(nearest-t,axis=1)));self.progress=max(self.progress,float(self.arc[idx]+u[idx]*(self.arc[idx+1]-self.arc[idx])))
        times=np.arange(.5,horizon+1e-8,.5)
        extra=neural[-1]+np.maximum(times[6:]-3.,0.)[:,None]*(neural[-1]-neural[-2])/.5
        schedule=np.vstack([neural,extra])
        requested=np.cumsum(np.linalg.norm(np.diff(np.vstack([np.zeros(2),schedule]),axis=0),axis=1))
        world=np.column_stack([np.interp(self.progress+requested,self.arc,self.world[:,j]) for j in range(3)])
        return ((world-t)@r)[:,:2],dict(active=True,source='committed_world_detour',progress_m=self.progress,route_points=len(self.world),fresh_hard_constraints_required=True,ground_truth_used=False)

    def center_path(self,info):
        if self.world is None:return None
        r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t'])
        return ((self.world-self.initial_offset_world-t)@r)[:,:2]
