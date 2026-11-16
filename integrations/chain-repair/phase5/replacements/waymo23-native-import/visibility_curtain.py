"""Experimental conservative continuation of image-truncated observed surfaces.

An extension is UNKNOWN geometry, not a measured obstacle or a classified car.
Only sensor points, calibration, ground and the shared metric budget are inputs.
The caller keeps these constraints separate from measured obstacle points.
"""
import numpy as np
from sklearn.cluster import DBSCAN


def _projection(points, camera):
    from input_geometry import intrinsic_matrix
    cp=(np.asarray(camera['l2c'])@np.c_[points,np.ones(len(points))].T).T[:,:3]
    k=intrinsic_matrix(camera['intrinsic'])[:3,:3]
    uv=cp@k.T
    return cp,uv[:,:2]/np.maximum(cp[:,2,None],1e-8),k


def _visible_free(points,obs,info,budget):
    from depth_contract import geometry_depth,reliable_mask
    free=np.zeros(len(points),bool)
    for name in obs['depth']:
        depth=geometry_depth(obs,name);cp,uv,k=_projection(points,info['cam_params'][name])
        stride=4 if name in obs.get('surface_depth',{}) else 1
        xy=np.rint(uv/stride).astype(int)*stride
        inside=(cp[:,2]>.1)&(xy[:,0]>=0)&(xy[:,0]<depth.shape[1])&(xy[:,1]>=0)&(xy[:,1]<depth.shape[0])
        index=np.flatnonzero(inside);x,y=xy[index].T
        measured=depth[y,x];valid=reliable_mask(obs,name)[y,x]
        # A neighboring sparse ray is not exact evidence for this ray. Charge
        # its transverse projection error in addition to the metric budget.
        ray_error=np.linalg.norm((uv[index]-xy[index])/np.array([k[0,0],k[1,1]]),axis=1)*np.maximum(measured,cp[index,2])
        free[index]|=valid&(measured>cp[index,2]+budget+ray_error)
    return free


def _endpoint_crop(support, origin, tangent, edge, sign, plane, obs, info, budget):
    """Only a fitted-line endpoint at a horizontal image edge supports extension.

    A cropped unrelated cluster point or bottom/top clipping does not establish
    continuation of the fitted surface. Return camera/edge witnesses, not an
    unrestricted license to extend both ends of an infinite line.
    """
    from depth_contract import geometry_depth, reliable_mask
    span=(support[:,:2]-origin)@tangent
    endpoint=support[abs(span-edge)<=budget]
    witnesses=[]
    if len(endpoint)<3:return witnesses
    delta=tangent*sign*.25
    endxy=origin+edge*tangent
    pair=np.array([endxy,endxy+delta])
    ground=np.c_[pair,np.ones(2)]@plane
    for name in obs['depth']:
        depth=geometry_depth(obs,name)
        cp,uv,_=_projection(endpoint,info['cam_params'][name])
        xy=np.rint(uv/4).astype(int)*4
        inside=(cp[:,2]>.1)&(xy[:,0]>=0)&(xy[:,0]<depth.shape[1])&(xy[:,1]>=0)&(xy[:,1]<depth.shape[0])
        idx=np.flatnonzero(inside);x,y=xy[idx].T
        valid=reliable_mask(obs,name)[y,x]&(abs(depth[y,x]-cp[idx,2])<=budget)
        _,enduv,_=_projection(np.c_[pair,ground+np.median(endpoint[:,2]-np.c_[endpoint[:,:2],np.ones(len(endpoint))]@plane)],info['cam_params'][name])
        du=enduv[1,0]-enduv[0,0]
        if du<0 and np.sum(valid&(x<8))>=3:witnesses.append((name,'left'))
        if du>0 and np.sum(valid&(x>=depth.shape[1]-8))>=3:witnesses.append((name,'right'))
    return witnesses


def _supported_extension(samples,plane,witnesses,obs,info,vehicle):
    supported=np.zeros(len(samples),bool)
    ground=np.c_[samples,np.ones(len(samples))]@plane
    for name,side in witnesses:
        cp,uv,_=_projection(np.c_[samples,ground+vehicle.body_height_m/2],info['cam_params'][name])
        width=obs['surface_depth'][name].shape[1]
        outside=(uv[:,0]<8) if side=='left' else (uv[:,0]>=width-8)
        supported|=(cp[:,2]>.1)&outside
    # A witness stops supporting the continuation at the first re-entry or
    # near-plane crossing. Do not resume it beyond an unsupported interval.
    return np.logical_and.accumulate(supported)


def build(obs,info,vehicle,budget=.5,max_range=50.):
    from depth_surface_geometry import ground_plane,points_from_depth
    if not obs.get('surface_depth'):
        return np.empty((0,2)),dict(available=False,reason='no_explicit_front_surface')
    plane=ground_plane(obs,info)
    if plane is None:return np.empty((0,2)),dict(available=False,reason='no_supported_ground')
    points=points_from_depth(obs,info,stride=4)
    height=points[:,2]-np.c_[points[:,:2],np.ones(len(points))]@plane
    points=points[(height>.2)&(height<vehicle.body_height_m)]
    _,indices=np.unique(np.floor(points/.08).astype(int),axis=0,return_index=True)
    points=points[indices]
    if len(points)<5:return np.empty((0,2)),dict(available=True,components=[],reason='no_supported_body_cluster')
    labels=DBSCAN(eps=.24,min_samples=5).fit_predict(points)
    rng=np.random.default_rng(0);unknown=[];rows=[]
    for label in sorted(set(labels)-{-1}):
        cluster=points[labels==label]
        if len(cluster)<50:continue
        xy=cluster[:,:2];best=None
        for _ in range(100):
            a,b=xy[rng.choice(len(xy),2,replace=False)]
            direction=b-a;length=np.linalg.norm(direction)
            if length<.1:continue
            direction/=length;normal=np.array([-direction[1],direction[0]])
            distances=abs((xy-a)@normal);inside=distances<.12
            if best is None or inside.sum()>best.sum():best=inside
        if best is None or best.sum()<50 or best.mean()<.5:continue
        support=xy[best];origin=np.median(support,axis=0)
        _,vectors=np.linalg.eigh(np.cov((support-origin).T));tangent=vectors[:,-1]
        span=(support-origin)@tangent;lo,hi=np.quantile(span,[.005,.995])
        if hi-lo<2*budget:continue
        boundaries=[];added=[];endpoint_witnesses=[]
        for edge,sign in [(lo,-1),(hi,1)]:
            witnesses=_endpoint_crop(cluster[best],origin,tangent,edge,sign,plane,obs,info,budget)
            endpoint_witnesses.append(witnesses)
            if not witnesses:
                boundaries.append(False);continue
            ts=edge+sign*np.arange(.25,2*max_range,.25)
            samples=origin+ts[:,None]*tangent
            good=np.linalg.norm(samples,axis=1)<max_range-.5
            samples=samples[good]
            samples=samples[_supported_extension(samples,plane,witnesses,obs,info,vehicle)]
            if not len(samples):boundaries.append(False);continue
            # All three body elevations need free-space evidence; a road or
            # roof return alone cannot establish clearance for the full body.
            free=np.ones(len(samples),bool)
            for height in (.2,vehicle.body_height_m/2,vehicle.body_height_m):
                ground=np.c_[samples,np.ones(len(samples))]@plane
                free&=_visible_free(np.c_[samples,ground+height],obs,info,budget)
            consecutive=0;bounded=False
            for i in range(len(samples)):
                consecutive=consecutive+1 if free[i] else 0
                if consecutive>=3:bounded=True;break
                if not free[i]:added.append(samples[i])
            boundaries.append(bounded)
        if added:unknown.extend(added)
        rows.append(dict(cluster_points=len(cluster),line_inliers=int(best.sum()),endpoint_crop_witnesses=endpoint_witnesses,
            origin=origin.tolist(),tangent=tangent.tolist(),observed_span_m=[float(lo),float(hi)],
            unknown_points=len(added),free_bounded_ends=boundaries,continuation_xy=np.asarray(added).reshape(-1,2).tolist()))
    return np.asarray(unknown).reshape(-1,2),dict(available=True,components=rows,
        scope='unknown continuation of a truncated surface; not a measured obstacle',
        ground_truth_used=False,network_threshold_modified=False)


class VisibilityCurtainMemory:
    """World-registered unknown continuations survive blindness until disproved."""
    def __init__(self,vehicle,resolution=.5,max_range=50.):
        self.vehicle=vehicle;self.resolution=float(resolution);self.max_range=float(max_range)
        self.cells={};self.scene=None;self.time=None;self.hypotheses=[]

    def update(self,obs,info):
        from runtime_configuration import configured_reliability
        config=configured_reliability();now=float(info['timestamp']);scene=info['scene_token']
        rotation=np.asarray(info['explicit_l2g_r_mat']);translation=np.asarray(info['explicit_l2g_t'])
        if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.isfinite(now):
            raise ValueError('Invalid visibility-memory pose')
        if self.time is not None and (scene!=self.scene or now<=self.time):
            self.cells={};self.hypotheses=[]
        self.scene=scene;self.time=now;removed=0
        if self.cells:
            keys=list(self.cells);world=np.array([self.cells[key]['point'] for key in keys])
            local=(world-translation)@rotation
            free=np.zeros(len(local),bool)
            if obs.get('surface_depth'):
                free[:]=True
                for height in (.2,self.vehicle.body_height_m/2,self.vehicle.body_height_m):
                    sample=local.copy();sample[:,2]+=height-self.vehicle.body_height_m/2
                    free&=_visible_free(sample,obs,info,config.obstacle_margin_m)
            for key,is_free,distance in zip(keys,free,np.linalg.norm(local[:,:2],axis=1)):
                cell=self.cells[key]
                if is_free:
                    if cell['free_count']==0:cell['free_first']=now
                    cell['free_count']+=1
                else:cell['free_count']=0
                contradicted=(cell['free_count']>=config.confirmation_observations and now-cell['free_first']>=config.confirmation_seconds-1e-6)
                if contradicted or distance>self.max_range:
                    del self.cells[key];removed+=1
        current=np.empty((0,2));audit=dict(available=False,reason='missing_supported_depth')
        if obs.get('depth') and info.get('depth_encoding')=='expected_camera_z_v1':
            current,audit=build(obs,info,self.vehicle,config.obstacle_margin_m,self.max_range)
            accepted=[];next_hypotheses=[];used=set()
            for component in audit.get('components',[]):
                raw=np.asarray(component.pop('continuation_xy',[]),dtype=float).reshape(-1,2)
                if not len(raw):continue
                origin_local=np.r_[component['origin'],0.]
                origin=origin_local@rotation.T+translation
                direction=rotation@np.r_[component['tangent'],0.]
                direction/=np.linalg.norm(direction)
                span=float(np.ptp(component['observed_span_m']))
                best=None
                for j,old in enumerate(self.hypotheses):
                    if j in used or now-old['last']>config.max_gap_seconds:continue
                    cosine=float(abs(direction@old['direction']))
                    angular=max(span,old['span'])*np.sqrt(max(0.,1-cosine*cosine))
                    delta=origin-old['origin'];normal=delta-direction*(delta@direction)
                    along=abs(float(delta@direction))
                    if angular<=.12 and np.linalg.norm(normal)<=config.obstacle_margin_m and along<=max(span,old['span'])+config.obstacle_margin_m:
                        best=j;break
                if best is None:
                    track=dict(origin=origin,direction=direction,span=span,first=now,last=now,count=1)
                else:
                    used.add(best);track=dict(self.hypotheses[best]);track.update(origin=origin,direction=direction,span=span,last=now,count=track['count']+1)
                next_hypotheses.append(track)
                confirmed=track['count']>=config.confirmation_observations and now-track['first']>=config.confirmation_seconds-1e-6
                component['continuation_confirmed']=bool(confirmed)
                component['confirmation_observations']=track['count']
                if confirmed:accepted.extend(raw)
            self.hypotheses=next_hypotheses
            # Tests and callers may supply legacy build results without component
            # geometry; they must not be silently promoted to hard constraints.
            current=np.asarray(accepted).reshape(-1,2)
            audit['confirmed_current_unknown_points']=len(current)
            if len(current):
                # Anchor the vertical probe to observed local ground; never use
                # simulator object positions or unobserved shape dimensions.
                from depth_surface_geometry import ground_plane
                plane=ground_plane(obs,info)
                local=np.c_[current,np.c_[current,np.ones(len(current))]@plane+self.vehicle.body_height_m/2]
                world=local@rotation.T+translation
                for point,key in zip(world,np.floor(world/self.resolution).astype(int)):
                    self.cells[tuple(key)]=dict(point=point,free_count=0,free_first=now)
        if self.cells:
            centers=(np.array(list(self.cells),dtype=float)+.5)*self.resolution
            local=(centers-translation)@rotation
            xy=np.unique(np.round(local[:,:2],7),axis=0)
        else:xy=np.empty((0,2))
        radius=float(np.sqrt(3)*self.resolution/2+.12+.125)
        return xy,dict(policy='endpoint_supported_unknown_visibility_v2',cells=len(self.cells),
            points=len(xy),removed_cells=removed,current_unknown_points=len(current),current=audit,
            point_radius_m=radius,ground_truth_used=False,
            clearing='three repeated full-body visible-free observations; blindness is not free',
            uncertainty='unknown region; finite sensor range; physical shape not certified',
            generation='horizontal endpoint crop and repeated world-line consistency required')
