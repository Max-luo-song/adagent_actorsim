"""Observed RGB-D geometry, pose registered; no object/asset/scene truth."""
import numpy as np
from depth_surface_geometry import ground_plane, points_from_depth
from depth_contract import reliable_mask


class DepthObstacleMemory:
    def __init__(self, vehicle, resolution=.5, max_range=50.):
        self.vehicle=vehicle;self.resolution=float(resolution);self.max_range=float(max_range)
        self.cells={};self.scene=None;self.time=None

    def _free(self, points, obs, info):
        from runtime_configuration import configured_reliability
        configured_reliability()  # Install the shared NumPy-only geometry import path.
        from input_geometry import intrinsic_matrix
        free=np.zeros(len(points),bool)
        for name,depth in obs.get('depth',{}).items():
            camera=info['cam_params'][name];cp=(np.asarray(camera['l2c'])@np.c_[points,np.ones(len(points))].T).T[:,:3]
            k=intrinsic_matrix(camera['intrinsic'])[:3,:3];uv=cp@k.T;uv=uv[:,:2]/np.maximum(cp[:,2,None],1e-8)
            good=(cp[:,2]>.1)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0]);idx=np.flatnonzero(good)
            xy=uv[idx].astype(int);z=depth[xy[:,1],xy[:,0]]
            # Missing/occluded measurements are not evidence of free space.
            reliable=reliable_mask(obs,name)[xy[:,1],xy[:,0]]
            free[idx]|=reliable&np.isfinite(z)&(z>.1)&(z>cp[idx,2]+.5)
        return free

    def update(self, obs, info):
        from runtime_configuration import configured_reliability
        cfg=configured_reliability();now=float(info['timestamp']);scene=info['scene_token']
        rotation=np.asarray(info['explicit_l2g_r_mat']);translation=np.asarray(info['explicit_l2g_t'])
        if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.isfinite(now):raise ValueError('Invalid geometry map pose/time')
        if self.time is not None and (scene!=self.scene or now<=self.time):self.cells={}
        self.scene=scene;self.time=now;removed=0;plane=None
        if obs.get('depth') and info.get('depth_encoding')=='expected_camera_z_v1':
            plane=ground_plane(obs,info)
        if self.cells:
            keys=list(self.cells);world=np.array([self.cells[k]['point'] for k in keys]);local=(world-translation)@rotation
            free=self._free(local,obs,info) if not obs.get('depth') or info.get('depth_encoding')=='expected_camera_z_v1' else np.zeros(len(local),bool)
            for k,is_free,distance in zip(keys,free,np.linalg.norm(local[:,:2],axis=1)):
                cell=self.cells[k]
                if is_free:
                    if not cell['free_count']:cell['free_first']=now
                    cell['free_count']+=1
                else:cell['free_count']=0
                contradicted=(cell['free_count']>=cfg.confirmation_observations and now-cell['free_first']>=cfg.confirmation_seconds-1e-6)
                unconfirmed=(cell['count']<cfg.confirmation_observations and now-cell['seen']>cfg.max_gap_seconds)
                if contradicted or unconfirmed or distance>self.max_range:
                    del self.cells[k];removed+=1
        if plane is not None:
            points=points_from_depth(obs,info,stride=6)
            height=points[:,2]-np.c_[points[:,:2],np.ones(len(points))]@plane
            points=points[(height>.2)&(height<self.vehicle.body_height_m)&(np.linalg.norm(points[:,:2],axis=1)<self.max_range-.5)]
            world=points@rotation.T+translation;voxel=np.floor(world/self.resolution).astype(int)
            _,indices=np.unique(voxel,axis=0,return_index=True)
            for index in indices:
                key=tuple(voxel[index]);previous=self.cells.get(key)
                self.cells[key]=dict(point=world[index],count=(previous['count']+1 if previous else 1),seen=now,free_count=0,free_first=now)
        if self.cells:
            # Voxel centers plus the existing half-diagonal allowance enclose
            # observed XY cells even after rotation into the current model frame.
            centers=(np.array(list(self.cells),dtype=float)+.5)*self.resolution
            local=(centers-translation)@rotation
            # Only XY is rasterized: keep the exact height-bounded observations.
            xy=local[:,:2];xy=xy[np.linalg.norm(xy,axis=1)<self.max_range-.5]
            # Merge repeated vertical cells without moving horizontal centers.
            xy=np.unique(np.round(xy,7),axis=0)
        else:xy=np.empty((0,2))
        audit=dict(policy='observed_depth_geometry_with_free_space_clearing_v1',points=len(xy),cells=len(self.cells),removed_cells=removed,
            depth_available=bool(obs.get('depth')),ground_available=plane is not None,voxel_resolution_m=self.resolution,
            body_height_m=self.vehicle.body_height_m,voxel_half_diagonal_m=float(np.sqrt(3)*self.resolution/2),ground_truth_used=False,
            clearing='repeated visible free-space evidence; occlusion is not empty',
            forecast='current observed surfaces held hard; not a learned motion forecast')
        return xy,audit
