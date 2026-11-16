"""Recover a newly inconsistent UNKNOWN margin without crossing its cells.

All original raster, depth, proposal and road constraints remain active. Only
an existing overlap with the uncertainty margin of inferred cells can receive
a monotone, time-bounded restoration envelope. Unknown cells themselves remain
hard. Neither missing observations nor elapsed time certifies free space.
"""
import copy,math
import numpy as np
from bounded_visibility_geometry import TemporalOccupancy as OriginalTemporalOccupancy,polygon_clearance,projected_cell
from depth_sensor_map import DepthObstacleMemory as OriginalDepthObstacleMemory
RESTORE_SECONDS=3.0 # frozen reference horizon; deadline persists across replans

def unknown_clearance(states,centers,rotation,vehicle,extra):
    expanded=np.c_[np.asarray(states),np.full(len(states),vehicle.half_width_m),np.full(len(states),vehicle.half_length_m)]
    return polygon_clearance(expanded,centers,projected_cell(rotation),extra)

class RecoveryDepthMemory(OriginalDepthObstacleMemory):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.recovery_episode=None
    def update(self,obs,info):
        points,audit=super().update(obs,info)
        geometry=audit['unknown_visibility_geometry'];now=float(info['timestamp']);token=info['scene_token']
        offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
        initial=np.array([*-offset,0.,info['ego_velo'],info['ego_steer']])
        centers=np.asarray(audit['unknown_visibility_points_xy']).reshape(-1,2)
        extra=geometry['point_radius_m']-np.sqrt(3)*.25+self.vehicle.clearance_margin_m+1e-7
        gap=float(unknown_clearance(initial[None],centers,info['explicit_l2g_r_mat'],self.vehicle,extra)[0])
        episode=self.recovery_episode
        if episode is not None and (episode['scene_token']!=token or now<=episode['timestamp']):episode=None
        if gap>=-1e-5:episode=None
        elif episode is None:
            episode=dict(scene_token=token,start=now,deadline=now+RESTORE_SECONDS,initial_gap_m=gap,timestamp=now)
        else:episode=dict(episode,timestamp=now)
        self.recovery_episode=episode
        geometry['initial_unknown_margin_recovery']=dict(episode=copy.deepcopy(episode),current_gap_m=gap if np.isfinite(gap) else None,
            policy='monotone_unknown_margin_restoration_v1',unknown_cells_removed=False,free_space_asserted=False)
        return points,audit

class TemporalOccupancy(OriginalTemporalOccupancy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        known_kwargs=dict(kwargs,unknown_points=None,unknown_cell_rotation=None)
        self.known=OriginalTemporalOccupancy(*args,**known_kwargs)
        self.recovery=None;self.recovery_rejection=None
    def configure_recovery(self,initial,info,reliability,proposal_gap):
        meta=(reliability or {}).get('unknown_visibility_geometry',{}).get('initial_unknown_margin_recovery',{})
        episode=meta.get('episode');now=float(info['timestamp'])
        if not episode or not self.cell_mode:return
        if not all(math.isfinite(float(episode[k])) for k in ('start','deadline','initial_gap_m','timestamp')) or episode['initial_gap_m']>=0 or now<episode['start'] or abs(episode['deadline']-episode['start']-RESTORE_SECONDS)>1e-6:
            raise ValueError('Invalid unknown-margin restoration deadline')
        expected=dict(scene_token=info['scene_token'],timestamp=now)
        if any(episode.get(k)!=v for k,v in expected.items()):raise ValueError('Stale unknown-margin restoration state')
        gap=float(self.unknown_gap(initial[None],False)[0]);physical=float(self.unknown_gap(initial[None],True)[0])
        elapsed=np.clip((now-float(episode['start']))/RESTORE_SECONDS,0.,1.)
        allowed=float(episode['initial_gap_m'])*(1.-elapsed**2)
        known=float(self.known.clearance(initial[None],[0.])[0])
        if gap>=-1e-5:return
        if known < -1e-5 or proposal_gap < -1e-5 or physical < -1e-5:
            self.recovery_rejection='measured_geometry_proposal_or_inferred_cell_overlap';return
        if now>=episode['deadline'] or gap<allowed-1e-5:
            self.recovery_rejection='restoration_deadline_or_progress_violated';return
        self.recovery=dict(episode,now=now,initial_physical_gap_m=physical,initial_nominal_gap_m=gap)
    def unknown_gap(self,states,physical=False):
        if not self.cell_mode:raise ValueError('Restoration requires explicit projected unknown voxels')
        expanded=np.c_[states,np.full(len(states),self.vehicle.half_width_m),np.full(len(states),self.vehicle.half_length_m)]
        return polygon_clearance(expanded,self.cell_centers,self.cell_polygon,0. if physical else self.vehicle.clearance_margin_m+self.cell_extra,tree=self.cell_tree)
    def clearance(self,states,times,unknown_times=None):
        if self.recovery is None:return super().clearance(states,times)
        unknown_times=np.asarray(times if unknown_times is None else unknown_times)
        elapsed=np.clip((self.recovery['now']+unknown_times-self.recovery['start'])/RESTORE_SECONDS,0.,1.)
        floor=self.recovery['initial_gap_m']*(1.-elapsed**2)
        nominal=self.unknown_gap(states,False)
        return np.minimum(self.known.clearance(states,times),np.minimum(nominal-floor,self.unknown_gap(states,True)))
    def audit(self):
        result=super().audit();result['unknown_margin_recovery']=copy.deepcopy(self.recovery)
        result['unknown_margin_recovery_rejection']=self.recovery_rejection
        result['unknown_voxels_remain_hard']=True;result['observed_depth_and_occupancy_relaxed']=False
        return result

    def restoration_clearance(self,states,times,sweep):
        # Physical unknown voxels retain the full between-sample bound.
        # An already violated epistemic margin gets a sampled restoration
        # envelope; its original swept margin returns by the fixed deadline.
        times=np.asarray(times);elapsed=np.clip((self.recovery['now']+times-self.recovery['start'])/RESTORE_SECONDS,0.,1.)
        floor=self.recovery['initial_gap_m']*(1.-elapsed**2)
        nominal=self.unknown_gap(states,False)
        margin_progress=nominal-floor-np.where(elapsed>=1.,sweep,0.)
        physical=self.unknown_gap(states,True)-sweep
        return np.minimum(margin_progress,physical)
