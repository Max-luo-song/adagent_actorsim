"""Pose-aligned filtered model-label history; no future observations or references.

Interpolation weights are geometric label weights, not probabilities.
Current predictions remain authoritative until three covered observations.
"""
import numpy as np

POLICY='aligned_filtered_label_median_v2'

def pose2d(info):
    r=np.asarray(info['explicit_l2g_r_mat'],dtype=float);t=np.asarray(info['explicit_l2g_t'],dtype=float)
    if r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(t).all():raise ValueError('Invalid road pose metadata')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-4) or np.linalg.det(r)<.99:raise ValueError('Road pose is not rigid')
    yaw=np.arctan2(r[1,0],r[0,0]);c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,t[0]],[s,c,t[1]],[0.,0.,1.]])


def conservative_reproject(old_mask,old_bounds,old_pose,current_bounds,current_pose,shape):
    """Historical road vote only if every overlapped raster cell was road.

    A transformed cell's enclosing axis-aligned box is checked with an integral
    image. It may exclude valid road at a boundary but never uses center-only
    sampling to add road across an old non-road cell or a raster edge.
    """
    old_mask=np.asarray(old_mask,dtype=bool);ny,nx=shape
    x0,y0,x1,y1=np.asarray(current_bounds,dtype=float);dx=(x1-x0)/nx;dy=(y1-y0)/ny
    yy,xx=np.indices(shape);x=x0+xx*dx;y=y0+yy*dy
    transform=np.linalg.inv(old_pose)@current_pose
    corners=[]
    for a,b in ((0,0),(1,0),(1,1),(0,1)):
        p=np.stack([x+a*dx,y+b*dy,np.ones(shape)],axis=-1)
        corners.append(p@transform.T)
    corners=np.stack(corners)
    lo=corners[...,:2].min(0);hi=corners[...,:2].max(0)
    ox0,oy0,ox1,oy1=np.asarray(old_bounds,dtype=float);odx=(ox1-ox0)/old_mask.shape[1];ody=(oy1-oy0)/old_mask.shape[0]
    valid=(lo[...,0]>=ox0-1e-8)&(lo[...,1]>=oy0-1e-8)&(hi[...,0]<=ox1+1e-8)&(hi[...,1]<=oy1+1e-8)
    # Epsilon avoids counting a neighboring cell touched only at an edge.
    cx0=np.floor((lo[...,0]-ox0)/odx+1e-8).astype(int);cx1=np.ceil((hi[...,0]-ox0)/odx-1e-8).astype(int)
    cy0=np.floor((lo[...,1]-oy0)/ody+1e-8).astype(int);cy1=np.ceil((hi[...,1]-oy0)/ody-1e-8).astype(int)
    cx0=np.clip(cx0,0,old_mask.shape[1]);cx1=np.clip(cx1,0,old_mask.shape[1]);cy0=np.clip(cy0,0,old_mask.shape[0]);cy1=np.clip(cy1,0,old_mask.shape[0])
    integral=np.pad((~old_mask).astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
    blocked=integral[cy1,cx1]-integral[cy0,cx1]-integral[cy1,cx0]+integral[cy0,cx0]
    return valid&(blocked==0),valid


class RoadPredictionMemory:
    def __init__(self,vehicle,max_age_s=.25,max_frames=3):
        self.vehicle=vehicle;self.max_age_s=max_age_s;self.max_frames=max_frames;self.history=[]
    def update(self,contract,info):
        from drivable_area import DrivableArea,filter_road_speckles
        from scipy.ndimage import map_coordinates
        if 'temporal_drivable_mask' in contract:raise ValueError('Do not feed fused road masks back as independent observations')
        now=float(info['timestamp'])
        if not np.isfinite(now):raise ValueError('Non-finite road observation time')
        pose=pose2d(info);current=DrivableArea(contract,self.vehicle).mask
        scores=current.astype(float)  # geometric label interpolation weights, NOT probabilities
        if scores.shape!=current.shape or not np.isfinite(scores).all() or np.any(scores<0):raise ValueError('Invalid raw model road scores')
        if self.history and now<=self.history[-1]['timestamp']:raise ValueError('Non-increasing road observation clock')
        self.history=[row for row in self.history if now-row['timestamp']<=self.max_age_s+1e-8][-(self.max_frames-1):]
        fused_scores=scores.copy();used=np.zeros(scores.shape,bool)
        if len(self.history)>=2:
            b=np.asarray(contract['bounds_xy']);ny,nx=scores.shape;yy,xx=np.indices(scores.shape)
            points=np.stack([b[0]+(xx+.5)*(b[2]-b[0])/nx,b[1]+(yy+.5)*(b[3]-b[1])/ny,np.ones(scores.shape)],axis=-1)
            previous=[];valids=[]
            for row in self.history:
                q=points@(np.linalg.inv(row['pose'])@pose).T;ob=row['bounds'];old=row['scores']
                ix=(q[...,0]-ob[0])/((ob[2]-ob[0])/old.shape[1])-.5
                iy=(q[...,1]-ob[1])/((ob[3]-ob[1])/old.shape[0])-.5
                valid=(ix>=0)&(ix<=old.shape[1]-1)&(iy>=0)&(iy<=old.shape[0]-1)
                previous.append(map_coordinates(old,[iy,ix],order=1,mode='nearest',prefilter=False));valids.append(valid)
            used=np.logical_and.reduce(valids)
            median=np.median(np.stack([scores,*previous]),axis=0)
            fused_scores[used]=median[used]
        fused=fused_scores>.5
        if contract.get('road_mask_policy')=='bounded_road_speckle_filter_v1':fused,_=filter_road_speckles(fused,contract['bounds_xy'])
        audit=dict(policy=POLICY,source='frozen_uniad_map_head',observations=1+len(self.history),
            max_age_s=self.max_age_s,oldest_age_s=now-self.history[0]['timestamp'] if self.history else 0.,
            pose_source='model_explicit_l2g_metadata_planar_MPC_frame',historical_vote='bilinear_filtered_label_then_three_frame_median',
            raw_score_is_probability=False,decision_threshold=.5,insufficient_history='keep_current_prediction',
            fused_cells=int(used.sum()),current_filtered_road_cells=int(current.sum()),fused_road_cells=int(fused.sum()),
            added_cells=int((fused&~current).sum()),removed_cells=int((current&~fused).sum()),current_timestamp=now,
            ground_truth_used=False,ego_footprint_exemption=False,
            limitation='aligned filtered label consensus, not a certified road map; dynamic obstacle constraints remain independent')
        self.history.append(dict(scores=scores.copy(),bounds=np.asarray(contract['bounds_xy']).copy(),pose=pose,timestamp=now))
        return dict(contract,temporal_drivable_mask=fused.tolist(),road_temporal_consensus=audit)
