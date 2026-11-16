"""Frozen model map output -> metric full-footprint road constraint.

The source is UniAD's own map head, never simulator RoadNodes or annotations.
Map decoder mask scores are nonnegative ReLU outputs, NOT probabilities.
The model's existing >0.5 decision is preserved verbatim.
"""
import numpy as np
from scipy.spatial import cKDTree

SCHEMA='uniad_predicted_drivable_grid_v1'
AXES='rows_increasing_model_y_forward_columns_increasing_model_x_right'


def export_prediction(result,pc_range):
    if not isinstance(result,dict) or not isinstance(result.get('pts_bbox'),dict):
        raise ValueError('Missing UniAD map-head pts_bbox')
    values=result['pts_bbox']
    if 'drivable' not in values or 'score_list' not in values:
        raise ValueError('Missing predicted drivable mask/scores')
    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value)
    mask=array(values['drivable']);scores=array(values['score_list'])
    bounds=np.asarray(pc_range,dtype=float)
    if mask.ndim!=2 or min(mask.shape)<2 or mask.dtype!=np.bool_:
        raise ValueError('Invalid model drivable mask')
    if scores.ndim!=3 or scores.shape[1:]!=mask.shape or not len(scores):
        raise ValueError('Unaligned map-head score planes')
    raw=scores[-1].astype(float)
    if not np.isfinite(raw).all() or np.any(raw<0) or not np.array_equal(mask,raw>.5):
        raise ValueError('Drivable decoder decision contract differs')
    if bounds.shape!=(6,) or not np.isfinite(bounds).all() or np.any(bounds[3:]<=bounds[:3]):
        raise ValueError('Invalid map-head pc_range')
    return dict(drivable_mask=mask.copy(),drivable_raw_score=raw.astype(np.float32),
                drivable_bounds_xy=bounds[[0,1,3,4]],drivable_schema=np.asarray(SCHEMA),drivable_axes=np.asarray(AXES))


def from_saved(arrays):
    names=('drivable_mask','drivable_raw_score','drivable_bounds_xy','drivable_schema','drivable_axes')
    if not all(name in arrays for name in names):raise ValueError('Missing predicted drivable grid')
    schema=str(np.asarray(arrays['drivable_schema']).item());axes=str(np.asarray(arrays['drivable_axes']).item())
    mask=np.asarray(arrays['drivable_mask']);raw=np.asarray(arrays['drivable_raw_score'])
    if schema!=SCHEMA or axes!=AXES:raise ValueError('Unknown drivable coordinate contract')
    if mask.dtype!=np.bool_ or mask.ndim!=2 or raw.shape!=mask.shape or not np.isfinite(raw).all() or np.any(raw<0) or not np.array_equal(mask,raw>.5):
        raise ValueError('Invalid predicted drivable data')
    return dict(schema=schema,axes=axes,bounds_xy=np.asarray(arrays['drivable_bounds_xy']).tolist(),
                mask=mask.tolist(),source='frozen_uniad_map_head',threshold=.5,score_is_probability=False,
                ground_truth_used=False,road_mask_policy='bounded_road_speckle_filter_v1')


class DrivableArea:
    """Rectangle vs every non-road cell, including holes and raster exterior.

    Cell-center obstacle circles enclose entire cells; the existing rectangle
    SDF and unchanged vehicle clearance margin produce a conservative bound.
    Checking just the vehicle center/corners would miss interior sidewalk cells.
    """
    def __init__(self,contract,vehicle):
        if not isinstance(contract,dict) or contract.get('schema')!=SCHEMA or contract.get('axes')!=AXES:
            raise ValueError('Missing/unknown predicted drivable contract')
        if contract.get('source')!='frozen_uniad_map_head' or contract.get('ground_truth_used') is not False:
            raise ValueError('Road constraint must come from model perception')
        mask=np.asarray(contract.get('mask'));bounds=np.asarray(contract.get('bounds_xy'),dtype=float)
        if mask.ndim!=2 or min(mask.shape)<2 or mask.dtype!=np.bool_ or bounds.shape!=(4,) or not np.isfinite(bounds).all() or np.any(bounds[2:]<=bounds[:2]):
            raise ValueError('Invalid metric drivable raster')
        self.raw_mask=mask.copy();self.filter_audit=dict(enabled=False,changed_cells=0)
        if contract.get('road_mask_policy')=='bounded_road_speckle_filter_v1':
            mask,self.filter_audit=filter_road_speckles(mask,bounds)
        self.vehicle=vehicle;self.bounds=bounds;self.mask=mask
        self.dy=(bounds[3]-bounds[1])/mask.shape[0];self.dx=(bounds[2]-bounds[0])/mask.shape[1]
        rc=np.argwhere(~mask)
        self.points=np.c_[bounds[0]+(rc[:,1]+.5)*self.dx,bounds[1]+(rc[:,0]+.5)*self.dy]
        self.radius=float(np.hypot(self.dx,self.dy)/2)
        self.tree=cKDTree(self.points) if len(self.points) else None
    def clearance(self,states):
        from planning_guard import occupancy_clearance
        states=np.asarray(states,dtype=float)
        if states.ndim!=2 or states.shape[1]!=5 or not np.isfinite(states).all():raise ValueError('Invalid drivable footprint query')
        c=abs(np.cos(states[:,2]));s=abs(np.sin(states[:,2]))
        hx=self.vehicle.half_width_m*c+self.vehicle.half_length_m*s
        hy=self.vehicle.half_width_m*s+self.vehicle.half_length_m*c
        x0,y0,x1,y1=self.bounds
        result=np.minimum.reduce([states[:,0]-hx-x0,x1-states[:,0]-hx,states[:,1]-hy-y0,y1-states[:,1]-hy])-self.vehicle.clearance_margin_m
        if self.tree is not None and len(states):
            distance,_=self.tree.query(states[:,:2]);radius=2*np.hypot(self.vehicle.half_width_m,self.vehicle.half_length_m)
            neighbors=self.tree.query_ball_point(states[:,:2],distance+radius+1e-8)
            ids=np.unique(np.concatenate(neighbors)).astype(int)
            result=np.minimum(result,occupancy_clearance(states,self.points[ids],self.vehicle,self.radius))
        return result
    def audit(self):
        return dict(source='frozen_uniad_map_head',shape=list(self.mask.shape),bounds_xy=self.bounds.tolist(),
                    resolution_xy_m=[self.dx,self.dy],road_cells=int(self.mask.sum()),non_road_cells=int((~self.mask).sum()),
                    cell_half_diagonal_m=self.radius,axes=AXES,full_footprint=True,
                    missing_or_outside_means_free=False,ground_truth_used=False,
                    raw_road_cells=int(self.raw_mask.sum()),road_mask_filter=self.filter_audit,
                    limitation='model prediction; bounded speckle filtering is not ground-truth road certification')


def filter_road_speckles(mask,bounds):
    """Remove only fully enclosed, <1-square-metre non-road prediction components.

    Four-connected components touching any image edge or extending more than
    1.536 m on either axis remain hard constraints. No current-ego exemption.
    Raw predictions remain saved, with all changed cells auditable. This only
    filters a road semantic raster: occupancy/depth/VRU constraints stay hard.
    """
    from scipy.ndimage import label
    mask=np.asarray(mask,dtype=bool);bounds=np.asarray(bounds,dtype=float)
    cc,n=label(~mask,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]));size=np.bincount(cc.ravel())
    edge=set(np.unique(np.r_[cc[0],cc[-1],cc[:,0],cc[:,-1]]).tolist())
    dx=(bounds[2]-bounds[0])/mask.shape[1];dy=(bounds[3]-bounds[1])/mask.shape[0]
    changed=np.zeros(mask.shape,dtype=bool);removed=[]
    for k in np.flatnonzero((size>0)&(size*dx*dy<=1.0)):
        if k==0 or k in edge:continue
        rows,cols=np.where(cc==k)
        if (cols.max()-cols.min()+1)*dx>1.536+1e-9 or (rows.max()-rows.min()+1)*dy>1.536+1e-9:continue
        changed[rows,cols]=True;removed.append(dict(cells=int(size[k]),rows=rows.tolist(),columns=cols.tolist()))
    return mask|changed,dict(enabled=True,policy='bounded_road_speckle_filter_v1',
        connectivity=4,max_area_m2=1.0,max_extent_m=1.536,edge_components_preserved=True,
        changed_cells=int(changed.sum()),removed_components=removed,raw_predictions_preserved=True,
        occupancy_depth_vru_constraints_unchanged=True)
