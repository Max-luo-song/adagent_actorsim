"""Whole world-voxel projection, with unchanged sensor and body margins."""
import numpy as np
from scipy.spatial import ConvexHull
from planning_guard import TemporalOccupancy as OriginalTemporalOccupancy

def projected_cell(rotation,resolution=.5):
    r=np.asarray(rotation,dtype=float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-5):raise ValueError('Invalid cell rotation')
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    xy=signs@r[:,:2]*resolution/2
    return xy[ConvexHull(xy).vertices]

def polygon_clearance(states,centers,polygon,margin,tree=None):
    states=np.asarray(states);centers=np.asarray(centers).reshape(-1,2)
    if not len(centers):return np.full(len(states),np.inf)
    result=[];edges=np.roll(polygon,-1,axis=0)-polygon
    normals=np.c_[-edges[:,1],edges[:,0]];normals/=np.linalg.norm(normals,axis=1)[:,None]
    if tree is not None:distances,nearest=tree.query(states[:,:2])
    cell_radius=float(np.linalg.norm(polygon,axis=1).max())
    for row,state in enumerate(states):
        yaw=state[2];right=np.array([np.cos(yaw),-np.sin(yaw)]);forward=np.array([np.sin(yaw),np.cos(yaw)])
        axes=np.vstack([right,forward,normals]);body_radius=np.abs(axes@right)*state[5]+np.abs(axes@forward)*state[6]
        projected=polygon@axes.T
        chosen=centers
        if tree is not None:
            near_delta=(centers[int(nearest[row])]-state[:2])@axes.T
            upper=float(np.maximum(near_delta+projected.min(0)-body_radius,-near_delta-projected.max(0)-body_radius).max())
            radius=max(float(distances[row])+1e-8,np.sqrt(2.)*(upper+np.hypot(state[5],state[6])+cell_radius)+1e-7)
            chosen=centers[tree.query_ball_point(state[:2],radius)]
        delta=(chosen-state[:2])@axes.T
        gaps=np.maximum(delta+projected.min(0)-body_radius,-delta-projected.max(0)-body_radius)
        result.append(float(gaps.max(1).min()-margin))
    return np.asarray(result)

class TemporalOccupancy(OriginalTemporalOccupancy):
    def __init__(self,*args,unknown_points=None,unknown_point_radius=None,unknown_cell_rotation=None,
                 observed_points=None,observed_point_radius=None,observed_cell_rotation=None,**kwargs):
        self.cell_mode=unknown_cell_rotation is not None
        self.depth_cell_mode=observed_cell_rotation is not None
        self.cell_centers=np.asarray(unknown_points if unknown_points is not None else [],dtype=float).reshape(-1,2)
        self.depth_centers=np.asarray(observed_points if observed_points is not None else [],dtype=float).reshape(-1,2)
        if not np.isfinite(self.cell_centers).all() or not np.isfinite(self.depth_centers).all():raise ValueError('Invalid cell centers')
        if self.cell_mode:
            self.cell_polygon=projected_cell(unknown_cell_rotation)
            self.cell_extra=float(unknown_point_radius)-np.sqrt(3)*.5/2
            if self.cell_extra< -1e-10 or not np.isfinite(self.cell_extra):raise ValueError('Invalid unknown uncertainty')
            self.cell_extra=max(0.,self.cell_extra)+1e-7
        if self.depth_cell_mode:
            self.depth_polygon=projected_cell(observed_cell_rotation)
            self.depth_extra=float(observed_point_radius)-np.sqrt(3)*.5/2
            if self.depth_extra< -1e-10 or not np.isfinite(self.depth_extra):raise ValueError('Invalid depth uncertainty')
            self.depth_extra=max(0.,self.depth_extra)+1e-7
        from scipy.spatial import cKDTree
        self.cell_tree=cKDTree(self.cell_centers) if len(self.cell_centers) else None
        self.depth_tree=cKDTree(self.depth_centers) if len(self.depth_centers) else None
        super().__init__(*args,unknown_points=None if self.cell_mode else unknown_points,
            unknown_point_radius=unknown_point_radius if unknown_point_radius is not None else np.sqrt(2)*.25,
            observed_points=None if self.depth_cell_mode else observed_points,
            observed_point_radius=observed_point_radius if observed_point_radius is not None else np.sqrt(2)*.25,**kwargs)
    def clearance(self,states,times):
        result=super().clearance(states,times)
        if not (self.cell_mode or self.depth_cell_mode):return result
        expanded=np.c_[states,np.full(len(states),self.vehicle.half_width_m),np.full(len(states),self.vehicle.half_length_m)]
        if self.cell_mode:result=np.minimum(result,polygon_clearance(expanded,self.cell_centers,self.cell_polygon,self.vehicle.clearance_margin_m+self.cell_extra,tree=self.cell_tree))
        if self.depth_cell_mode:result=np.minimum(result,polygon_clearance(expanded,self.depth_centers,self.depth_polygon,self.vehicle.clearance_margin_m+self.depth_extra,tree=self.depth_tree))
        return result
    def audit(self):
        d=super().audit()
        if self.cell_mode:d.update(unknown_visibility_cells=len(self.cell_centers),unknown_geometry='complete_projected_world_voxel_SAT',unknown_cell_polygon_xy=self.cell_polygon.tolist(),unknown_additional_margin_m=self.cell_extra,unknown_constraint_remains_hard=True)
        if self.depth_cell_mode:d.update(observed_depth_cells=len(self.depth_centers),observed_depth_geometry='complete_projected_world_voxel_SAT',observed_depth_additional_margin_m=self.depth_extra,observed_depth_constraint_remains_hard=True)
        return d
