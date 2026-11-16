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

def polygon_clearance(states,centers,polygon,margin):
    states=np.asarray(states);centers=np.asarray(centers).reshape(-1,2)
    if not len(centers):return np.full(len(states),np.inf)
    result=[];edges=np.roll(polygon,-1,axis=0)-polygon
    normals=np.c_[-edges[:,1],edges[:,0]];normals/=np.linalg.norm(normals,axis=1)[:,None]
    for state in states:
        yaw=state[2];right=np.array([np.cos(yaw),-np.sin(yaw)]);forward=np.array([np.sin(yaw),np.cos(yaw)])
        axes=np.vstack([right,forward,normals]);body_radius=np.abs(axes@right)*state[5]+np.abs(axes@forward)*state[6]
        projected=polygon@axes.T;delta=(centers-state[:2])@axes.T
        gaps=np.maximum(delta+projected.min(0)-body_radius,-delta-projected.max(0)-body_radius)
        result.append(float(gaps.max(1).min()-margin))
    return np.asarray(result)

class TemporalOccupancy(OriginalTemporalOccupancy):
    def __init__(self,*args,unknown_points=None,unknown_point_radius=None,unknown_cell_rotation=None,**kwargs):
        self.cell_mode=unknown_cell_rotation is not None
        if not self.cell_mode:
            super().__init__(*args,unknown_points=unknown_points,unknown_point_radius=unknown_point_radius if unknown_point_radius is not None else np.sqrt(2)*.25,**kwargs);return
        self.cell_centers=np.asarray(unknown_points if unknown_points is not None else [],dtype=float).reshape(-1,2)
        if not np.isfinite(self.cell_centers).all():raise ValueError('Invalid unknown cell centers')
        self.cell_polygon=projected_cell(unknown_cell_rotation)
        # Old radius = full 3D voxel half-diagonal + line-fit + sample spacing.
        self.cell_extra=float(unknown_point_radius)-np.sqrt(3)*.5/2
        if self.cell_extra<0 or not np.isfinite(self.cell_extra):raise ValueError('Invalid unknown cell uncertainty')
        super().__init__(*args,unknown_points=None,**kwargs)
    def clearance(self,states,times):
        result=super().clearance(states,times)
        if not self.cell_mode:return result
        expanded=np.c_[states,np.full(len(states),self.vehicle.half_width_m),np.full(len(states),self.vehicle.half_length_m)]
        return np.minimum(result,polygon_clearance(expanded,self.cell_centers,self.cell_polygon,self.vehicle.clearance_margin_m+self.cell_extra))
    def audit(self):
        d=super().audit()
        if self.cell_mode:d.update(unknown_visibility_cells=len(self.cell_centers),unknown_geometry='complete_projected_world_voxel_SAT',unknown_cell_polygon_xy=self.cell_polygon.tolist(),unknown_additional_margin_m=self.cell_extra,unknown_constraint_remains_hard=True)
        return d
