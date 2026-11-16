"""Valid empty masks preserve all grid and external-geometry constraints."""
import numpy as np
from planning_guard import GRID_HALF_DIAGONAL_M, TemporalOccupancy as PreviousTemporalOccupancy
class TemporalOccupancy(PreviousTemporalOccupancy):
    def __init__(self,masks,vehicle,observed_points=None,observed_point_radius=GRID_HALF_DIAGONAL_M,unknown_points=None,unknown_point_radius=GRID_HALF_DIAGONAL_M):
        masks=np.asarray(masks)
        if masks.shape!=(1,5,200,200) or not np.isfinite(masks).all():
            raise ValueError('Unexpected temporal occupancy shape')
        self.observed_points=np.asarray(observed_points if observed_points is not None else [],dtype=float).reshape(-1,2)
        if not np.isfinite(self.observed_points).all():raise ValueError('Invalid observed geometry')
        self.observed_point_radius=float(observed_point_radius)
        if not np.isfinite(self.observed_point_radius) or self.observed_point_radius<0:raise ValueError('Invalid observed cell bound')
        from scipy.spatial import cKDTree
        self.observed_tree=cKDTree(self.observed_points) if len(self.observed_points) else None
        self.unknown_points=np.asarray(unknown_points if unknown_points is not None else [],dtype=float).reshape(-1,2)
        self.unknown_point_radius=float(unknown_point_radius)
        if not np.isfinite(self.unknown_points).all() or not np.isfinite(self.unknown_point_radius) or self.unknown_point_radius<0:raise ValueError('Invalid unknown visibility geometry')
        self.unknown_tree=cKDTree(self.unknown_points) if len(self.unknown_points) else None
        self.vehicle=vehicle;self.points=[]
        for mask in masks[0]:
            rc=np.argwhere(mask!=0)
            self.points.append((rc[:,[1,0]].astype(float)-100.)*.5+.25)
        self.counts=[len(p) for p in self.points]
        # A valid all-zero prediction contributes no neural obstacle cells.
