import numpy as np
from empty_bounded_geometry import TemporalOccupancy as OriginalTemporalOccupancy,polygon_clearance
from mpc_native_v1.native import occupancy_clearance
from collections import OrderedDict
from mpc_perf_v1.exact_memo import key
from polygon_backend import CachedPolygon,STATS

def raster_clearance(self, states, times):
    states = np.asarray(states)
    times = np.asarray(times)
    if len(states) != len(times) or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('Invalid occupancy query times')
    c = np.abs(np.cos(states[:, 2]))
    s = np.abs(np.sin(states[:, 2]))
    half_x = self.vehicle.half_width_m * c + self.vehicle.half_length_m * s
    half_y = self.vehicle.half_width_m * s + self.vehicle.half_length_m * c
    result = np.minimum(50.0 - np.abs(states[:, 0]) - half_x, 50.0 - np.abs(states[:, 1]) - half_y) - self.vehicle.clearance_margin_m
    for indices,points in self._groups(times):
        if len(points):
            result[indices] = np.minimum(result[indices], occupancy_clearance(states[indices], points, self.vehicle))
    if len(self.observed_points):
        (distance, _) = self.observed_tree.query(states[:, :2])
        neighborhoods = self.observed_tree.query_ball_point(states[:, :2], distance + 2 * np.hypot(self.vehicle.half_width_m, self.vehicle.half_length_m) + 1e-08)
        ids = np.unique(np.concatenate(neighborhoods)).astype(int)
        result = np.minimum(result, occupancy_clearance(states, self.observed_points[ids], self.vehicle, self.observed_point_radius))
    if len(self.unknown_points):
        (distance, _) = self.unknown_tree.query(states[:, :2])
        neighborhoods = self.unknown_tree.query_ball_point(states[:, :2], distance + 2 * np.hypot(self.vehicle.half_width_m, self.vehicle.half_length_m) + 1e-08)
        ids = np.unique(np.concatenate(neighborhoods)).astype(int)
        result = np.minimum(result, occupancy_clearance(states, self.unknown_points[ids], self.vehicle, self.unknown_point_radius))
    return result

class TemporalOccupancy(OriginalTemporalOccupancy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._schedule=OrderedDict()
        if self.cell_mode:self._unknown_polygon=CachedPolygon(self.cell_centers,self.cell_polygon,self.vehicle.clearance_margin_m+self.cell_extra,tree=self.cell_tree)
        if self.depth_cell_mode:self._depth_polygon=CachedPolygon(self.depth_centers,self.depth_polygon,self.vehicle.clearance_margin_m+self.depth_extra,tree=self.depth_tree)
    def _groups(self,times):
        k=key(times)
        if k in self._schedule:
            STATS['schedule_hits']+=1;self._schedule.move_to_end(k);return self._schedule[k]
        STATS['schedule_misses']+=1
        index=np.clip(times/.5,0.,4.);lo=np.floor(index+1e-8).astype(int);hi=np.ceil(index-1e-8).astype(int)
        groups=[]
        for a,b in set(zip(lo.tolist(),hi.tolist())):
            indices=np.flatnonzero((lo==a)&(hi==b))
            points=self.points[a] if a==b else np.concatenate([self.points[a],self.points[b]])
            groups.append((indices,points))
        self._schedule[k]=groups
        if len(self._schedule)>32:self._schedule.popitem(last=False)
        return groups
    def clearance(self,states,times):
        result=raster_clearance(self,states,times)
        if not (self.cell_mode or self.depth_cell_mode):return result
        expanded=np.c_[states,np.full(len(states),self.vehicle.half_width_m),np.full(len(states),self.vehicle.half_length_m)]
        if self.cell_mode:result=np.minimum(result,self._unknown_polygon.clearance(expanded))
        if self.depth_cell_mode:result=np.minimum(result,self._depth_polygon.clearance(expanded))
        return result
