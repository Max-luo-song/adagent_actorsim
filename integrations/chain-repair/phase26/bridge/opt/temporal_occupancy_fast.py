"""Faster TemporalOccupancy: identical constraint math, less redundant work.

Two semantics-preserving changes vs the phase11/repo chain:

1. Per-instance cache of the concatenated point arrays (and their cKDTree) for
   each occupancy time-bin pair. The masks are fixed at build time, so these
   arrays are invariant for the instance's lifetime; the instance is rebuilt
   every simulation step, so the cache can never outlive the data it derives
   from. Invalidation boundary: instance construction (new masks -> new object).

2. Exact pruning for grid bins, the same bound the installed code already uses
   for observed/unknown points: for rectangle SDF,
   sdf(q) >= dist(q, centre) - R_body, and the nearest-centre point satisfies
   sdf(p*) <= dist(p*, centre). Points farther than nearest + 2*R_body can
   never attain the minimum, so dropping them does not change the result in
   exact arithmetic. Margin/radius subtractions are uniform constants and do
   not affect the argmin.

The constraint formula, distance definitions, safety margins, candidates,
solver tolerances and budgets are untouched (same occupancy_clearance /
polygon_clearance calls on a provably sufficient subset).
"""
import numpy as np
from scipy.spatial import cKDTree

from planning_guard import occupancy_clearance  # noqa: F401  (re-export for audits)
from bounded_visibility_geometry import TemporalOccupancy as _Base, polygon_clearance


class TemporalOccupancyFast(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pair_cache = {}
        self._body_diameter = 2*np.hypot(self.vehicle.half_width_m, self.vehicle.half_length_m)

    def _pair_points(self, a, b):
        entry = self._pair_cache.get((a, b))
        if entry is None:
            points = self.points[a] if a == b else np.concatenate([self.points[a], self.points[b]])
            entry = (points, cKDTree(points) if len(points) else None)
            self._pair_cache[(a, b)] = entry
        return entry

    def clearance(self, states, times):
        states = np.asarray(states)
        times = np.asarray(times)
        if len(states) != len(times) or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError('Invalid occupancy query times')
        index = np.clip(times/.5, 0., 4.)
        lo = np.floor(index+1e-8).astype(int)
        hi = np.ceil(index-1e-8).astype(int)
        c = np.abs(np.cos(states[:, 2])); s = np.abs(np.sin(states[:, 2]))
        half_x = self.vehicle.half_width_m*c+self.vehicle.half_length_m*s
        half_y = self.vehicle.half_width_m*s+self.vehicle.half_length_m*c
        result = np.minimum(50.-np.abs(states[:, 0])-half_x, 50.-np.abs(states[:, 1])-half_y)-self.vehicle.clearance_margin_m
        for a, b in set(zip(lo.tolist(), hi.tolist())):
            indices = np.flatnonzero((lo == a) & (hi == b))
            points, tree = self._pair_points(a, b)
            if not len(points):
                continue
            if tree is not None and len(points) > 64:
                distance, _ = tree.query(states[indices, :2])
                keep = tree.query_ball_point(states[indices, :2], distance+self._body_diameter+1e-8)
                ids = np.unique(np.concatenate(keep)).astype(int)
                points = points[ids]
            result[indices] = np.minimum(result[indices], occupancy_clearance(states[indices], points, self.vehicle))
        # Observed/unknown branches: installed exact pruning already; reuse unchanged.
        if len(self.observed_points):
            distance, _ = self.observed_tree.query(states[:, :2])
            neighborhoods = self.observed_tree.query_ball_point(states[:, :2], distance+self._body_diameter+1e-8)
            ids = np.unique(np.concatenate(neighborhoods)).astype(int)
            result = np.minimum(result, occupancy_clearance(states, self.observed_points[ids], self.vehicle, self.observed_point_radius))
        if len(self.unknown_points):
            distance, _ = self.unknown_tree.query(states[:, :2])
            neighborhoods = self.unknown_tree.query_ball_point(states[:, :2], distance+self._body_diameter+1e-8)
            ids = np.unique(np.concatenate(neighborhoods)).astype(int)
            result = np.minimum(result, occupancy_clearance(states, self.unknown_points[ids], self.vehicle, self.unknown_point_radius))
        # polygon (rotated world-voxel) part from the phase11 subclass, unchanged:
        if self.cell_mode or self.depth_cell_mode:
            expanded = np.c_[states, np.full(len(states), self.vehicle.half_width_m), np.full(len(states), self.vehicle.half_length_m)]
            if self.cell_mode:
                result = np.minimum(result, polygon_clearance(expanded, self.cell_centers, self.cell_polygon, self.vehicle.clearance_margin_m+self.cell_extra, tree=self.cell_tree))
            if self.depth_cell_mode:
                result = np.minimum(result, polygon_clearance(expanded, self.depth_centers, self.depth_polygon, self.vehicle.clearance_margin_m+self.depth_extra, tree=self.depth_tree))
        return result
