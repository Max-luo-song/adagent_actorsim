import numpy as np
from mpc_perf_v1.fast_guard import PlanningGuard as PreviousGuard
from planning_guard import rectangle_motion_envelope,sample_half_intervals
from mpc_native_v1.native import rectangle_separation,envelope_separation
from .cached_geometry import STATS

class PlanningGuard(PreviousGuard):
    def _evaluate(self,states,times,half_intervals):
        if not len(self.hard_boxes):return np.full(len(states),np.inf)
        states=np.asarray(states);times=np.asarray(times)
        gaps=np.empty((len(states),len(self.hard_boxes)))
        ordinary=~(self.hard_envelopes|self.hard_oriented)
        if np.any(ordinary):
            gaps[:,ordinary]=rectangle_separation(states,self.hard_boxes[ordinary],times,self.vehicle.half_width_m,self.vehicle.half_length_m)
        STATS['overwritten_rectangle_pairs_avoided']+=len(states)*int(np.count_nonzero(~ordinary))
        mask=self.hard_envelopes
        if np.any(mask):
            end=np.minimum(times+half_intervals,times[-1])
            gaps[:,mask]=envelope_separation(states,self.hard_boxes[mask],end,self.hard_radii[mask],self.vehicle.half_width_m,self.vehicle.half_length_m,ages=self.hard_ages[mask])
        oriented=self.hard_oriented
        if np.any(oriented):
            end=np.minimum(times+half_intervals,times[-1])
            gaps[:,oriented]=rectangle_motion_envelope(states,self.hard_boxes[oriented],end,self.vehicle.half_width_m,self.vehicle.half_length_m,ages=self.hard_ages[oriented],heading_ranges=self.hard_heading_ranges[oriented])
        speed=np.linalg.norm(self.hard_boxes[:,7:9],axis=1);speed=np.where(mask|oriented,0.,speed)
        return np.min(gaps-self.hard_margins[None]-half_intervals[:,None]*speed[None],axis=1)
    def clearance(self,states,times):
        if not len(self.hard_boxes):return np.full(len(states),np.inf)
        return self._evaluate(states,times,sample_half_intervals(times))
    def soft_cost(self,states,times):
        if not len(self.soft_boxes):return 0.
        gaps=rectangle_separation(states,self.soft_boxes,times,self.vehicle.half_width_m,self.vehicle.half_length_m)-self.soft_margins[None]
        return float(np.mean(np.maximum(-gaps,0)**2))
