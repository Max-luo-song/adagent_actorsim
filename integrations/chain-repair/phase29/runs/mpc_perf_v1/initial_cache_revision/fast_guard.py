"""Keep the original trajectory guard; batch independent spatial queries."""
import numpy as np
from planning_guard import PlanningGuard as OriginalPlanningGuard,rectangle_separation,envelope_separation,rectangle_motion_envelope,sample_half_intervals

class PlanningGuard(OriginalPlanningGuard):
    def _evaluate(self,states,times,half_intervals):
        states=np.asarray(states,dtype=float);times=np.asarray(times,dtype=float)
        if not len(self.hard_boxes):return np.full(len(states),np.inf)
        gaps=np.empty((len(states),len(self.hard_boxes)))
        ordinary=~(self.hard_envelopes|self.hard_oriented)
        if np.any(ordinary):
            gaps[:,ordinary]=rectangle_separation(states,self.hard_boxes[ordinary],times,self.vehicle.half_width_m,self.vehicle.half_length_m)
        if np.any(self.hard_envelopes):
            mask=self.hard_envelopes;end=np.minimum(times+half_intervals,times[-1])
            gaps[:,mask]=envelope_separation(states,self.hard_boxes[mask],end,self.hard_radii[mask],self.vehicle.half_width_m,self.vehicle.half_length_m,ages=self.hard_ages[mask])
        if np.any(self.hard_oriented):
            mask=self.hard_oriented;end=np.minimum(times+half_intervals,times[-1])
            gaps[:,mask]=rectangle_motion_envelope(states,self.hard_boxes[mask],end,self.vehicle.half_width_m,self.vehicle.half_length_m,ages=self.hard_ages[mask],heading_ranges=self.hard_heading_ranges[mask])
        speed=np.linalg.norm(self.hard_boxes[:,7:9],axis=1)
        speed=np.where(self.hard_envelopes|self.hard_oriented,0.,speed)
        return np.min(gaps-self.hard_margins[None]-half_intervals[:,None]*speed[None],axis=1)
    def clearance(self,states,times):
        # Solver trajectories retain the original kernel and arithmetic.
        return OriginalPlanningGuard.clearance(self,states,times)
    def clearance_independent(self,states,time=0.):
        # Exactly the original singleton query for every point. No temporal
        # interpolation or artificial trajectory through the point batch.
        states=np.asarray(states,dtype=float)
        if not np.isfinite(time) or time<0:raise ValueError('Invalid independent query time')
        result=self._evaluate(states,np.full(len(states),float(time)),np.zeros(len(states)))
        # Only boolean road occupancy consumes this API. Resolve boundary
        # values with the exact old singleton arithmetic to avoid sign flips.
        for i in np.flatnonzero(np.abs(result)<=1e-10):
            result[i]=OriginalPlanningGuard.clearance(self,states[i:i+1],[time])[0]
        return result
