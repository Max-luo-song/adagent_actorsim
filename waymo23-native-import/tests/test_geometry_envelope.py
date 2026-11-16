"""Conservative geometry coverage without training or scene truth."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'waymo172_closed_loop_v1')]
from planning_guard import envelope_separation,PlanningGuard,ObstacleMemory
from runtime_configuration import load_vehicle_config

class GeometryEnvelopeTests(unittest.TestCase):
    def test_encloses_arbitrary_obstacle_heading(self):
        box=np.array([[0.,0.,0.,2.,5.,1.,0.,0.,0.]])
        radius=np.hypot(2.,5.)/2
        for yaw in np.linspace(-np.pi,np.pi,61):
            for a,b in ((-1,-1),(-1,1),(1,-1),(1,1)):
                point=np.array([a*np.cos(yaw)-b*2.5*np.sin(yaw),a*np.sin(yaw)+b*2.5*np.cos(yaw)])
                state=np.array([[*point,0.,0.,0.]])
                self.assertLessEqual(envelope_separation(state,box,[0.],[radius],0.,0.)[0,0],1e-12)

    def test_stationary_hypothesis_not_erased_by_velocity(self):
        box=np.array([[0.,3.,0.,2.,4.,1.,0.,0.,5.]])
        state=np.array([[0.,0.,0.,0.,0.]])
        gap=envelope_separation(state,box,[3.],[np.hypot(2.,4.)/2],.8,1.5)[0,0]
        self.assertLess(gap,0.)

    def test_ego_keeps_rectangle_and_narrow_side_clearance(self):
        box=np.array([[1.7,0.,0.,.4,.4,1.,0.,0.,0.]])
        gap=envelope_separation(np.zeros((1,5)),box,[0.],[.3],.8,1.5)[0,0]
        self.assertAlmostEqual(gap,.6)

    def test_motion_uncertainty_does_not_expand_sideways(self):
        box=np.array([[5.,-20.,0.,2.,4.,1.,0.,0.,20.]])
        gap=envelope_separation(np.zeros((1,5)),box,[2.],[2.3],.8,1.5)[0,0]
        self.assertAlmostEqual(gap,1.9)

    def test_crossing_segment_is_blocked_between_endpoints(self):
        box=np.array([[-10.,0.,0.,2.,4.,1.,0.,10.,0.]])
        gap=envelope_separation(np.zeros((1,5)),box,[2.],[2.3],.8,1.5)[0,0]
        self.assertLess(gap,0.)

    def test_retained_score_does_not_reduce_uncertainty_margin(self):
        o=dict(candidate_index=1,state='network_retained',raw_score=.95,box=[0,10,0,2,4,1,0,0,0])
        g=PlanningGuard(dict(config={},obstacles=[o]),load_vehicle_config(),np.zeros((1,5)),[0.])
        self.assertEqual(g.rows[0]['effective_margin_m'],.5)
        self.assertEqual(g.rows[0]['geometry'],'heading_free_disk_motion_envelope')

    def test_memory_keeps_max_size_and_world_residual(self):
        m=ObstacleMemory();info=dict(timestamp=0.,scene_token='a',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
        def update(box):
            return m.update(dict(config={},obstacles=[dict(candidate_index=0,state='network_retained',raw_score=.8,box=box,label=0)]),np.array([box]),[.8],[0],info)
        update([0,8,0,2,5,1,0,0,0]);info['timestamp']=.1
        update([.6,8,0,2,4,1,0,0,0]);info['timestamp']=.2
        r=update([.6,8,0,2,4,1,0,0,0]);row=r['obstacles'][0]
        self.assertAlmostEqual(row['geometry_radius_m'],np.hypot(2,5)/2)
        self.assertAlmostEqual(row['geometry_position_uncertainty_m'],.2)

if __name__=='__main__':unittest.main()
