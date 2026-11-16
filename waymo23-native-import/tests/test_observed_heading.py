"""Evidence-gated orientation geometry, independent of scene and asset."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning_guard import OccupancyHeadingSupport,rectangle_motion_envelope,rectangle_axes,PlanningGuard
from runtime_configuration import load_vehicle_config
class HeadingSupportTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((200,200));self.mask[112:120,98:102]=1
        self.box=np.array([0.,8.,0.,2.,4.,1.,0.,0.,0.])
        self.evidence=dict(geometry_heading_confirmed=True,geometry_heading_span_rad=0.,geometry_dimensions_wl_m=[2.,4.])
    def refine(self,box=None,evidence=None,mask=None):
        return OccupancyHeadingSupport(self.mask if mask is None else mask).refine(self.box if box is None else box,self.evidence if evidence is None else evidence,.5,np.sqrt(5))
    def test_agreement_encloses_all_bounded_headings(self):
        result=self.refine(evidence=dict(self.evidence,geometry_heading_span_rad=.3));self.assertIsNotNone(result);box,audit=result
        for yaw in np.linspace(-audit['heading_range_rad'],audit['heading_range_rad'],61):
            right,forward=rectangle_axes(yaw)
            for a,b in ((-1,-1),(-1,1),(1,-1),(1,1)):
                point=a*right+b*2*forward
                self.assertTrue(np.all(abs(point)<=np.array(audit['bounded_wl_m'])/2+1e-12))
    def test_no_temporal_evidence_preserves_disk(self):
        self.assertIsNone(self.refine(evidence={}))
    def test_heading_disagreement_preserves_disk(self):
        wrong=self.box.copy();wrong[6]=np.deg2rad(40.)
        self.assertIsNone(self.refine(box=wrong))
    def test_missing_or_merged_component_preserves_disk(self):
        self.assertIsNone(self.refine(mask=np.zeros((200,200))))
        mask=self.mask.copy();mask[115,101:130]=1
        self.assertIsNone(self.refine(mask=mask))
    def test_occluded_or_invalid_heading_keeps_disk(self):
        self.assertIsNone(self.refine(evidence=dict(self.evidence,state='last_known_confirmed')))
        for span in (float('nan'),-.1,np.pi/2):
            self.assertIsNone(self.refine(evidence=dict(self.evidence,geometry_heading_span_rad=span)))

    def test_metric_margin_unchanged_after_refinement(self):
        obstacle=dict(self.evidence,candidate_index=0,box=self.box.tolist(),state='network_retained',raw_score=.9)
        guard=PlanningGuard(dict(config={},obstacles=[obstacle]),load_vehicle_config(),np.zeros((1,5)),[0.],current_occupancy=self.mask)
        self.assertEqual(guard.rows[0]['effective_margin_m'],.5)
        self.assertEqual(guard.rows[0]['geometry'],'occupancy_supported_heading_motion_envelope')
    def test_motion_includes_stationary_and_crossing_hypotheses(self):
        states=np.zeros((1,5));box=self.box.copy();box[0:2]=[-8,0];box[7]=8
        self.assertLess(rectangle_motion_envelope(states,box[None],[2.],.8,1.5)[0,0],0.)
        box[0:2]=[0,3];box[7:9]=[0,8]
        self.assertLess(rectangle_motion_envelope(states,box[None],[2.],.8,1.5)[0,0],0.)
    def test_continuous_angle_and_motion_bounds_cover_sampled_corners(self):
        box=self.box.copy();box[7:9]=[1.,-.5];theta=.3
        for time in (0.,.7,2.):
            for fraction in (0.,.5,1.):
                for yaw in np.linspace(-theta,theta,13):
                    right,forward=rectangle_axes(yaw)
                    for a,b in ((-1,-1),(-1,1),(1,-1),(1,1)):
                        point=box[:2]+fraction*time*box[7:9]+a*right+b*2*forward
                        state=np.array([[*point,.2,0.,0.]])
                        gap=rectangle_motion_envelope(state,box[None],[time],0.,0.,heading_ranges=[theta])[0,0]
                        self.assertLessEqual(gap,1e-10)

    def test_geometry_is_rigid_transform_invariant(self):
        states=np.array([[3.,4.,.2,0.,0.],[4.,6.,-.3,0.,0.]])
        box=self.box.copy();box[7:9]=[.2,-.5];times=np.array([1.,2.])
        expected=rectangle_motion_envelope(states,box[None],times,.8,1.5,heading_ranges=[.3])
        angle=.7;rotation=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        offset=np.array([13.,-8.]);other=states.copy();other[:,:2]=states[:,:2]@rotation.T+offset;other[:,2]+=angle
        box[:2]=rotation@box[:2]+offset;box[6]+=angle;box[7:9]=rotation@box[7:9]
        actual=rectangle_motion_envelope(other,box[None],times,.8,1.5,heading_ranges=[.3])
        np.testing.assert_allclose(actual,expected,atol=1e-10)
if __name__=='__main__':unittest.main()
