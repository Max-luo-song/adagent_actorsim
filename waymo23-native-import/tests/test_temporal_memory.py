"""Temporal occupancy and poisoning/duplication regressions."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning_guard import TemporalOccupancy,ObstacleMemory
from runtime_configuration import load_vehicle_config
class OccupancyTests(unittest.TestCase):
    def test_future_collision_is_not_current_overlap(self):
        masks=np.zeros((1,5,200,200));masks[0,4,100,100]=1
        occ=TemporalOccupancy(masks,load_vehicle_config())
        gaps=occ.clearance(np.zeros((2,5)),[0.,2.])
        self.assertGreater(gaps[0],0);self.assertLess(gaps[1],0)
    def test_current_collision_is_still_blocked(self):
        masks=np.zeros((1,5,200,200));masks[0,0,100,100]=1
        self.assertLess(TemporalOccupancy(masks,load_vehicle_config()).clearance(np.zeros((1,5)),[0.])[0],0)
    def test_near_future_hazard_is_preserved_and_empty_bins_finite(self):
        masks=np.zeros((1,5,200,200));masks[0,1,100,100]=1
        gaps=TemporalOccupancy(masks,load_vehicle_config()).clearance(np.zeros((3,5)),[0.,.1,3.])
        self.assertTrue(np.isfinite(gaps).all());self.assertLess(gaps[1],0);self.assertGreater(gaps[0],0)
    def test_all_empty_remains_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'Empty occupancy'):TemporalOccupancy(np.zeros((1,5,200,200)),load_vehicle_config())
class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory=ObstacleMemory();self.info=dict(scene_token='a',timestamp=0.,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
    def update(self,t,score=.8,x=0.,y=8.,n=1):
        self.info['timestamp']=t;boxes=np.array([[x+i*.1,y,0,2,4,1,0,0,0] for i in range(n)],float)
        rows=[dict(candidate_index=i,state='network_retained',raw_score=score,box=box.tolist(),label=0) for i,box in enumerate(boxes)] if score>=.4 else []
        return self.memory.update(dict(config={},obstacles=rows),boxes,np.full(n,score),np.zeros(n),self.info)
    def confirm(self):
        for t in (0.,.1,.2):self.update(t)
    def test_low_score_queries_cannot_drag_anchor_towards_ego(self):
        self.confirm()
        for t,y in [(0.3,7.),(.4,6.),(.5,5.),(.6,4.)]:r=self.update(t,.1,y=y)
        remembered=next(o for o in r['obstacles'] if o['state']=='last_known_confirmed')
        self.assertAlmostEqual(remembered['box'][1],8.);self.assertEqual(r['obstacle_memory']['tracks'],1)
    def test_confirmed_loss_does_not_expire_as_free_space(self):
        self.confirm()
        for t in np.arange(.3,1.6,.1):r=self.update(float(t),.01)
        self.assertTrue(any(o['state']=='last_known_confirmed' for o in r['obstacles']))
    def test_single_frame_track_does_not_become_permanent(self):
        self.update(0.)
        for t in np.arange(.1,1.1,.1):r=self.update(float(t),.01)
        self.assertEqual(r['obstacle_memory']['tracks'],0)
    def test_duplicate_queries_do_not_grow_permanent_track_count(self):
        for t in np.arange(0.,1.,.1):r=self.update(float(t),n=3)
        self.assertEqual(r['obstacle_memory']['tracks'],1)
    def test_matched_detection_uses_filtered_anchor_without_changing_raw_output(self):
        self.update(0.,y=8.);self.update(.1,y=8.)
        result=self.update(.2,y=7.7)
        current=next(o for o in result['obstacles'] if o['candidate_index']==0)
        self.assertAlmostEqual(current['box'][1],8.)
        self.assertAlmostEqual(current['raw_box'][1],7.7)
        self.assertEqual(current['geometry_anchor'],'temporal_world_median')

    def test_motion_compensated_filter_does_not_lag_moving_vehicle(self):
        for time in (0.,.1,.2):
            boxes=np.array([[0,8+2*time,0,2,4,1,0,0,2]],float)
            report=dict(config={},obstacles=[dict(candidate_index=0,state='network_retained',raw_score=.8,box=boxes[0].tolist(),label=0)])
            self.info['timestamp']=time
            result=self.memory.update(report,boxes,[.8],[0],self.info)
        self.assertAlmostEqual(result['obstacles'][0]['box'][1],8.4)
        self.assertAlmostEqual(result['obstacles'][0]['box'][8],2.)

class GeometryHistoryTests(unittest.TestCase):
    def test_floating_vehicle_quarantined_by_independent_ground(self):
        xx,yy=np.meshgrid(np.arange(-2,3,.2),np.arange(6,11,.2))
        ground=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])
        memory=ObstacleMemory(ground)
        boxes=np.array([[0,8,0,2,4,1.5,0,0,0],[1,8,3,2,4,1.5,0,0,0]],float)
        rows=[dict(candidate_index=i,state='network_retained',raw_score=.8,box=b.tolist(),label=0) for i,b in enumerate(boxes)]
        original=dict(config={},obstacles=rows)
        info=dict(timestamp=0.,scene_token='a',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
        result=memory.update(original,boxes,[.8,.8],[0,0],info)
        self.assertEqual(len(original['obstacles']),2)
        self.assertEqual([o['candidate_index'] for o in result['obstacles']],[0])
        self.assertEqual(result['geometry_rejections'][0]['candidate_index'],1)
    def test_stale_motion_cannot_sweep_entire_history_through_ego(self):
        memory=ObstacleMemory();info=dict(scene_token='a',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3),ego_velo=2.)
        for t in np.arange(0.,1.3,.1):
            info['timestamp']=float(t)
            boxes=np.array([[0,8,0,2,4,1,0,0,-3.]],float)
            rows=[dict(candidate_index=0,state='network_retained',raw_score=.8,box=boxes[0].tolist(),label=0)] if t<.3 else []
            result=memory.update(dict(config={},obstacles=rows),boxes,[.8 if t<.3 else .01],[0],info)
        row=next(o for o in result['obstacles'] if o['state']=='last_known_confirmed')
        self.assertTrue(row['motion_model_expired']);self.assertEqual(row['box'][7:9],[0.,0.])
        self.assertAlmostEqual(row['box'][1],8.)

if __name__=='__main__':unittest.main()
