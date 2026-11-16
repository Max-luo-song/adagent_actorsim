from bootstrap import *
import unittest,types,numpy as np
from exact_queries import RoadPoseCache
from drivable_area import DrivableArea,SCHEMA,AXES
from runtime_configuration import load_vehicle_config
from native_objective import NativeAlignedObjective
class ExactQueryTests(unittest.TestCase):
 def road(self,mask):return DrivableArea({'schema':SCHEMA,'axes':AXES,'bounds_xy':[-10,-10,10,10],'mask':mask.tolist(),'ground_truth_used':False,'source':'frozen_uniad_map_head','threshold':.5},load_vehicle_config())
 def test_road_queries_match_full_batch_with_holes_rotations_and_duplicates(self):
  rng=np.random.default_rng(23);mask=rng.uniform(size=(30,30))>.15;mask[:,0]=False;road=self.road(mask);states=np.c_[rng.uniform(-12,12,(80,2)),rng.uniform(-3.1,3.1,80),rng.uniform(0,8,80),rng.uniform(-.4,.4,80)];states[10:20]=states[:10];expected=road.clearance(states);cached=RoadPoseCache(road.clearance);actual=cached(states);np.testing.assert_array_equal(actual,expected);evaluated=cached.metrics['evaluated_rows'];changed=states.copy();changed[:,3:]+=1.;np.testing.assert_array_equal(cached(changed),road.clearance(changed));self.assertEqual(cached.metrics['evaluated_rows'],evaluated)
 def test_road_cache_has_no_rounding_no_invalid_bypass_or_scene_leak(self):
  road=self.road(np.ones((30,30),dtype=bool));cached=RoadPoseCache(road.clearance);states=np.zeros((1,5));a=cached(states);a[0]=-99.;self.assertGreater(cached(states)[0],0.);states[0,0]=1e-12;cached(states);self.assertEqual(cached.metrics['evaluated_rows'],2);states[0,3]=np.nan
  with self.assertRaises(ValueError):cached(states)
  other=self.road(np.zeros((30,30),dtype=bool));self.assertLess(RoadPoseCache(other.clearance)(np.zeros((1,5)))[0],0.)
 def test_late_controls_do_not_repeat_unchanged_ttc_queries(self):
  calls=[];temporal=types.SimpleNamespace(clearance=lambda states,times:(calls.append(states.copy()) or np.full(len(states),100.)));guard=types.SimpleNamespace(clearance=lambda states,times:np.full(len(states),100.));v=load_vehicle_config();plan=np.c_[np.zeros(6),np.arange(1,7)*.5];objective=NativeAlignedObjective(plan,v,None,temporal,guard,lambda states:np.full(len(states),100.));states=np.zeros((61,5));states[:,1]=np.arange(61)*.1;states[:,3]=1.;lidar=states[:,:2].copy();controls=np.zeros((6,2));before=objective.components(states,lidar,controls);count=len(calls);self.assertEqual(count,2);states[30:,0]+=.2;after=objective.components(states,lidar,controls);self.assertEqual(len(calls),count);self.assertEqual(before['ttc_proxy'],after['ttc_proxy']);states[5,1]+=.01;objective.components(states,lidar,controls);self.assertGreater(len(calls),count)
if __name__=='__main__':unittest.main(verbosity=2)
