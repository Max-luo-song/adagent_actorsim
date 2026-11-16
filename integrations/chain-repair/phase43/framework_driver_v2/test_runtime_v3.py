from bootstrap import *
import unittest,numpy as np
from prefix_rollout import PrefixRollout
from empty_controller import rollout
from grouped_occupancy import GroupedOccupancy
from mpc_native_v1.native import occupancy_clearance
from runtime_configuration import load_vehicle_config
from planning_guard import GRID_HALF_DIAGONAL_M
from polygon_backend import CompiledCachedPolygon,BulkCompiledPolygon
from bulk_road_cache import BulkRoadPoseCache
from scipy.spatial import cKDTree
class RuntimeTests(unittest.TestCase):
 def test_prefix_exact_for_finite_difference_constant_and_linear(self):
  rng=np.random.default_rng(562)
  for mode in ('constant','linear'):
   for interval in (.5,1.):
    engine=PrefixRollout(7.1,.04,[.8,-1.2],.1,2.7,interval,mode,capacity=18)
    for _ in range(4):
     base=rng.normal(0.,.07,(6,2));queries=[base.copy()]
     for j in range(12):
      q=base.copy();q.flat[j]+=np.sqrt(np.finfo(float).eps);queries.append(q)
     for q in queries:
      actual=engine(q);truth=rollout(q,7.1,.04,np.array([.8,-1.2]),.1,2.7,interval,mode)
      for x,y in zip(actual,truth):self.assertEqual(x.tobytes(),y.tobytes())
     actual[0][:]=999.;self.assertLess(float(engine(queries[-1])[0][0,0]),10.)
    self.assertGreater(engine.metrics['reused_ticks'],0);self.assertLessEqual(len(engine.cache),18)
 def test_prefix_isolation_and_nearby_first_control_never_reused(self):
  q=np.zeros((6,2));engine=PrefixRollout(8.,0.,[0.,0.],.1,2.7,.5,'constant');engine(q);before=engine.metrics['reused_ticks'];q[0,0]=1e-12;actual=engine(q);self.assertEqual(engine.metrics['reused_ticks'],before)
  for x,y in zip(actual,rollout(q,8.,0.,np.zeros(2))):self.assertEqual(x.tobytes(),y.tobytes())
  other=PrefixRollout(3.,0.,[0.,0.],.1,2.7,.5,'constant');self.assertNotEqual(engine(q)[0][1,3],other(q)[0][1,3])
 def test_grouped_occupancy_matches_original_for_empty_and_rotated_groups(self):
  rng=np.random.default_rng(432);v=load_vehicle_config();s=rng.normal(size=(81,5));s[:,:2]*=15.;groups=[];expected=np.full(len(s),np.inf)
  for i in range(5):
   ids=np.arange(i,len(s),5);points=rng.uniform(-20,20,(i*23,2));groups.append((ids,points))
   if len(points):expected[ids]=occupancy_clearance(s[ids],points,v)
  actual=GroupedOccupancy(groups,len(s)).clearance(s,v,GRID_HALF_DIAGONAL_M);self.assertEqual(actual.tobytes(),expected.tobytes())
  self.assertEqual(GroupedOccupancy([],0).clearance(s[:0],v,GRID_HALF_DIAGONAL_M).shape,(0,))
 def test_bulk_polygon_exact_keys_duplicates_mutation_and_eviction(self):
  rng=np.random.default_rng(12);points=rng.normal(size=(60,2));polygon=np.array([[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]]);old=CompiledCachedPolygon(points,polygon,.2,tree=cKDTree(points),capacity=8);new=BulkCompiledPolygon(points,polygon,.2,tree=cKDTree(points),capacity=8)
  for _ in range(40):
   s=rng.normal(size=(9,7));s[:,5:]=[1.,2.];s[2]=s[0];truth=old.clearance(s);self.assertEqual(new.clearance(s).tobytes(),truth.tobytes());value=new.clearance(s);value[:]=999.;self.assertEqual(new.clearance(s).tobytes(),truth.tobytes())
  self.assertLessEqual(len(new.cache),8);self.assertLessEqual(len(new.batches),32)
 def test_bulk_road_cache_exact_subtle_changes_and_no_snapshot_leak(self):
  def raw(s):return s[:,0]+2*s[:,1]+3*s[:,2]
  cache=BulkRoadPoseCache(raw,capacity=3);s=np.zeros((5,5));s[1,0]=1e-12;a=cache(s);self.assertEqual(a[1],1e-12);a[:]=100.;np.testing.assert_array_equal(cache(s),raw(s));other=BulkRoadPoseCache(lambda s:raw(s)+1.);np.testing.assert_array_equal(other(s),raw(s)+1.)
 def test_raster_backend_is_bound_to_snapshot_and_matches_at_fractional_times(self):
  from empty_native_temporal import TemporalOccupancy
  from performance_backend import use_features
  rng=np.random.default_rng(14);v=load_vehicle_config();m=np.zeros((1,5,200,200));rr=rng.integers(40,160,(5,20,2))
  for k in range(5):m[0,k,rr[k,:,0],rr[k,:,1]]=1.
  with use_features({}):old=TemporalOccupancy(m,v)
  with use_features({'batched_occupancy':True}):new=TemporalOccupancy(m,v)
  s=rng.normal(size=(23,5));times=np.linspace(0.,3.1,len(s));self.assertFalse(old._batch_occupancy_enabled);self.assertTrue(new._batch_occupancy_enabled)
  self.assertEqual(old.clearance(s,times).tobytes(),new.clearance(s,times).tobytes())
  small=s.astype(np.float32);self.assertEqual(old.clearance(small,times).dtype,new.clearance(small,times).dtype);self.assertEqual(old.clearance(small,times).tobytes(),new.clearance(small,times).tobytes())
  for _ in range(35):
   times=times+1e-8;self.assertEqual(old.clearance(s,times).tobytes(),new.clearance(s,times).tobytes())
  self.assertLessEqual(len(new._grouped_occupancy_cache),32)
if __name__=='__main__':unittest.main(verbosity=2)
