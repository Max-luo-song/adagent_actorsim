from bootstrap import *
import json,unittest,numpy as np
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation
from bounded_visibility_geometry import polygon_clearance,projected_cell
from fast_polygon import PolygonKernel
from polygon_backend import CachedPolygon,CompiledCachedPolygon,use_backend
from mpc_reuse_v1.cached_geometry import CachedPolygon as ReferenceCachedPolygon
class PolygonTests(unittest.TestCase):
 def states(self,rng,n):
  s=np.zeros((n,7));s[:,:2]=rng.uniform(-20.,20.,(n,2));s[:,2]=rng.uniform(-np.pi,np.pi,n);s[:,3:5]=rng.normal(size=(n,2));s[:,5:7]=rng.uniform(.2,3.,(n,2));return s
 def test_random_polygons_voxels_and_tree_match_full_reference(self):
  rng=np.random.default_rng(491);count=0;maximum=0.;equal=0
  for case in range(12):
   points=rng.uniform(-24,24,(400,2));s=self.states(rng,90)
   if case%2:polygon=projected_cell(Rotation.random(random_state=rng).as_matrix())
   else:
    raw=rng.uniform(-.9,.9,(14,2));polygon=raw[ConvexHull(raw).vertices]
   kernel=PolygonKernel(polygon);tree=cKDTree(points)
   truth=polygon_clearance(s,points,polygon,.2000001)
   for search in [None,tree]:
    actual=kernel.clearance(s,points,.2000001,tree=search);np.testing.assert_allclose(actual,truth,rtol=0.,atol=1e-12);count+=len(s);equal+=int(np.sum(actual.view(np.uint64)==truth.view(np.uint64)));maximum=max(maximum,float(np.max(np.abs(actual-truth))))
  (HERE/'polygon_numerical_check.json').write_text(json.dumps(dict(passed=True,poses=count,bitwise_equal=equal,maximum_absolute_error_m=maximum,tolerance_m=1e-12),indent=2)+'\n')
 def test_contact_boundaries_and_variable_body_sizes(self):
  rng=np.random.default_rng(11);polygon=np.array([[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]])
  s=np.zeros((8,7));s[:,5]=1.;s[:,6]=2.;s[:,0]=np.array([-1e-8,-1e-10,0.,1e-10,1e-8,0.,0.,0.])+1.45;s[5:,2]=[np.pi/2,np.pi,-np.pi]
  points=np.zeros((1,2));truth=polygon_clearance(s,points,polygon,.2);actual=PolygonKernel(polygon).clearance(s,points,.2,tree=cKDTree(points));np.testing.assert_allclose(actual,truth,rtol=0.,atol=1e-12);np.testing.assert_array_equal(actual[:2]<0,truth[:2]<0);np.testing.assert_array_equal(actual[3:5]>0,truth[3:5]>0)
 def test_empty_float32_nonfinite_and_nonmutation(self):
  rng=np.random.default_rng(5);s=self.states(rng,8);points=rng.normal(size=(20,2));polygon=np.array([[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]]);kernel=PolygonKernel(polygon)
  saved=[x.copy() for x in [s,points,polygon]];actual=kernel.clearance(s,points,.2);np.testing.assert_allclose(actual,polygon_clearance(s,points,polygon,.2),rtol=0.,atol=1e-12)
  for x,y in zip([s,points,polygon],saved):np.testing.assert_array_equal(x,y)
  self.assertEqual(kernel.clearance(s[:0],points,.2).shape,(0,));self.assertTrue(np.isinf(kernel.clearance(s,points[:0],.2)).all())
  f=s.astype(np.float32);np.testing.assert_array_equal(kernel.clearance(f,points,.2),polygon_clearance(f,points,polygon,.2))
  f=s.copy();f[0,0]=np.nan
  with np.errstate(invalid='ignore'):np.testing.assert_allclose(kernel.clearance(f,points,.2),polygon_clearance(f,points,polygon,.2),rtol=0.,atol=0.,equal_nan=True)
 def test_backend_context_and_exact_cache_are_isolated(self):
  rng=np.random.default_rng(6);s=self.states(rng,8);points=rng.normal(size=(20,2));polygon=np.array([[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]])
  with use_backend(False):
   original=CachedPolygon(points,polygon,.2,tree=cKDTree(points));self.assertIsInstance(original,ReferenceCachedPolygon)
   with use_backend(True):compiled=CachedPolygon(points,polygon,.2,tree=cKDTree(points));self.assertIsInstance(compiled,CompiledCachedPolygon)
   self.assertIsInstance(CachedPolygon(points,polygon,.2),ReferenceCachedPolygon)
  value=compiled.clearance(np.vstack([s,s]));np.testing.assert_array_equal(value[:8],value[8:]);np.testing.assert_allclose(value[:8],original.clearance(s),rtol=0.,atol=1e-12)
  np.testing.assert_array_equal(value,compiled.clearance(np.vstack([s,s])))
  other=CompiledCachedPolygon(points+10.,polygon,.2,tree=cKDTree(points+10.));self.assertGreater(float(np.max(np.abs(other.clearance(s)-value[:8]))),.1)
if __name__=='__main__':unittest.main(verbosity=2)
