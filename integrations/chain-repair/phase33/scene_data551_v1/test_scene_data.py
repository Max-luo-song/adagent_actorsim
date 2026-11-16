
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import ast,unittest,tempfile
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.spatial.transform import Rotation
from scene_data import CV_TO_SR,sr_to_native,dense_route,read_lidar,ROOT
from point_semantics import labels_for_points,project,physical_lidar_labels,background_frame_labels,UNKNOWN
class DataTests(unittest.TestCase):
 def test_axis_roundtrip(self):
  p=np.array([[3.,2.,1.]])
  np.testing.assert_allclose(sr_to_native(p),[[-2.,-1.,3.]])
  np.testing.assert_allclose(sr_to_native(p)@CV_TO_SR.T,p)
 def test_native_route_uses_unmodified_densification(self):
  poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[0,1,2];dense=dense_route(poses);self.assertGreater(len(dense),3);np.testing.assert_allclose(dense[[0,-1],:3,3],[[0,0,0],[0,0,2]])
 def test_lidar_layout_and_invalid_record(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'one.bin';a=np.zeros((2,14),dtype='<f4');a[:,3:6]=[[3,2,1],[4,3,2]];a[:,10]=[0,1];a[:,13]=[0,4];a.tofile(p);self.assertEqual(read_lidar(p).shape,(2,14));p.write_bytes(b'bad');self.assertRaises(ValueError,read_lidar,p)
 def test_radial_and_tangential_distortion(self):
  # Camera point (1,.5,1), k1=.1,p1=.01,p2=.02, unit focal.
  points=np.array([[1.,-1.,-.5]])
  pixel,z,valid=project(points,np.eye(4),[10,10,10,10,.1,0,.01,.02,0],[100,100])
  # r2=1.25; x_d=1.125+.01+.065=1.2; y_d=.5625+.0175+.02=.6
  np.testing.assert_array_equal(pixel,[[22,16]]);self.assertTrue(valid[0])
 def test_occluded_lidar_remains_unknown(self):
  camera={'0':{'camera_waymo_to_ego':np.eye(4),'intrinsics_9':[10,10,10,10,0,0,0,0,0]}}
  mask=np.full((20,20),9,dtype=np.uint8);mask[10,10]=0
  labels,votes=labels_for_points(np.array([[5,0,0],[10,0,0],[3,100,0]]),camera,{0:mask},{'0':[20,20]},.5)
  np.testing.assert_array_equal(labels,[0,UNKNOWN,UNKNOWN]);self.assertEqual(int(votes[0]),1)
 def test_physical_lidar_contradictions_stay_unknown(self):
  labels,counts=physical_lidar_labels(np.array([0,0,10,8]),[True,False,False,False]);np.testing.assert_array_equal(labels,[0,UNKNOWN,UNKNOWN,8]);self.assertEqual(counts['road_without_geometric_ground_points'],1)
 def test_dense_gaussians_all_receive_observed_support(self):
  bg=np.c_[np.linspace(-.1,.1,10),np.zeros((10,2))];index,labels=background_frame_labels(bg,np.array([[0.,0.,0.]]),np.array([2],dtype=np.uint8),.25);self.assertEqual(len(index),10);self.assertTrue(np.all(labels==2))
 def test_cross_class_nearest_conflict_stays_unknown(self):
  index,labels=background_frame_labels(np.zeros((1,3)),np.array([[-.1,0,0],[.1,0,0]]),np.array([0,1],dtype=np.uint8),.25);self.assertEqual(len(index),0)
 def test_unobserved_background_gets_no_label(self):
  index,labels=background_frame_labels(np.array([[1.,0,0]]),np.zeros((1,3)),np.array([2],dtype=np.uint8),.25);self.assertEqual(len(index),0)
 def test_unknown_image_semantics_never_vote(self):
  c={'camera_waymo_to_ego':np.eye(4),'intrinsics_9':[10,10,10,10,0,0,0,0,0]};labels,votes=labels_for_points(np.array([[5,0,0]]),{'0':c},{0:np.full((20,20),255,dtype=np.uint8)},{'0':[20,20]},.5);self.assertEqual(int(labels[0]),UNKNOWN);self.assertEqual(int(votes[0]),0)
 def test_conflicting_cameras_remain_unknown(self):
  c={'camera_waymo_to_ego':np.eye(4),'intrinsics_9':[10,10,10,10,0,0,0,0,0]};labels,_=labels_for_points(np.array([[5,0,0]]),{'0':c,'1':c},{0:np.zeros((20,20),dtype=np.uint8),1:np.ones((20,20),dtype=np.uint8)},{'0':[20,20],'1':[20,20]},.5);self.assertEqual(int(labels[0]),UNKNOWN)
if __name__=='__main__':unittest.main(verbosity=2)
