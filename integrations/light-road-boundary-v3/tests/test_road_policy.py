
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

import copy,sys,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'light-cutin-diagnostic')]
from common import configure
configure()
from road_reference import RoadReference
from runtime_configuration import load_vehicle_config
from live_env import Motion
from tracker import Tracker
from drivable_area import SCHEMA,AXES

def fixture(motion=None):
    mount=np.eye(4);mount[1,3]=.985793
    theta=motion.yaw if motion else 0.;right=motion.x if motion else 0.
    a=-np.tan(theta);b=(1.856-right-np.sin(theta)*mount[1,3])/np.cos(theta)
    packet=dict(schema='observed_white_road_edge_v1',status='fresh',timestamp=0.,ground_truth_used=False,
        source='current_camera_rgb_ground_projection',edges=[dict(side='right',a=a,b=b,
        valid_y_range=[-5.,50.],observed_y_range=[6.,45.],uncertainty_m=.15)])
    x=-50+(np.arange(200)+.5)*.5;mask=np.repeat((x<2.5)[None],200,axis=0)
    arrays=dict(drivable_mask=mask,drivable_raw_score=mask.astype(float),drivable_bounds_xy=np.array([-50.,-50.,50.,50.]),
        drivable_schema=np.asarray(SCHEMA),drivable_axes=np.asarray(AXES))
    info=dict(model_lidar_frame=dict(virtual_to_ego=mount))
    raw=np.c_[[.004,.054,.059,.097,.164,.454],np.arange(1,7)*2-mount[1,3]]
    return packet,arrays,info,raw

class RoadPolicyTests(unittest.TestCase):
    def test_raw_bias_is_reduced_without_changing_neural_arrays_or_tracker(self):
        packet,arrays,info,raw=fixture();saved=raw.copy();mask=arrays['drivable_mask'].copy()
        v=load_vehicle_config();road=RoadReference(v);m=Motion(v.dt,v.wheelbase_m,4.)
        with patch('road_reference.detect',return_value=packet):ref,audit=road.process(raw,arrays,{},info,m)
        np.testing.assert_array_equal(raw,saved);np.testing.assert_array_equal(arrays['drivable_mask'],mask)
        np.testing.assert_array_equal(ref[:,1],raw[:,1]);self.assertLess(abs(ref[-1,0]),.05)
        self.assertTrue(audit['reference_certificate']['passed'])
        nominal,_=Tracker(v).control(ref,info,m);action,gate=road.admit(m,nominal)
        self.assertIsNotNone(action);self.assertTrue(gate['certificate']['passed'])
    def test_whole_body_is_rejected_even_when_center_is_inside_white_line(self):
        packet,arrays,info,raw=fixture();v=load_vehicle_config();road=RoadReference(v)
        with patch('road_reference.detect',return_value=packet):road.observed({},info,arrays)
        cert=road.certificate(np.array([[1.,5.,0.,4.,0.]]))
        self.assertFalse(cert['passed']);self.assertLess(cert['white_edge_swept_clearance_m'],0)
    def test_missing_boundary_never_admits_unverified_acceleration(self):
        packet,arrays,info,raw=fixture();packet.update(status='unavailable',edges=[])
        v=load_vehicle_config();road=RoadReference(v);m=Motion(v.dt,v.wheelbase_m,4.)
        with patch('road_reference.detect',return_value=packet):road.process(raw,arrays,{},info,m)
        action,gate=road.admit(m,dict(acc_mps2=1.,steer_rate_radps=0.))
        self.assertIsNone(action);self.assertFalse(gate['brake_rejection']['current_edge_coverage_complete'])
    def test_stopping_envelope_rejects_future_crossing_not_just_current_body(self):
        packet,arrays,info,raw=fixture();v=load_vehicle_config();road=RoadReference(v);m=Motion(v.dt,v.wheelbase_m,8.);m.steer=.5
        with patch('road_reference.detect',return_value=packet):road.process(raw,arrays,{},info,m)
        action,gate=road.admit(m,dict(acc_mps2=1.,steer_rate_radps=.1))
        self.assertIsNone(action);self.assertLess(gate['brake_rejection']['white_edge_swept_clearance_m'],0.)
    def test_unobserved_longitudinal_band_is_not_free_space(self):
        packet,arrays,info,raw=fixture();v=load_vehicle_config();road=RoadReference(v)
        with patch('road_reference.detect',return_value=packet):road.observed({},info,arrays)
        cert=road.certificate(np.array([[0.,60.,0.,4.,0.]]))
        self.assertFalse(cert['current_edge_coverage_complete']);self.assertFalse(cert['passed'])
    def test_model_road_cost_penalizes_departure_but_is_not_white_line_truth(self):
        packet,arrays,info,raw=fixture();v=load_vehicle_config();road=RoadReference(v)
        with patch('road_reference.detect',return_value=packet):road.observed({},info,arrays)
        inside=road.soft.distance_cost(np.array([[0.,5.,0.,4.,0.]]),1)
        outside=road.soft.distance_cost(np.array([[5.,5.,0.,4.,0.]]),1)
        self.assertGreater(outside,inside)
    def test_four_second_native_motion_loop_does_not_accumulate_raw_right_bias(self):
        v=load_vehicle_config();on=Motion(v.dt,v.wheelbase_m,4.);off=copy.copy(on)
        road=RoadReference(v);tracker_on=Tracker(v);tracker_off=Tracker(v)
        for _ in range(40):
            packet,arrays,info,raw=fixture(on)
            with patch('road_reference.detect',return_value=packet):adopted,audit=road.process(raw,arrays,{},info,on)
            nominal,_=tracker_on.control(adopted,info,on);action,gate=road.admit(on,nominal)
            self.assertIsNotNone(action,gate)
            on.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
            original,_=tracker_off.control(raw,info,off)
            off.advance(dict(acc=original['acc_mps2'],steer_rate=original['steer_rate_radps']))
        self.assertLess(abs(on.x),.15);self.assertGreater(off.x,.3)
        self.assertLess(abs(on.yaw),np.radians(3))
        print('SYNTHETIC ROAD LOOP ONLY:',dict(raw_right_m=off.x,corrected_right_m=on.x,corrected_yaw_deg=np.degrees(on.yaw)))

if __name__=='__main__':unittest.main()
