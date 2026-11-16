import unittest
import numpy as np
from uniad_bridge import to_uniad,B,T,ORDER
from scene_api import BASE


class UniADBridgeTests(unittest.TestCase):
    def packet(self):
        ego=np.eye(4);a=.4;ego[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]];ego[:3,3]=[2,3,.5]
        cameras={}
        for name in ORDER:
            ext=np.eye(4);ext[:3,3]=[1,.2,1.4]
            cameras[name]=dict(rgb=np.zeros((4,6,3),np.uint8),depth=np.ones((4,6)),
                K=np.array([[8,0,3],[0,8,2],[0,0,1.]]),camera_to_ego=ext,camera_to_world=ego@ext)
        return dict(schema='unified_observation_v1',camera_order=list(ORDER),ego_to_world=ego,cameras=cameras,
            timestamp_s=.2,source_frame=2,training=False,formal_evaluation_admitted=False,
            edit_trust=dict(edited=True,status='diagnostic_unverified'),managed_objects={'secret_asset_truth':{}})
    def test_virtual_mount_and_global_sensor_projection_are_consistent(self):
        packet=self.packet();mount=np.eye(4);mount[:3,3]=[.4,-.7,1.8]
        _,info=to_uniad(packet,mount,speed_mps=2.,steering_rad=0.,command=2,scene_token='arm')
        virtual=np.eye(4);virtual[:3,:3]=info['explicit_l2g_r_mat'];virtual[:3,3]=info['explicit_l2g_t']
        np.testing.assert_allclose(virtual,B@(T@packet['ego_to_world']@T.T)@B.T@mount)
        for name in ORDER:
            # The same source-world point must project identically via both paths.
            world=np.array([8.,4.,1.,1.]);local=np.linalg.inv(virtual)@B@T@world
            expected=np.linalg.inv(packet['cameras'][name]['camera_to_world'])@world
            actual=info['cam_params'][name]['l2c']@local
            np.testing.assert_allclose(actual,expected,atol=1e-10)
    def test_no_target_geometry_in_model_input_and_camera_order_fixed(self):
        obs,info=to_uniad(self.packet(),np.eye(4),speed_mps=2.,steering_rad=0.,command=2,scene_token='arm')
        self.assertEqual(tuple(obs['rgb']),ORDER)
        self.assertFalse(any(key in info for key in ('managed_objects','asset_to_world','asset_wlh','target_box')))
        self.assertFalse(info['control_actions_executed']);self.assertFalse(info['formal_evaluation_admitted'])
    def test_inconsistent_calibration_rejected(self):
        packet=self.packet();packet['cameras']['CAM_FRONT']['camera_to_world'][0,3]+=1
        with self.assertRaises(ValueError):
            to_uniad(packet,np.eye(4),speed_mps=2.,steering_rad=0.,command=2,scene_token='arm')

if __name__=='__main__':unittest.main()
