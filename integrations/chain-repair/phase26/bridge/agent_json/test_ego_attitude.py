
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

import copy,json,math
pass  # Path is provided by the portable boundary above
import unittest
import numpy as np
import test_agent_bridge as fixtures
import agent_bootstrap as bootstrap
from runtime_configuration import load_vehicle_config
from scene_edit_session import saf
from ego_trajectory import compile_trajectory
from ego_attitude import gravity_up,corrected_pose,horizon_metrics
from agent_jobs import run_job


def rpy(roll,pitch,yaw):
    r,p,y=np.radians([roll,pitch,yaw]);cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])


class AttitudeTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.BridgeTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.bridge=self.f.bridge
        self.vehicle=load_vehicle_config();reader=saf.SceneSpec.read.side_effect
        def extended(path):
            spec=reader(path);spec.frames=150;spec.vehicle=self.vehicle;return spec
        saf.SceneSpec.read.side_effect=extended
        origin=np.eye(4);origin[:3,:3]=rpy(4.17,1.8,-6.03);self.origin=origin
        np.savetxt(self.f.root/'inputs/ego_pose/000.txt',origin)
        for i in range(80,121):
            pose=np.eye(4);pose[:3,:3]=rpy(6.55-(i-80)*.04,.8,9+(i-80)*.03);pose[:3,3]=[i*.3,1,8]
            np.savetxt(self.f.root/'inputs/ego_pose'/f'{i:03d}.txt',pose)
        self.f.load();self.spec=self.bridge.session.spec
        backend=self.bridge.session.backend;old_render=backend.render
        cv=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        self.mounts={}
        for name,yaw in [('CAM_FRONT',0),('CAM_FRONT_LEFT',60),('CAM_FRONT_RIGHT',-60)]:
            mount=np.eye(4);mount[:3,:3]=rpy(1,0,yaw)@cv;mount[:3,3]=[1.5,0,2.1];self.mounts[name]=mount
        def render(pose,time):
            raw=old_render(pose,time)
            for name,cam in raw['cameras'].items():cam['camera_to_world']=pose@self.mounts[name]
            return raw
        backend.render=render
        self.job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/ego_horizon_comparison.json').read_text()))
        self.path=self.job['steps'][2]['request']['args']['trajectory']
    def args(self,path):return dict(expected_generation=self.bridge.session.generation,expected_revision=0,trajectory=path,compare_recorded=False)
    def test_gravity_is_transformed_from_raw_world_not_scene_z(self):
        up=gravity_up(self.spec)
        np.testing.assert_allclose(up,self.origin[:3,:3].T@[0,0,1],atol=1e-12)
        self.assertGreater(np.linalg.norm(up-[0,0,1]),.05)
    def test_leveling_preserves_positions_motion_and_projected_heading(self):
        old=copy.deepcopy(self.path);old['attitude_policy']='initial_ego'
        before=compile_trajectory(old,self.spec);after=compile_trajectory(self.path,self.spec);up=gravity_up(self.spec)
        for a,b in zip(before['samples'],after['samples']):
            x=np.array(a['ego_to_world']);y=np.array(b['ego_to_world'])
            np.testing.assert_array_equal(x[:3,3],y[:3,3]);np.testing.assert_allclose(y[:3,2],up,atol=1e-12)
            np.testing.assert_array_equal(a['motion_ego_to_world'],b['motion_ego_to_world'])
            heading=x[:3,0]-up*(up@x[:3,0]);heading/=np.linalg.norm(heading)
            np.testing.assert_allclose(heading,y[:3,0],atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(y[:3,:3]),1)
    def test_recorded_up_preserves_recorded_bank_instead_of_zeroing_it(self):
        path=copy.deepcopy(self.path);path['attitude_policy']='recorded_up'
        c=compile_trajectory(path,self.spec)
        for row in c['samples']:
            np.testing.assert_allclose(np.array(row['ego_to_world'])[:3,2],np.array(row['recorded_ego_to_world'])[:3,2],atol=1e-12)
    def test_default_legacy_pose_remains_exactly_initial_ego_composition(self):
        path=copy.deepcopy(self.path);path.pop('attitude_policy')
        c=compile_trajectory(path,self.spec)
        for row in c['samples']:np.testing.assert_array_equal(row['ego_to_world'],row['motion_ego_to_world'])
        self.assertEqual(c['attitude_policy'],'initial_ego')
    def test_invalid_policy_and_parallel_heading_are_rejected(self):
        path=copy.deepcopy(self.path);path['attitude_policy']='rotate_image'
        with self.assertRaises(ValueError):compile_trajectory(path,self.spec)
        with self.assertRaises(ValueError):corrected_pose(np.eye(4),'gravity_level',[1,0,0])
    def test_mount_roll_remains_calibrated_and_projection_is_yaw_invariant(self):
        c=compile_trajectory(self.path,self.spec);angles=[]
        for row in c['samples']:
            pose=np.asarray(row['ego_to_world']);cams={name:dict(camera_to_world=(pose@mount).tolist(),K=np.eye(3).tolist()) for name,mount in self.mounts.items()}
            metrics=horizon_metrics(dict(ego_to_world=pose.tolist(),cameras=cams),c['gravity_up_source'])
            angles.append(metrics['cameras']['CAM_FRONT']['projected_horizon_tilt_degrees'])
            self.assertLess(metrics['body_up_tilt_degrees'],1e-4)
        np.testing.assert_allclose(angles,[-1.]*41,atol=1e-8)
    def test_preview_uses_corrected_pose_for_all_camera_geometry(self):
        path=copy.deepcopy(self.path);path['waypoints']=path['waypoints'][:3]
        response=self.f.send('preview_ego_trajectory',self.args(path));self.assertEqual(response['status'],'ok',response)
        manifest=_hugsim_expand(json.loads(Path(response['result']['manifest']).read_text()))
        for pair in manifest['frames']:
            packet=pair['new'];pose=np.asarray(packet['ego_to_world'])
            for name,cam in packet['cameras'].items():
                np.testing.assert_allclose(cam['camera_to_world'],pose@self.mounts[name],atol=1e-12)
                np.testing.assert_allclose(cam['camera_to_ego'],self.mounts[name],atol=1e-12)
            self.assertLess(packet['attitude']['body_up_tilt_degrees'],1e-4)
        self.assertFalse(self.bridge.session.editor.started)
    def test_complete_horizon_comparison_outputs_real_video_and_audit(self):
        job=copy.deepcopy(self.job);job['steps'][0]['request']['args'].update(expected_generation=self.bridge.session.generation,expected_revision=0)
        result=run_job(job,self.bridge,self.f.root/'horizon_job')
        self.assertTrue(result['passed']);self.assertEqual(result['actual_renders'],82)
        self.assertTrue(result['checks']['ego_attitude_comparison_passed'])
        report=_hugsim_expand(json.loads((self.f.root/'horizon_job/attitude_comparison/comparison.json').read_text()))
        self.assertTrue(report['passed']);self.assertTrue(Path(report['video']).is_file());self.assertFalse(report['image_rotation'])
        self.assertGreater(report['metrics'][-1]['before']['body_up_tilt_degrees'],4)
        self.assertLess(report['metrics'][-1]['after']['body_up_tilt_degrees'],1e-4)
    def test_actual_motion_driving_supports_level_attitude_without_warping_motion(self):
        path=copy.deepcopy(self.path);path['waypoints']=path['waypoints'][:8]
        args=self.args(path);args.pop('compare_recorded')
        result=self.f.send('execute_ego_trajectory',args);self.assertEqual(result['status'],'ok',result)
        self.assertTrue(all(result['result']['audit'].values()));self.assertEqual(self.bridge.control_count,7)
        data=_hugsim_expand(json.loads(Path(result['result']['manifest']).read_text()))
        self.assertTrue(data['tracking']['passed'])
        self.assertTrue(all(row['body_up_tilt_degrees']<1e-4 for row in data['attitude']))
        self.assertEqual(self.bridge.driving_state['attitude_policy'],'gravity_level')
        pose=np.asarray(self.bridge.driving_state['ego_to_world']);nominal=np.asarray(self.bridge.driving_state['motion_ego_to_world'])
        np.testing.assert_array_equal(pose[:3,3],nominal[:3,3]);self.assertGreater(np.max(np.abs(pose[:3,:3]-nominal[:3,:3])),.05)

if __name__=='__main__':unittest.main()
