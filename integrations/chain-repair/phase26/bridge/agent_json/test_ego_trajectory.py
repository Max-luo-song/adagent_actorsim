
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

import copy
import json
import unittest
pass  # Path is provided by the portable boundary above
import numpy as np
import test_agent_bridge as fixtures
from agent_jobs import run_job
from ego_trajectory import compile_trajectory,trajectory_metrics
import agent_bootstrap as bootstrap


class EgoTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.BridgeTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.bridge=self.fixture.bridge;self.root=self.fixture.root
        for i in range(81,90):
            pose=np.eye(4);pose[0,3]=(i-80)*.3
            np.savetxt(self.root/'inputs/ego_pose'/f'{i:03d}.txt',pose)
        self.fixture.load()
        self.path=dict(schema='ego_trajectory_v1',coordinate_frame='start_ego',start_frame=80,
            waypoints=[dict(time_s=i/10,forward_m=i*.3,left_m=i*.1,yaw_degrees=i*2) for i in range(3)])
    def args(self):
        return dict(expected_generation=self.bridge.session.generation,expected_revision=self.bridge.session.revision,trajectory=copy.deepcopy(self.path))
    def test_local_coordinates_use_source_axes_and_real_recorded_anchor(self):
        anchor=np.eye(4);anchor[:2,:2]=[[0,-1],[1,0]];anchor[:3,3]=[10,20,2]
        np.savetxt(self.root/'inputs/ego_pose/080.txt',anchor)
        result=compile_trajectory(self.path,self.bridge.session.spec)
        np.testing.assert_allclose(np.array(result['samples'][1]['ego_to_world'])[:3,3],[9.9,20.3,2])
        np.testing.assert_allclose(result['samples'][0]['ego_to_world'],anchor)
        self.assertEqual(result['samples'][2]['source_frame'],82)
        self.assertAlmostEqual(result['samples'][2]['timestamp_s'],8.2)
        self.assertFalse(result['control_executed']);self.assertFalse(result['collision_checked'])
    def test_source_world_poses_are_used_without_second_coordinate_conversion(self):
        path=copy.deepcopy(self.path);path['coordinate_frame']='source_world'
        matrices=[]
        for i in range(3):
            pose=np.eye(4);pose[:3,3]=[10+i,5,2];matrices.append(pose)
        path['waypoints']=[dict(time_s=i/10,ego_to_world=pose.tolist()) for i,pose in enumerate(matrices)]
        result=compile_trajectory(path,self.bridge.session.spec)
        for row,pose in zip(result['samples'],matrices):np.testing.assert_array_equal(row['ego_to_world'],pose)
    def test_entire_path_is_validated_before_any_render_or_edit(self):
        variants=[]
        for key,val in [('time_s',0),('time_s',.15),('time_s',float('nan')),('left_m',True)]:
            path=copy.deepcopy(self.path);path['waypoints'][1][key]=val;variants.append(path)
        path=copy.deepcopy(self.path);path['start_frame']=89;variants.append(path)
        path=copy.deepcopy(self.path);path['coordinate_frame']='hugsim_world';variants.append(path)
        path=copy.deepcopy(self.path);path['waypoints'][2]['time_s']=.4;variants.append(path)
        path=copy.deepcopy(self.path);path['waypoints'][0]['time_s']=.1;variants.append(path)
        for path in variants:
            args=self.args();args['trajectory']=path
            result=self.fixture.send('preview_ego_trajectory',args)
            self.assertEqual(result['status'],'error')
        self.assertEqual(self.bridge.render_count,0);self.assertFalse(self.bridge.session.editor.started)
        self.assertEqual(self.bridge.session.revision,0)
    def test_validate_only_and_version_guards_allocate_no_renders(self):
        self.assertEqual(self.fixture.send('validate_ego_trajectory',self.args())['status'],'ok')
        args=self.args();args['expected_generation']='old'
        self.fixture.error(self.fixture.send('preview_ego_trajectory',args),'stale_generation')
        args=self.args();args['expected_revision']=1
        self.fixture.error(self.fixture.send('preview_ego_trajectory',args),'stale_revision')
        self.assertEqual(self.bridge.render_count,0)
    def test_real_video_packet_poses_edits_and_idempotent_replay(self):
        self.fixture.send('edit',self.fixture.args(self.fixture.insert()))
        version=self.bridge.session.generation;revision=self.bridge.session.revision
        req=self.fixture.req('preview_ego_trajectory',self.args(),'path_once')
        first=self.bridge.handle(req);self.assertEqual(first['status'],'ok',first)
        replay=self.bridge.handle(req);self.assertTrue(replay['replayed'])
        result=first['result'];self.assertTrue(Path(result['video']).is_file());self.assertEqual(result['rendered_packets'],6)
        packets=_hugsim_expand(json.loads(Path(result['manifest']).read_text()))
        self.assertTrue(packets['completed']);self.assertEqual(len(packets['frames']),3)
        for index,pair in enumerate(packets['frames']):
            self.assertEqual(pair['new']['timestamp_s'],pair['baseline']['timestamp_s'])
            np.testing.assert_allclose(pair['new']['ego_to_world'],result['trajectory']['samples'][index]['ego_to_world'])
            for camera in pair['new']['cameras'].values():
                np.testing.assert_allclose(camera['camera_to_world'],pair['new']['ego_to_world'])
                self.assertTrue(Path(camera['rgb']).is_file());self.assertTrue(Path(camera['depth']).is_file())
        self.assertEqual(self.bridge.render_count,6);self.assertIsNone(self.bridge.session.backend._last)
        self.assertEqual(self.bridge.session.generation,version);self.assertEqual(self.bridge.session.revision,revision)
        self.assertEqual(len(self.bridge.session.objects),1);self.assertFalse(self.bridge.session.editor.started)
        self.assertEqual(self.fixture.send('edit',self.fixture.args(self.fixture.translate()))['status'],'ok')
    def test_after_observe_requires_reset(self):
        self.fixture.send('observe',self.fixture.observation())
        self.fixture.error(self.fixture.send('preview_ego_trajectory',self.args()),'preview_requires_pre_run')
        self.assertEqual(self.bridge.render_count,1)
    def test_optional_single_path_and_job_acceptance_include_every_packet(self):
        args=self.args();args['compare_recorded']=False
        job=dict(schema='agent_job_v1',steps=[dict(request=self.fixture.req('preview_ego_trajectory',args))],
            verification=dict(minimum_coverage={'ego_trajectory_previews':1}))
        acceptance=run_job(job,self.bridge,self.root/'trajectory_job')
        self.assertTrue(acceptance['passed']);self.assertEqual(acceptance['rendered_packets'],3)
        self.assertEqual(acceptance['actual_renders'],3);self.assertFalse(acceptance['gpu_render_executed'])
        self.assertEqual(acceptance['coverage']['ego_trajectory_previews'],1)
        self.assertTrue(acceptance['checks']['ego_trajectory_previews_verified'])
    def test_public_five_step_example_renders_all_21_samples(self):
        for i in range(90,101):
            pose=np.eye(4);pose[0,3]=(i-80)*.3
            np.savetxt(self.root/'inputs/ego_pose'/f'{i:03d}.txt',pose)
        from scene_edit_session import saf
        reader=saf.SceneSpec.read.side_effect
        def extended(path):
            spec=reader(path);spec.frames=110;return spec
        saf.SceneSpec.read.side_effect=extended
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/ego_trajectory.json').read_text()))
        job['steps'][0]['request']['args'].update(expected_generation=self.bridge.session.generation,expected_revision=0)
        acceptance=run_job(job,self.bridge,self.root/'public_ego_job')
        self.assertTrue(acceptance['passed']);self.assertEqual(acceptance['actual_renders'],42)
        self.assertEqual(acceptance['completed_steps'],5)
        self.assertEqual(acceptance['coverage']['native_edits'],1)
        self.assertEqual(acceptance['coverage']['ego_trajectory_validations'],1)
    def test_large_public_arc_has_four_metre_lateral_motion_and_tangent_yaw(self):
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/ego_trajectory_large.json').read_text()))
        path=job['steps'][1]['request']['args']['trajectory']
        self.assertEqual(path,job['steps'][2]['request']['args']['trajectory'])
        self.bridge.session.spec.frames=150
        for i in range(90,121):np.savetxt(self.root/'inputs/ego_pose'/f'{i:03d}.txt',np.eye(4))
        result=compile_trajectory(path,self.bridge.session.spec)
        poses=[np.asarray(row['ego_to_world']) for row in result['samples']]
        self.assertEqual(result['sample_count'],41);self.assertEqual(result['duration_s'],4)
        np.testing.assert_allclose((np.linalg.inv(poses[0])@poses[-1])[:2,3],[12,4],atol=1e-6)
        yaw=np.degrees(np.arctan2(poses[-1][1,0],poses[-1][0,0]))
        self.assertAlmostEqual(yaw,36.869898,places=5)
        # Arc positions and heading follow the same circle, avoiding lateral sliding.
        for row in path['waypoints']:
            angle=np.radians(row['yaw_degrees'])
            np.testing.assert_allclose([row['forward_m'],row['left_m']],[20*np.sin(angle),20*(1-np.cos(angle))],atol=1e-6)
    def test_video_metrics_use_actual_packet_positions_and_relative_heading(self):
        base=np.eye(4);base[:2,:2]=[[0,-1],[1,0]];base[:3,3]=[10,20,0]
        delta=np.eye(4);a=np.radians(30);delta[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]];delta[:2,3]=[12,4]
        packet=lambda pose:dict(ego_to_world=pose.tolist(),timestamp_s=8)
        metrics=trajectory_metrics([dict(baseline=packet(base),new=packet(base@delta))])[0]
        self.assertAlmostEqual(metrics['new_forward_m'],12);self.assertAlmostEqual(metrics['new_left_m'],4)
        self.assertAlmostEqual(metrics['relative_yaw_degrees'],30)
        self.assertAlmostEqual(metrics['separation_m'],np.hypot(12,4))

    def test_nonrigid_world_pose_and_bad_compare_flag_rejected(self):
        args=self.args();args['compare_recorded']='yes'
        self.fixture.error(self.fixture.send('preview_ego_trajectory',args),'invalid_trajectory')
        args=self.args();path=args['trajectory'];path['coordinate_frame']='source_world'
        pose=np.eye(4);pose[0,0]=2
        path['waypoints']=[dict(time_s=i/10,ego_to_world=pose.tolist()) for i in range(3)]
        self.fixture.error(self.fixture.send('validate_ego_trajectory',args),'invalid_trajectory')
        self.assertEqual(self.bridge.render_count,0)

    def test_failure_keeps_incomplete_manifest_and_no_successful_video(self):
        backend=self.bridge.session.backend;original=backend.render
        def render(pose,time):
            if time>8:raise RuntimeError('explicit second-frame raster failure')
            return original(pose,time)
        backend.render=render
        result=self.fixture.send('preview_ego_trajectory',self.args(),'partial_path')
        self.fixture.error(result,'backend_error')
        manifest=_hugsim_expand(json.loads((self.bridge.output/'trajectories/partial_path/packets.json').read_text()))
        self.assertFalse(manifest['completed']);self.assertEqual(len(manifest['frames']),1)
        self.assertFalse((self.bridge.output/'trajectories/partial_path/ego_trajectory_comparison.mp4').exists())
        self.assertFalse(self.bridge.session.editor.started);self.assertIsNone(backend._last)

if __name__=='__main__':unittest.main()
