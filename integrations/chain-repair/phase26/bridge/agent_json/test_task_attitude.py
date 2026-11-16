"""Regression for the ordinary camera branch omitted by task unification."""

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

import copy,json,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
import test_ego_attitude as fixtures
from ego_attitude import gravity_up
from ego_trajectory import recorded_pose
from scene_tasks import compile_task
from scene_agent_client import SceneAgentClient


class TaskAttitudeTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.AttitudeTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.b=self.f.bridge;self.client=SceneAgentClient(self.b.handle)
        cv=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        for name,yaw in [('CAM_FRONT',0),('CAM_FRONT_LEFT',60),('CAM_FRONT_RIGHT',-60)]:self.f.mounts[name][:3,:3]=fixtures.rpy(.15,0,yaw)@cv
    def preview(self,policy=None):
        args=dict(expected_generation=self.b.session.generation,expected_revision=0,source_frame=100)
        if policy is not None:args['attitude_policy']=policy
        result=self.f.f.send('preview',args);self.assertEqual(result['status'],'ok',result);return result['result']
    def task(self,**changes):
        value=dict(schema='scene_task_v1',task_id='level_views',scene_id='552',previews=[dict(kind='camera',args=dict(source_frame=100))],verification_frames=[100],reset_after=True)
        value.update(changes);return value
    def test_ordinary_preview_can_level_all_three_cameras_without_changing_mounts(self):
        raw=self.preview();level=self.preview('gravity_level')
        self.assertGreater(raw['attitude']['body_up_tilt_degrees'],1);self.assertLess(level['attitude']['body_up_tilt_degrees'],1e-4)
        np.testing.assert_array_equal(np.array(raw['ego_to_world'])[:3,3],np.array(level['ego_to_world'])[:3,3])
        self.assertEqual(raw['timestamp_s'],level['timestamp_s']);self.assertTrue(level['attitude']['gravity_alignment_passed'])
        for name in level['cameras']:
            a,b=raw['cameras'][name],level['cameras'][name]
            np.testing.assert_array_equal(a['K'],b['K']);np.testing.assert_allclose(a['camera_to_ego'],b['camera_to_ego'],atol=1e-12)
            np.testing.assert_allclose(b['camera_to_world'],np.asarray(level['ego_to_world'])@self.f.mounts[name],atol=1e-12)
        self.assertFalse(self.b.session.editor.started)
    def test_small_followup_job_checks_pose_and_mounts_without_same_pose_RGB_rule(self):
        import agent_bootstrap as bootstrap
        from agent_jobs import run_job
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/camera_level_followup.json').read_text()))
        # Keep the fixture's calibrated camera rig while starting a clean generation.
        job['steps'][0]['request']['action']='reset'
        job['steps'][0]['request']['args']=dict(expected_generation=self.b.session.generation,expected_revision=0)
        job['verification']['minimum_coverage'].update(scene_loads=0,resets=1)
        result=run_job(job,self.b,self.f.f.root/'two_view_followup')
        self.assertTrue(result['passed']);self.assertEqual(result['actual_renders'],2)
        report=_hugsim_expand(json.loads((self.f.f.root/'two_view_followup/camera_level_comparison.json').read_text()))
        self.assertTrue(report['passed']);self.assertFalse(report['gpu_render_executed']);self.assertTrue(report['camera_mount_and_K_preserved'])
    def test_legacy_ordinary_preview_retains_recorded_pose_by_default(self):
        raw=self.preview();np.testing.assert_array_equal(raw['ego_to_world'],recorded_pose(self.b.session.spec,100))
        self.assertEqual(raw['attitude_policy'],'initial_ego')
    def test_task_injects_one_policy_into_every_branch(self):
        path=copy.deepcopy(self.f.path);path['waypoints']=path['waypoints'][:3]
        task=self.task(previews=[dict(kind='camera',args=dict(source_frame=100)),dict(kind='ego_trajectory',args=dict(trajectory=path)),dict(kind='actor_trajectory',args=dict(object_id='native:SMPLNodes:0',start_frame=80,end_frame=82,camera_source_frame=80))],ego_execution=dict(driver='reference_tracking',trajectory=path))
        job=compile_task(task)
        for step in job['steps']:
            req=step['request'];args=req['args']
            if req['action'] in ('preview','preview_actor_trajectory'):self.assertEqual(args['attitude_policy'],'gravity_level')
            if req['action'] in ('preview_ego_trajectory','validate_ego_execution','execute_ego_trajectory'):self.assertEqual(args['trajectory']['attitude_policy'],'gravity_level')
            if req['action']=='preview_ego_trajectory':self.assertTrue(args['level_recorded_baseline'])
    def test_task_corrects_ordinary_and_ego_baseline_and_verifies_reset(self):
        path=copy.deepcopy(self.f.path);path['waypoints']=path['waypoints'][:3]
        r=self.client.run_task(self.task(previews=[dict(kind='camera',args=dict(source_frame=100)),dict(kind='ego_trajectory',args=dict(trajectory=path,compare_recorded=True))]));self.assertEqual(r['status'],'ok',r)
        result=r['result'];self.assertTrue(result['acceptance']['checks']['task_gravity_alignment'])
        audit=result['metrics']['camera_horizon'];self.assertEqual(audit['observations'],9);self.assertLess(audit['max_camera_horizon_tilt_degrees'],.5)
        self.assertTrue(all(row['aligned'] for row in audit['metrics']));self.assertEqual(result['context']['phase'],'pre_run')
    def test_task_detects_remaining_bad_camera_mount_despite_level_ego(self):
        cv=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        self.f.mounts['CAM_FRONT_RIGHT'][:3,:3]=fixtures.rpy(5,0,-60)@cv
        r=self.client.run_task(self.task());self.assertEqual(r['status'],'error',r)
        self.assertIn('task_gravity_alignment',r['error']['details']['error']['failed_checks'])
        self.assertFalse(r['error']['details']['acceptance']['passed'])
    def test_explicit_raw_task_can_be_used_for_diagnostic_comparison(self):
        r=self.client.run_task(self.task(attitude_policy='initial_ego'));self.assertEqual(r['status'],'ok',r)
        audit=r['result']['metrics']['camera_horizon'];self.assertFalse(audit['applicable'])
        self.assertGreater(audit['max_camera_horizon_tilt_degrees'],1)
    def test_conflicting_preview_policy_rejected_before_reset(self):
        before=self.b.context()
        for task in [self.task(previews=[dict(kind='camera',args=dict(source_frame=100,attitude_policy='initial_ego'))]),self.task(attitude_policy='rotate_png')]:
            r=self.client.run_task(task);self.assertEqual(r['error']['code'],'invalid_scene_task');self.assertEqual(before,self.b.context())

if __name__=='__main__':unittest.main()
