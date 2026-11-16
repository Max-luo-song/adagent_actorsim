
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
import math
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import test_agent_bridge as fixtures
import agent_bootstrap as bootstrap
from agent_jobs import run_job
from ego_trajectory import compile_trajectory
from ego_execution import execution_plan,TrackingDriver,motion_state,motion_pose,clip_action
from runtime_configuration import load_vehicle_config
from scene_edit_session import saf


class ZeroSolver:
    def solve(self,state,reference):return [SimpleNamespace(input_trajectory=np.zeros((len(reference)-1,2)))]


class EgoExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.BridgeTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.bridge=self.fixture.bridge;self.root=self.fixture.root;self.vehicle=load_vehicle_config()
        reader=saf.SceneSpec.read.side_effect
        def extended(path):
            spec=reader(path);spec.frames=150;spec.vehicle=self.vehicle;return spec
        saf.SceneSpec.read.side_effect=extended
        for i in range(81,121):
            pose=np.eye(4);pose[0,3]=(i-80)*.3;np.savetxt(self.root/'inputs/ego_pose'/f'{i:03d}.txt',pose)
        self.fixture.load()
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/ego_driving.json').read_text()))
        self.path=job['steps'][1]['request']['args']['trajectory']
    def args(self,path=None):
        return dict(expected_generation=self.bridge.session.generation,expected_revision=self.bridge.session.revision,
            trajectory=copy.deepcopy(self.path if path is None else path))
    def short_path(self,count=5):
        path=copy.deepcopy(self.path);path['waypoints']=path['waypoints'][:count];return path
    def driver(self,path=None,**args):
        compiled=compile_trajectory(self.path if path is None else path,self.bridge.session.spec)
        return TrackingDriver(execution_plan(compiled,self.vehicle,args),self.vehicle),compiled
    def test_actual_ilqr_and_motion_track_large_left_arc(self):
        driver,compiled=self.driver();errors=[]
        for i in range(40):
            row=driver.advance(i);actual=motion_pose(driver.plan['anchor'],driver.motion)
            errors.append(np.linalg.norm(actual[:3,3]-np.asarray(compiled['samples'][i+1]['ego_to_world'])[:3,3]))
            self.assertLessEqual(abs(row['action']['steer_rate_radps']),math.radians(15)+1e-10)
            self.assertLessEqual(abs(row['action']['acc_mps2']),2+1e-10)
        self.assertLess(max(errors),.3);self.assertLess(errors[-1],.15)
        self.assertLess(driver.motion.x,-3.5);self.assertLess(driver.motion.yaw,-.5)
        self.assertEqual(row['supplied_reference_samples'],2)
        self.assertEqual(row['solver_only_terminal_extension'],4)
    def test_target_waypoints_never_overwrite_motion_state(self):
        driver,compiled=self.driver();driver.solver=ZeroSolver()
        for i in range(40):driver.advance(i)
        self.assertEqual(driver.motion.x,0);self.assertEqual(driver.motion.yaw,0)
        final=motion_pose(driver.plan['anchor'],driver.motion)
        self.assertGreater(np.linalg.norm(final[:3,3]-np.asarray(compiled['samples'][-1]['ego_to_world'])[:3,3]),4)
    def test_shared_action_limits_respect_speed_and_steering_boundaries(self):
        driver,_=self.driver();m=driver.motion;m.speed=self.vehicle.max_speed_mps;m.steer=math.radians(80)
        action=clip_action(100,100,m,self.vehicle)
        self.assertEqual(action['acc_mps2'],0);self.assertEqual(action['steer_rate_radps'],0)
        m.speed=0;m.steer=-math.radians(80)
        action=clip_action(-100,-100,m,self.vehicle)
        self.assertEqual(action['acc_mps2'],0);self.assertEqual(action['steer_rate_radps'],0)
        with self.assertRaises(ValueError):clip_action(float('nan'),0,m,self.vehicle)
    def test_invalid_execution_is_rejected_before_render_or_control(self):
        for extra in [dict(initial_speed_mps=-1),dict(initial_speed_mps=True),dict(initial_steer_rad=2),dict(max_tracking_error_m=0)]:
            args=self.args();args.update(extra)
            self.fixture.error(self.fixture.send('execute_ego_trajectory',args),'invalid_execution')
        self.assertEqual(self.bridge.render_count,0);self.assertEqual(self.bridge.control_count,0)
        self.assertFalse(self.bridge.session.editor.started)
    def test_nonplanar_world_trajectory_is_not_silently_flattened(self):
        path=copy.deepcopy(self.path);path['coordinate_frame']='source_world'
        path['waypoints']=[dict(time_s=i/10,ego_to_world=(np.eye(4)+np.array([[0,0,0,i*.3],[0,0,0,0],[0,0,0,i*.1],[0,0,0,0]])).tolist()) for i in range(3)]
        self.fixture.error(self.fixture.send('validate_ego_execution',self.args(path)),'invalid_execution')
        self.assertEqual(self.bridge.control_count,0);self.assertEqual(self.bridge.render_count,0)
    def test_validation_and_stale_version_never_seal_or_advance(self):
        response=self.fixture.send('validate_ego_execution',self.args());self.assertEqual(response['status'],'ok')
        self.assertFalse(response['control_executed']);self.assertEqual(response['controls_executed'],0)
        args=self.args();args['expected_generation']='old'
        self.fixture.error(self.fixture.send('execute_ego_trajectory',args),'stale_generation')
        args=self.args();args['expected_revision']=1
        self.fixture.error(self.fixture.send('execute_ego_trajectory',args),'stale_revision')
        self.assertFalse(self.bridge.session.editor.started)
    def test_execution_replay_reset_and_edit_guard_use_real_scene_session(self):
        self.fixture.send('edit',self.fixture.args(self.fixture.insert()))
        self.fixture.send('edit',self.fixture.args(self.fixture.translate()))
        req=self.fixture.req('execute_ego_trajectory',self.args(self.short_path()),'drive_once')
        response=self.bridge.handle(req);self.assertEqual(response['status'],'ok',response)
        self.assertTrue(response['control_executed']);self.assertEqual(response['controls_executed'],4)
        self.assertEqual(self.bridge.render_count,5);self.assertEqual(self.bridge.control_count,4)
        self.assertTrue(self.bridge.handle(req)['replayed']);self.assertEqual(self.bridge.control_count,4)
        result=response['result'];self.assertTrue(all(result['audit'].values()))
        self.assertEqual(response['context']['phase'],'observing');self.assertTrue(Path(result['video']).is_file())
        state=self.fixture.send('get_ego_state')['result']['driving_state'];self.assertEqual(state['status'],'completed')
        manifest=_hugsim_expand(json.loads(Path(result['manifest']).read_text()))
        for packet in manifest['frames']:
            self.assertFalse(packet['preview_only']);self.assertTrue(packet['observation_history_committed'])
            for cam in packet['cameras'].values():np.testing.assert_allclose(cam['camera_to_world'],packet['ego_to_world'])
        self.fixture.error(self.fixture.send('edit',self.fixture.args(self.fixture.translate())),'edit_requires_pre_run')
        self.fixture.error(self.fixture.send('execute_ego_trajectory',self.args(self.short_path())),'execution_requires_pre_run')
        reset=self.fixture.send('reset',dict(expected_generation=self.bridge.session.generation,expected_revision=2))
        self.assertEqual(reset['status'],'ok');self.assertTrue(all(reset['result']['reset_audit'].values()))
        self.assertIsNone(self.fixture.send('get_ego_state')['result']['driving_state'])
        replay=self.bridge.handle(req);self.assertTrue(replay['replayed']);self.assertEqual(self.bridge.control_count,4)
        self.assertIsNone(self.bridge.driving_state);self.assertFalse(self.bridge.session.editor.started)
    def test_two_sample_path_completes_without_advancing_past_endpoint(self):
        response=self.fixture.send('execute_ego_trajectory',self.args(self.short_path(2)))
        self.assertEqual(response['status'],'ok',response)
        self.assertEqual(response['result']['executed_controls'],1);self.assertEqual(self.bridge.render_count,2)
        self.assertEqual(self.bridge.driving_state['timestamp_s'],8.1)
    def test_failure_after_one_action_reports_real_partial_execution(self):
        backend=self.bridge.session.backend;original=backend.render
        def failed(pose,time):
            if time>8:raise RuntimeError('explicit post-control raster failure')
            return original(pose,time)
        backend.render=failed
        response=self.fixture.send('execute_ego_trajectory',self.args(self.short_path()),'partial_drive')
        self.fixture.error(response,'backend_error');self.assertTrue(response['control_executed'])
        self.assertEqual(response['controls_executed'],1);self.assertEqual(self.bridge.control_count,1)
        self.assertEqual(self.bridge.driving_state['status'],'failed')
        m=_hugsim_expand(json.loads((self.bridge.output/'driving/partial_drive/execution.json').read_text()))
        self.assertFalse(m['completed']);self.assertEqual(len(m['trace']),1);self.assertEqual(len(m['frames']),1)
        self.assertFalse((self.bridge.output/'driving/partial_drive/ego_driving_execution.mp4').exists())
    def test_solver_nonfinite_failure_never_executes_an_action(self):
        solver=ZeroSolver()
        def bad(state,ref):return [SimpleNamespace(input_trajectory=np.full((len(ref)-1,2),np.nan))]
        solver.solve=bad
        with patch('ego_execution.make_solver',return_value=solver):
            result=self.fixture.send('execute_ego_trajectory',self.args(self.short_path()))
        self.fixture.error(result,'backend_error');self.assertFalse(result['control_executed'])
        self.assertEqual(self.bridge.control_count,0);self.assertEqual(self.bridge.render_count,1)
        self.assertEqual(self.bridge.driving_state['status'],'failed')
    def test_public_eight_step_job_includes_controls_and_reset(self):
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/ego_driving.json').read_text()))
        job['steps'][0]['request']['args'].update(expected_generation=self.bridge.session.generation,expected_revision=0)
        result=run_job(job,self.bridge,self.root/'driving_job')
        self.assertTrue(result['passed']);self.assertEqual(result['executed_controls'],40)
        self.assertEqual(result['actual_renders'],41);self.assertEqual(result['rendered_packets'],41)
        self.assertEqual(result['committed_observations'],41);self.assertEqual(result['pre_run_previews'],0)
        self.assertTrue(result['closed_loop_driving']);self.assertFalse(result['gpu_render_executed'])
        self.assertFalse(result['uniad_inference_executed']);self.assertEqual(result['coverage']['ego_driving_executions'],1)
        self.assertEqual(result['coverage']['resets'],1);self.assertEqual(result['expected_error_count'],1)
    def test_heading_failure_is_not_hidden_by_zero_position_error(self):
        path=self.short_path(3)
        for i,point in enumerate(path['waypoints']):
            point.update(forward_m=0,left_m=0,yaw_degrees=i*15)
        with patch('ego_execution.make_solver',return_value=ZeroSolver()):
            result=self.fixture.send('execute_ego_trajectory',self.args(path))
        self.assertEqual(result['status'],'ok',result)
        tracking=result['result']['tracking']
        self.assertEqual(tracking['max_position_error_m'],0)
        self.assertAlmostEqual(tracking['max_heading_error_degrees'],30)
        self.assertFalse(tracking['passed'])

    def test_inaccurate_motion_fails_tracking_acceptance_instead_of_claiming_success(self):
        req=self.fixture.req('execute_ego_trajectory',self.args(),'bad_track')
        job=dict(schema='agent_job_v1',steps=[dict(request=req)])
        with patch('ego_execution.make_solver',return_value=ZeroSolver()):
            with self.assertRaises(RuntimeError):run_job(job,self.bridge,self.root/'tracking_fail')
        data=_hugsim_expand(json.loads((self.root/'tracking_fail/acceptance.json').read_text()))
        self.assertFalse(data['passed']);self.assertTrue(data['checks']['ego_execution_verified'])
        self.assertFalse(data['checks']['ego_execution_tracking_within_tolerance'])
        self.assertTrue(data['closed_loop_driving']);self.assertEqual(data['executed_controls'],40)
        self.assertGreater(_hugsim_expand(json.loads((self.root/'tracking_fail/ego_driving_index.json').read_text()))[0]['tracking']['final_position_error_m'],4)

if __name__=='__main__':unittest.main()
