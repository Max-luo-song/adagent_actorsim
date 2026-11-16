
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

import ast,copy,json,pickle,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from common import configure,BUNDLE,ROOT,read
from tracker import reference,native_legacy_reference,Tracker
from scenario551 import generate


class CoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure()
    def test_native_heading_bug_and_calibrated_mount(self):
        mount=np.eye(4);mount[:2,3]=[.2,.985793]
        raw=np.c_[np.zeros(6),np.arange(1,7)*2.5]-mount[:2,3]
        states,_=reference(raw,mount)
        self.assertAlmostEqual(states[-1,0],15.);self.assertAlmostEqual(states[2,3],5.)
        self.assertEqual(float(np.abs(states[:,1]).max()),0.);self.assertEqual(float(np.abs(states[:,2]).max()),0.)
        stock=native_legacy_reference(np.c_[np.zeros(6),np.arange(1,7)*2.5])
        self.assertAlmostEqual(float(stock[1,2]),np.pi/2)
    def test_actual_motion_signs_and_native_action_units(self):
        from live_env import Motion
        m=Motion(.1,2.7,5.);m.advance(dict(acc=1.,steer_rate=.1))
        self.assertAlmostEqual(m.speed,5.1);self.assertAlmostEqual(m.steer,.01)
        self.assertAlmostEqual(m.z,.51);self.assertEqual(m.x,0.)
        self.assertGreater(m.yaw,0.);m.advance(dict(acc=0.,steer_rate=0.));self.assertGreater(m.x,0.)
    def test_no_heavy_minimize_or_recovery_in_light_control(self):
        from live_env import Motion
        from runtime_configuration import load_vehicle_config
        tracker=Tracker(load_vehicle_config());m=Motion(.1,2.7,5.)
        with patch('scipy.optimize.minimize',side_effect=AssertionError('Heavy optimizer forbidden')):
            action,audit=tracker.control(np.c_[np.zeros(6),np.arange(1,7)*2.5],dict(model_lidar_frame=dict(virtual_to_ego=np.eye(4))),m)
        self.assertFalse(audit['mpc_search']);self.assertFalse(audit['recovery_override']);self.assertEqual(set(action),{'acc_mps2','steer_rate_radps'})


class NativeInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure();cls.tmp=tempfile.TemporaryDirectory();cls.folder=Path(cls.tmp.name);cls.report=generate(cls.folder)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_real_551_identity_motion_and_other_vehicle_audit(self):
        self.assertTrue(self.report['passed']);self.assertEqual(self.report['config']['annotation_id'],'o24FGKGBO7aPlRVYkDyU7A')
        self.assertLess(self.report['metrics']['max_acceleration_mps2'],8.)
        self.assertGreater(self.report['minimum_other_rigid_polygon_distance_m'],0.)
    def test_actual_shared_tensor_and_heldout_render_context(self):
        from native_car_edit import VehiclePrograms
        from environment import install_trajectory,actor_arrays
        with np.load(BUNDLE/'actors.npz') as z:
            state={key.split('__',1)[1]:torch.as_tensor(z[key].copy()) for key in z.files if key.startswith('RigidNodes__')}
        model=SimpleNamespace(**state);backend=SimpleNamespace(trainer=SimpleNamespace(models={'RigidNodes':model},test_timesteps=list(range(10,199,10))),
            env=SimpleNamespace(models={'RigidNodes':state}),torch=torch)
        spec=SimpleNamespace(fps=10.,frames=199);owner=VehiclePrograms(backend,spec);owner.activation_frame=12
        backend._last=None;backend.clear_observation_history=lambda:None
        session=SimpleNamespace(backend=backend,spec=spec,vehicles=owner,revision=0,
            editor=SimpleNamespace(require_current=lambda:None,started=False,journal=[]))
        before=actor_arrays(session);original=state['instances_trans'].clone()
        with patch('native_radiance.radiance_state',return_value=None):receipt=install_trajectory(session,self.folder/'target_track.npz',1.2)
        after=actor_arrays(session)
        np.testing.assert_array_equal(before['RigidNodes__instances_trans'][:13],after['RigidNodes__instances_trans'][:13])
        np.testing.assert_array_equal(before['RigidNodes__instances_trans'][:,[i for i in range(15) if i!=3]],after['RigidNodes__instances_trans'][:,[i for i in range(15) if i!=3]])
        self.assertTrue(receipt['renderer_state_collision_shared_tensor'])
        t=state['instances_trans'];saved=t.clone()
        with session.vehicles.render_context(10):self.assertTrue(torch.equal(t,saved))
        with session.vehicles.render_context(20):
            self.assertTrue(torch.equal(t[19,3],t[20,3]) and torch.equal(t[21,3],t[20,3]))
        self.assertTrue(torch.equal(t,saved));session.vehicles.restore();self.assertTrue(torch.equal(t,original))
    def test_unmodified_native_collision_kernel_boundary(self):
        from environment import native_kernels
        kernel=native_kernels();ego=[0,0,0,2,4,1.6,0]
        self.assertTrue(kernel['fg_collision_det'](ego,[[1,0,0,2,4,1.6,0]]));self.assertFalse(kernel['fg_collision_det'](ego,[[10,0,0,2,4,1.6,0]]))
        box=torch.tensor([[1,1,0],[1,-1,0],[1,1,2],[1,-1,2],[-1,-1,0],[-1,1,0],[-1,-1,2],[-1,1,2]],dtype=torch.float32)
        self.assertFalse(kernel['bg_collision_det'](torch.tensor([[0.,0.,1.]]*100),box))
        self.assertTrue(kernel['bg_collision_det'](torch.tensor([[0.,0.,1.]]*101),box))


class PatchAndPrefixTests(unittest.TestCase):
    def test_worker_requires_real_first_packet_before_process_launch(self):
        from runtime import start_worker,dump_pickle
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);folder=out/'uniad_worker'
            with patch('runtime.subprocess.Popen') as launch:
                with self.assertRaises(FileNotFoundError):start_worker(folder,out,123)
                launch.assert_not_called()
            obs={'camera_pixels':np.arange(6).reshape(2,3)};info={'timestamp':0.}
            dump_pickle(folder/'step_000/observation.pkl',(obs,info))
            def launch_checked(*args,**kwargs):
                with (folder/'step_000/observation.pkl').open('rb') as f:actual,metadata=pickle.load(f)
                np.testing.assert_array_equal(actual['camera_pixels'],obs['camera_pixels'])
                self.assertEqual(metadata,info)
                return SimpleNamespace()
            with patch('runtime.subprocess.Popen',side_effect=launch_checked),patch('runtime.wait_json',return_value={'ready':True}):
                process,log,ready=start_worker(folder,out,123)
                log.close();self.assertTrue(ready['ready'])
    def test_real_loop_persists_observation_before_worker_boot(self):
        from common import HERE
        tree=ast.parse((HERE/'runtime.py').read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_arm')
        boot=next(n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='start_worker')
        loop=next(n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='index')
        writes=[n.lineno for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dump_pickle']
        self.assertIn(boot,list(ast.walk(loop)));self.assertLess(max(writes),boot.lineno)
    def test_patch_off_preserves_raw_reference(self):
        from runtime import processor
        raw=np.arange(12,dtype=float).reshape(6,2);adopted,audit,state=processor(raw,{},dict(speed_mps=4),None,None)
        np.testing.assert_array_equal(raw,adopted);self.assertFalse(audit['patch']);adopted[0]=0
        self.assertNotEqual(float(raw[0,1]),0.)
    def test_prefix_gate_rejects_actual_observation_or_action_divergence(self):
        from common import write
        from runtime import prefix_gate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'steps/000').mkdir(parents=True);(root/'uniad_worker/step_000').mkdir(parents=True);folder=root/'gate';folder.mkdir()
            receipt=dict(ego_to_world=np.eye(4).tolist(),image_sha256={'front':'exact'},action={'acc_mps2':0.,'steer_rate_radps':0.})
            write(root/'steps/000/receipt.json',receipt);np.savez(root/'uniad_worker/step_000/perception.npz',trajectory=np.zeros((6,2)))
            action=prefix_gate(folder,root,0,np.eye(4),{'front':'exact'},{'trajectory':np.zeros((6,2))},receipt['action'])
            self.assertEqual(action,receipt['action'])
            with self.assertRaises(ValueError):prefix_gate(folder,root,0,np.eye(4),{'front':'different'},{'trajectory':np.zeros((6,2))},receipt['action'])
            with self.assertRaises(ValueError):prefix_gate(folder,root,0,np.eye(4),{'front':'exact'},{'trajectory':np.zeros((6,2))},dict(acc_mps2=1.,steer_rate_radps=0.))


class DiagnosticEvidenceTests(unittest.TestCase):
    def test_target_truth_is_associated_only_in_posthoc_virtual_frame(self):
        from diagnostics import target_association
        info=dict(explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=[0.,1.,1.6])
        arrays=dict(boxes=np.array([[-2.,9.,0.,2.,5.,1.6,0.,0.,0.]]),scores=np.array([.9]))
        result=target_association([10.,2.,1.],info,arrays)
        np.testing.assert_allclose(result['current_target_virtual_xyz'],[-2.,9.,-.6],atol=1e-8)
        self.assertAlmostEqual(result['retained']['xy_distance_m'],0.);self.assertFalse(result['used_by_online_policy'])
    def test_native_environment_opacity_filter_is_preserved(self):
        from environment import Evaluation
        class Archive(dict):
            def __enter__(self):return self
            def __exit__(self,*args):return False
        def load(path,*args,**kwargs):
            name=Path(path).name
            if name=='background_sr_world.npy':return np.tile([0.,0.,1.],(101,1))
            if name=='background_semantics.npz':return Archive(class_id=np.full(101,2))
            if name=='native_dense_route.npy':
                poses=np.repeat(np.eye(4)[None],100,axis=0);poses[:,2,3]=np.arange(100)*2;return poses
            raise AssertionError(name)
        vehicle=SimpleNamespace(half_width_m=.8,half_length_m=1.5,body_height_m=1.6)
        with patch('environment.np.load',side_effect=load):
            low=Evaluation({},vehicle,np.full(101,.7),device='cpu');high=Evaluation({},vehicle,np.full(101,.9),device='cpu')
        self.assertFalse(low.check(np.eye(4),0)['background_collision'])
        self.assertTrue(high.check(np.eye(4),0)['background_collision'])
    def test_existing_valid_presentation_is_reused(self):
        from common import write,asset
        from diagnostics import artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'presentation_A_B';stage.mkdir();video=stage/'cached.mp4';video.write_bytes(b'fixture-cache-contract')
            plot=stage/'cached.png';plot.write_bytes(b'fixture-cache-contract')
            for name in ['A','B']:write(root/name/'run_report.json',dict(fixture=True))
            write(stage/'artifacts.json',dict(video=asset(video),overlay=asset(plot),full_decode_passed=True,source_reports=[asset(root/name/'run_report.json') for name in ['A','B']]))
            result=artifacts(root);self.assertTrue(result['reused'])
    def test_patch_pair_does_not_treat_original_B_failure_as_A_baseline_failure(self):
        from common import write
        from diagnostics import paired
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'scenario/scenario.json',dict(config=dict(intervention_start_s=.1)))
            observations=[dict(index=i,timestamp=i*.1,image_sha256={'front':'same'},motion_state=[0,0,0,0,0],pose=np.eye(4).tolist(),environment={}) for i in range(2)]
            report=dict(trace=[],observations=observations,vehicle_geometry={})
            for name in ['B','B_patch_on']:
                write(root/name/'run_report.json',report);np.savez(root/name/'actor_tracks.npz',same=np.ones(1))
            for i in range(2):write(root/'B_patch_on'/f'steps/{i:03d}/prefix_gate.json',dict(passed=True))
            summaries=[dict(any_anomaly=True),dict(any_anomaly=False)]
            with patch('diagnostics.arm_summary',side_effect=summaries):result=paired(root,'B','B_patch_on')
            self.assertEqual(result['comparison_role'],'external_patch');self.assertTrue(result['actor_tracks_identical'])
            self.assertEqual(result['conclusion'],'patch_pair_anomaly_resolved_normal_A_check_still_required')
    def test_read_only_motion_wrapper_returns_original_objects(self):
        from common import HERE,write
        tree=ast.parse((HERE/'worker_trace.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'step_000').mkdir()
            result=([],dict(all_traj_preds=torch.ones(1,1,2,6,12,5),all_traj_scores=torch.zeros(1,1,2,6),track_scores=torch.ones(1,1)))
            scope=dict(original=lambda *args,**kwargs:result,np=np,write=write,frozen_worker=SimpleNamespace(worker=SimpleNamespace(JOB=root)))
            exec(compile(ast.Module(body=[node],type_ignores=[]),'readonly_export_test','exec'),scope)
            returned=scope['observed'](None,outs_track={})
            self.assertIs(returned,result);self.assertTrue(torch.equal(result[1]['all_traj_preds'],torch.ones_like(result[1]['all_traj_preds'])))
            self.assertTrue((root/'step_000/motion_head_raw.npz').exists())
    def test_normal_yielding_stop_is_not_automatically_abnormal_stall(self):
        from common import write
        from diagnostics import arm_summary
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);observations=[dict(index=i,timestamp=i*.1,pose=np.eye(4).tolist(),motion_state=[0,0,0,0,0],environment=dict(collision=False,native_route_deviation=False)) for i in range(41)]
            def trace(speed):return [dict(index=i,control=dict(target_speed_mps=speed,solve_seconds=.01),tracking_error=dict(position_m=0.)) for i in range(40)]
            report=dict(status='completed',trace=trace(0.),observations=observations,wall_seconds=1.,termination_reason='simulation_observation_limit')
            write(root/'run_report.json',report);normal=arm_summary(root)
            self.assertFalse(normal['anomalies']['abnormal_stall']);self.assertTrue(normal['stop_recovery_unresolved'])
            report['trace']=trace(3.);write(root/'run_report.json',report);bad=arm_summary(root)
            self.assertTrue(bad['anomalies']['abnormal_stall'])
    def test_native_execution_stream_uses_recorded_future_not_neural_plan(self):
        from common import write,CR,SOURCE
        from scoring import score_arm
        configure();import native_scores
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm=root/'A';arm.mkdir();mount=np.eye(4);mount[1,3]=.985793
            write(root/'scene_manifest.json',read(CR/'phase31/stopped_vehicle551_v1/scene_manifest.json'))
            write(root/'scenario/scenario.json',dict(fixture=True))
            write(root/'plan.json',dict(seed=1,assets=dict(uniad_weights=dict(path='fixture',sha256='fixture'))))
            with np.load(BUNDLE/'actors.npz') as z:np.savez_compressed(arm/'actor_tracks.npz',**{k:z[k] for k in z.files})
            observations=[]
            for i in range(31):
                pose=np.eye(4);pose[0,3]=.2*i;observations.append(dict(index=i,pose=pose.tolist()))
                step=arm/f'uniad_worker/step_{i:03d}';step.mkdir(parents=True)
                info=dict(model_lidar_frame=dict(virtual_to_ego=mount),explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.array([0,.2*i,0])+mount[:3,3],timestamp=.1*i)
                with (step/'sensor_metadata.pkl').open('wb') as f:pickle.dump(info,f)
                np.savez(step/'perception.npz',trajectory=np.c_[np.zeros(6),np.arange(1,7)*10.])
                write(arm/f'steps/{i:03d}/reference.json',dict(raw_uniad=np.c_[np.zeros(6),np.arange(1,7)*10.].tolist(),adopted=np.c_[np.zeros(6),np.arange(1,7)*10.].tolist()))
            write(arm/'run_report.json',dict(status='completed',trace=[{}]*30,observations=observations))
            captured={}
            def fake_native(run,vehicle):
                data,manifest=native_scores.export(run,vehicle);captured.update(data=data,manifest=manifest)
                return dict(status='fixture_only_no_gpu',scores={})
            with patch.object(native_scores,'score',side_effect=fake_native):score_arm(root,'A')
            executed=captured['data']['executed_path'][0]['frames'][0]['planned_traj']['traj']
            raw=captured['data']['raw_uniad'][0]['frames'][0]['planned_traj']['traj']
            self.assertAlmostEqual(executed[-1][0],5.);self.assertGreater(raw[-1][0],40.)
            self.assertEqual(captured['manifest']['executed_path_keyframes'],6)
            self.assertTrue(captured['manifest']['executed_path_scored'])


if __name__=='__main__':unittest.main()
