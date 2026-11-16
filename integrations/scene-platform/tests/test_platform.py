"""CPU contract, dispatch, exactly-once control and cleanup regressions. No GPU claims."""

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

import copy,json,sys,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from platform_io import LOCAL,read,write,asset
from platform_catalog import Catalog
from platform_contracts import compile_task,tool_specs
from platform_runner import Runner
from platform_service import Service,simulate
from platform_results import collect

TASK=read(LOCAL/'examples/cutin_30.json')

class Contracts(unittest.TestCase):
    def test_stdio_client_uses_portable_tools(self):
        from client import Client
        with Client([sys.executable,'-B',str(LOCAL/'cli.py'),'stdio']) as client:
            result=client.call('capabilities1','get_scene_capabilities',{})
        self.assertTrue(result['ok']);self.assertFalse(result['result']['LLM_connected'])
        self.assertEqual(result['result']['task_schemas'],['scene_task_v1','scene_task_v2'])
    def test_bindings_and_budget(self):
        c=compile_task(TASK,Catalog());self.assertEqual(c['arm'],'B');self.assertEqual(c['control_steps'],30)
        self.assertEqual([m['module'] for m in c['module_calls']][:3],['scene_loader','native_vehicle_trajectory','closed_loop_driver'])
        self.assertEqual(c['source_end_frame'],40)
    def test_reject_unregistered_and_invalid_interventions(self):
        for route,value in [('scene_id','552'),('profile','latest'),('object_id','native:RigidNodes:4'),('program_id','arbitrary'),('control_steps',True),('control_steps',189),('weather','rainy')]:
            task=copy.deepcopy(TASK)
            target=task['simulation'] if route in ('profile','control_steps') else task['edits'][0] if route in ('object_id','program_id') else task
            target[route]=value
            with self.subTest(route=route,value=value),self.assertRaises(ValueError):compile_task(task,Catalog())
    def test_legacy_schema_uses_existing_authoritative_validator(self):
        t=dict(schema='scene_task_v1',task_id='legacy_preview',scene_id='552')
        self.assertEqual(compile_task(t,Catalog())['profile_id'],'legacy_scene_task_v1')
        self.assertEqual(len(tool_specs()[1]['parameters']['properties']['task']['oneOf']),2)

class DurableDispatch(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.runner=Runner(self.root/'store');self.executions=0
    def tearDown(self):self.temp.cleanup()
    def child(self,stage,job):
        if stage=='check':write(job/'preflight.json',dict(verified=True,cuda_available=True));return
        self.executions+=1;write(job/'task_result.json',dict(status='completed',cpu_fixture_only=True))
    def run_fixture(self,task=TASK):
        with patch.object(self.runner,'_child',side_effect=self.child),patch('platform_runner.ROOT',self.root),patch('platform_runner.subprocess.run',return_value=types.SimpleNamespace(stdout='')):
            return self.runner.run(task,'request1')
    def test_same_request_replays_without_controls(self):
        first=self.run_fixture();second=self.run_fixture()
        self.assertFalse(first['replayed']);self.assertTrue(second['replayed']);self.assertEqual(self.executions,1)
        other=copy.deepcopy(TASK);other['simulation']['control_steps']=31
        with self.assertRaises(ValueError):self.run_fixture(other)
    def test_interrupted_task_is_not_reexecuted(self):
        job=self.runner.prepare(TASK,'request1');write(job/'state.json',dict(state='running'))
        with self.assertRaises(RuntimeError):self.run_fixture()
        self.assertEqual(self.executions,0)
    def test_no_gpu_leaves_prepared(self):
        def child(stage,job):write(job/'preflight.json',dict(cuda_available=False))
        with patch.object(self.runner,'_child',side_effect=child),self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):self.runner.run(TASK,'request1')
        self.assertEqual(read(self.root/'store/jobs/request1/state.json')['state'],'prepared')
    def test_saved_task_mutation_rejected(self):
        job=self.runner.prepare(TASK,'request1');write(job/'task.json',dict(TASK,scene_id='552'))
        with self.assertRaises(ValueError):self.run_fixture()
    def test_mock_llm_calls_query_then_real_dispatch_boundary(self):
        calls=[]
        def run(task,rid):calls.append(task);return dict(result=dict(cpu_fixture_only=True),replayed=False)
        with patch.object(self.runner,'run',side_effect=run):
            receipt=simulate(dict(task=TASK),self.runner,self.root/'mock')
        self.assertEqual(calls,[TASK]);self.assertTrue(receipt['run_tool_dispatched']);self.assertFalse(receipt['LLM_invoked'])
        self.assertEqual([r['tool'] for r in read(self.root/'mock/tool_calls.json')],['get_scene_capabilities','run_scene_task'])
    def test_prepare_mode_never_dispatches_controls(self):
        with patch.object(self.runner,'run',side_effect=AssertionError('must not drive')):
            result=simulate(dict(task=TASK),self.runner,self.root/'mock',True)
        self.assertFalse(result['run_tool_dispatched']);self.assertFalse(result['controls_executed'])
    def test_mock_llm_records_unavailable_resource_without_claiming_execution(self):
        def run(task,rid):self.runner.prepare(task,rid);raise RuntimeError('CUDA unavailable')
        with patch.object(self.runner,'run',side_effect=run):r=simulate(dict(task=TASK),self.runner,self.root/'mock')
        self.assertEqual(r['status'],'deferred');self.assertTrue(r['run_tool_dispatched']);self.assertFalse(r['controls_executed'])
        self.assertFalse(read(self.root/'mock/tool_responses.json')[1]['ok'])

class Receipts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.task=copy.deepcopy(TASK);self.task['simulation']['control_steps']=2
        self.profile=Catalog().profile(self.task['simulation']['profile'])
        trace=[dict(index=i,perception_asset=dict(path='CPU fixture'),temporal=dict(history_bev_present=i>0)) for i in range(2)]
        self.report=dict(status='completed',real_closed_loop=True,trace=trace,observations=[dict(index=i) for i in range(3)],
            actor_intervention=dict(object_id='native:RigidNodes:3'),actor_sync=dict(passed=True),model_weights_unchanged=True,
            heavy_mpc_executed=False,termination_reason='source_scene_time_window_complete')
        write(self.root/'B/run_report.json',self.report);write(self.root/'B/native_eval/native_scores.json',dict(scores=dict(executed_path=dict(hdscore=.1))))
        write(self.root/'B/summary.json',dict(any_anomaly=True,anomalies=dict(environment_collision=True),stop_recovery_unresolved=True))
        write(self.root/'platform_edit_receipt.json',dict(command=self.task['edits'][0]))
        write(self.root/'platform_lifecycle/cleanup.json',dict(reset_verified=True,closed=True,recovery_required=False))
        for i in range(2):write(self.root/f'B/uniad_worker/step_{i:03d}/native_collision_audit.json',dict(index=i,enabled=True,native_code_modified=False,native_method_calls=1))
    def tearDown(self):self.temp.cleanup()
    def test_completed_execution_does_not_hide_driving_failure(self):
        r=collect(self.task,self.profile,self.root);self.assertEqual(r['status'],'completed')
        self.assertTrue(r['acceptance']['engineering_passed']);self.assertFalse(r['acceptance']['driving_success'])
        self.assertFalse(r['execution']['full_source_time_window_completed']);self.assertEqual(r['execution']['termination_reason'],'requested_control_budget_complete')
        self.assertTrue(r['execution']['native_collision_enabled']);self.assertEqual(r['metrics']['collision'],True)
    def test_missing_collision_receipt_fails(self):
        (self.root/'B/uniad_worker/step_001/native_collision_audit.json').unlink()
        r=collect(self.task,self.profile,self.root);self.assertFalse(r['acceptance']['engineering_passed'])
    def test_missing_reset_cannot_be_promoted(self):
        (self.root/'platform_lifecycle/cleanup.json').unlink()
        r=collect(self.task,self.profile,self.root);self.assertTrue(r['recovery_required']);self.assertEqual(r['status'],'failed')
    def test_imported_source_is_not_new_execution(self):
        r=collect(self.task,self.profile,self.root,reused=True);self.assertEqual(r['execution']['newly_executed_controls'],0)
        self.assertFalse(r['execution']['uniad_inference_executed']);self.assertTrue(r['execution']['source_uniad_inference_evidence'])

class Lifecycle(unittest.TestCase):
    def test_actual_adapter_calls_edit_and_reset_and_closes_on_reset_failure(self):
        import numpy as np
        from driver_adapter import instrument
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write(root/'initial_context.json',dict(ego_to_world=np.eye(4).tolist(),source_timestamp_s=1.))
            path=root/'scenario/target_track.npz';path.parent.mkdir();path.write_bytes(b'CPU fixture, no GPU data')
            class Backend:
                def __init__(self):self._last=None;self.value=0;self.closed=False;self.fail=False
                def render(self,pose,t):
                    if self.fail:raise RuntimeError('render reset fault')
                    self._last=1;return dict(cameras=dict(CAM_FRONT=dict(rgb=np.full((4,4,3),self.value,dtype=np.uint8))))
                def clear_observation_history(self):self._last=None
            class Session:
                def __init__(self):self.backend=None
                def load_scene(self,m):
                    self.backend=Backend();self.generation='before';self.revision=0;self.editor=types.SimpleNamespace(started=False)
                    self.vehicles=types.SimpleNamespace(programs={})
                def reset(self):self.backend.value=0;self.generation='after';self.revision=0;self.vehicles.programs={}
                def close(self):self.backend.closed=True
            def edit(s,p,t):
                self.assertIsNone(s.backend._last);s.backend.value=200;s.vehicles.programs={3:True};s.revision=1
                return dict(object_id='native:RigidNodes:3')
            native=types.SimpleNamespace(make_types=lambda:(Backend,Session));runtime=types.SimpleNamespace(install_trajectory=edit)
            environment=types.SimpleNamespace(actor_arrays=lambda s:dict(actor=np.array([s.backend.value])))
            with patch.dict(sys.modules,environment=environment):
                restore=instrument(runtime,native,TASK,root,lambda *args:None)
                try:
                    _,S=native.make_types();s=S();s.load_scene('manifest');runtime.install_trajectory(s,path,1.5);s.close()
                    self.assertTrue(read(root/'platform_lifecycle/cleanup.json')['reset_verified']);self.assertTrue(s.backend.closed)
                    s2=S();s2.load_scene('manifest');s2.backend.fail=True
                    with self.assertRaisesRegex(RuntimeError,'reset fault'):s2.close()
                    self.assertTrue(s2.backend.closed);self.assertTrue(read(root/'platform_lifecycle/cleanup.json')['recovery_required'])
                finally:restore()
            self.assertIs(runtime.install_trajectory,edit)

class Presentation(unittest.TestCase):
    def test_real_video_encoder_and_overlay_with_cpu_images(self):
        import numpy as np
        from PIL import Image
        from platform_results import present
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);observations=[]
            for i in range(3):
                pose=np.eye(4);pose[0,3]=i
                observations.append(dict(index=i,timestamp=i*.1,pose=pose.tolist(),motion_state=[0,0,0,2,0],
                    environment=dict(target_world=dict(box=[i+5,2,0,2,4,1,0]))))
                folder=root/f'B/steps/{i:03d}';folder.mkdir(parents=True)
                for name in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']:Image.new('RGB',(48,32),(i*50,100,100)).save(folder/(name+'.png'))
            write(root/'B/run_report.json',dict(observations=observations))
            r=present(root,'B');self.assertTrue(r['full_decode_passed']);self.assertEqual(r['frames'],3)
            self.assertEqual(len(r['artifacts']),2)
            self.assertEqual(present(root,'B'),r)

if __name__=='__main__':unittest.main()
