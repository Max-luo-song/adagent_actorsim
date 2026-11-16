
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
from unittest.mock import patch
import numpy as np
import agent_bootstrap as bootstrap
import test_agent_bridge as fixtures
from scene_tasks import compile_task,validate_task,execute_task,TASK_SCHEMA,RESULT_SCHEMA
from scene_agent_client import SceneAgentClient,tool_specs
from agent_jobs import validate_job


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.BridgeTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.b=self.f.bridge
        self.client=SceneAgentClient(self.b.handle)
    def task(self,**updates):
        task=dict(schema='scene_task_v1',task_id='integration',scene_id='552',edits=[self.f.insert(),self.f.translate()],previews=[dict(kind='camera',args=dict(source_frame=80))],verification_frames=[80],reset_after=True)
        task.update(updates);return task
    def test_full_task_uses_real_editor_and_returns_agent_artifacts(self):
        r=self.client.run_task(self.task());self.assertEqual(r['status'],'ok',r);result=r['result']
        self.assertEqual(result['schema'],'scene_task_result_v1');self.assertEqual(result['status'],'completed')
        self.assertEqual(result['execution']['actual_renders'],3);self.assertEqual(result['acceptance']['coverage']['edits'],2)
        self.assertFalse(result['execution']['gpu_render_executed']);self.assertFalse(result['execution']['uniad_inference_executed'])
        self.assertEqual(result['context']['phase'],'pre_run');self.assertEqual(result['context']['edit_revision'],0)
        self.assertTrue(any(a['kind']=='observation' for a in result['artifacts']));self.assertTrue(Path(result['result_file']).is_file())
        self.assertEqual(len(self.f.backends),1)
    def test_second_task_reuses_loaded_scene_with_reset_and_version_guard(self):
        a=self.client.run_task(self.task(),request_id='first');b=self.client.run_task(self.task(task_id='second'),request_id='second')
        self.assertEqual(a['status'],'ok',a);self.assertEqual(b['status'],'ok',b);self.assertEqual(len(self.f.backends),1)
        self.assertEqual(b['result']['acceptance']['scene_loads'],0);self.assertEqual(b['result']['acceptance']['scene_resets'],2)
        self.assertNotEqual(a['result']['context']['generation'],b['result']['context']['generation'])
    def test_task_replay_freezes_original_versions_and_has_no_side_effects(self):
        a=self.client.run_task(self.task(),request_id='once');counts=(self.b.render_count,self.b.reset_count,self.b.load_count)
        b=self.client.run_task(self.task(),request_id='once');self.assertTrue(b['replayed']);self.assertEqual(a['result'],b['result'])
        self.assertEqual(counts,(self.b.render_count,self.b.reset_count,self.b.load_count))
        with self.assertRaises(ValueError):self.client.run_task(self.task(task_id='different'),request_id='once')
    def test_cli_task_mode_writes_the_same_result_bundle(self):
        import io,sys
        from contextlib import redirect_stdout
        import run_agent_json
        task_file=self.f.root/'task_input.json';task_file.write_text(json.dumps(self.task()))
        output=self.f.root/'cli_output'
        argv=['run_agent_json.py','--task',str(task_file),'--output',str(output),'--catalog',str(self.f.catalog.path)]
        with patch.object(sys,'argv',argv),patch.object(run_agent_json,'AgentBridge',return_value=self.b),redirect_stdout(io.StringIO()):run_agent_json.main()
        result=_hugsim_expand(json.loads((output/'task_result.json').read_text()));self.assertEqual(result['status'],'completed')
        self.assertEqual(result['schema'],'scene_task_result_v1');self.assertEqual(result['execution']['actual_renders'],3)
        self.assertTrue((output/'task/compiled_job.json').is_file());self.assertTrue(self.b.closed)
    def test_stale_task_never_resets_or_edits_current_scene(self):
        generation=self.f.load();args=dict(task=self.task(),expected_generation=generation,expected_revision=3)
        r=self.f.send('run_scene_task',args);self.assertEqual(r['error']['code'],'stale_revision');self.assertEqual(self.b.reset_count,0)
    def test_bad_task_and_unsupported_driver_rejected_before_loading(self):
        for task in [self.task(ego_execution=dict(driver='resident_uniad',trajectory={})),self.task(weather='snowy'),self.task(reset_after='yes'),self.task(edits=[dict(operation='execute_shell',command='true')])]:
            r=self.client.run_task(task);self.assertEqual(r['status'],'error',r);self.assertEqual(r['error']['code'],'invalid_scene_task')
        self.assertEqual(self.b.load_count,0)
    def test_failed_scene_setup_does_not_reset_existing_user_edits(self):
        self.f.load();self.f.send('edit',self.f.args(self.f.insert()));before=self.b.context();objects=copy.deepcopy(self.b.session.objects)
        self.client.call('get_capabilities')
        r=self.client.run_task(self.task(scene_id='unregistered'));self.assertEqual(r['status'],'error')
        result=r['error']['details'];self.assertIsNone(result['cleanup']);self.assertFalse(result['recovery_required'])
        self.assertEqual(self.b.context(),before);self.assertEqual(self.b.session.objects,objects)
    def test_failed_mid_task_reports_partial_work_and_cleans_up(self):
        task=self.task(edits=[self.f.insert(),dict(operation='translate',object_id='native:SMPLNodes:999',offset_world_m=[.5,0,0])])
        r=self.client.run_task(task);self.assertEqual(r['error']['code'],'task_execution_failed',r)
        result=r['error']['details'];self.assertEqual(result['status'],'failed');self.assertEqual(result['acceptance']['coverage']['edits'],1)
        self.assertEqual(result['cleanup']['status'],'ok');self.assertFalse(result['recovery_required']);self.assertEqual(self.b.session.revision,0)
        self.assertFalse(self.b.session.objects);self.assertTrue(Path(result['result_file']).is_file())
    def test_failed_task_without_reset_retains_explicit_partial_state(self):
        task=self.task(verification_frames=[],reset_after=False,edits=[self.f.insert(),dict(operation='remove',object_id='managed:missing')])
        r=self.client.run_task(task);result=r['error']['details'];self.assertTrue(result['recovery_required']);self.assertIsNone(result['cleanup'])
        self.assertEqual(self.b.session.revision,1);self.assertTrue(self.b.session.objects)
    def test_scene_switch_uses_registered_scene_and_closes_old_backend(self):
        self.client.run_task(self.task());old=self.b.session.backend
        task=self.task(task_id='switch',scene_id='551',edits=[],previews=[],verification_frames=[])
        r=self.client.run_task(task);self.assertEqual(r['status'],'ok',r);self.assertTrue(old.closed);self.assertEqual(self.b.scene_id,'551')
    def test_no_nested_or_caller_supplied_task_versions(self):
        for bad in [self.task(edits=[dict(self.f.translate(),expected_revision=0)]),self.task(previews=[dict(kind='camera',args=dict(source_frame=80,expected_revision=0))]),self.task(scene_id={'$ref':'x.result.scene_id'})]:
            with self.assertRaises(ValueError):compile_task(bad)
        nested=dict(schema='agent_job_v1',steps=[dict(request=self.f.req('run_scene_task',{'task':self.task()}))])
        with self.assertRaises(ValueError):validate_job(nested)
    def test_tool_contract_and_dispatch_need_no_llm_dependency(self):
        r=self.client.dispatch('get_scene_capabilities',{});self.assertTrue(r['result']['scene_task']);self.assertFalse(r['result']['resident_uniad_adapter'])
        self.assertEqual([t['name'] for t in tool_specs()],['get_scene_capabilities','run_scene_task'])
        response=self.f.send('get_task_contract');self.assertEqual(response['result']['task_schema'],TASK_SCHEMA)
        with self.assertRaises(ValueError):self.client.dispatch('execute_shell',{})
    def test_rainy_factory_is_explicit_and_actor_and_depth_evidence_checked(self):
        factory=self.b.session.factory;created=[]
        def weather(spec):
            backend=factory(spec);backend.weather_records=[];render=backend.render
            def rainy(pose,timestamp):
                raw=render(pose,timestamp)
                for c in raw['cameras'].values():c['rgb'][:]=100
                backend.weather_records.append(dict(index=len(backend.weather_records),cameras={name:dict(actor_rgb_preserved=True,depth_unchanged=True,changed_pixels=24) for name in raw['cameras']}))
                return raw
            backend.render=rainy;created.append(backend);return backend
        self.b.weather_factory=weather
        a=self.client.run_task(self.task(weather='rainy'));self.assertEqual(a['status'],'ok',a)
        result=a['result'];self.assertEqual(result['execution']['weather_observations'],3);self.assertTrue(result['execution']['weather_rgb_preservation_verified'])
        self.assertTrue(result['acceptance']['checks']['weather_evidence_verified']);self.assertEqual(result['context']['weather'],'rainy')
        snapshot=next(a['path'] for a in result['artifacts'] if a['kind']=='weather_report');before=Path(snapshot).read_bytes()
        b=self.client.run_task(self.task(task_id='rain_again',weather='rainy'));self.assertEqual(b['status'],'ok',b);self.assertEqual(len(created),1)
        self.assertEqual(Path(snapshot).read_bytes(),before)
        c=self.client.run_task(self.task(task_id='clear_again'));self.assertEqual(c['status'],'ok',c);self.assertTrue(created[0].closed);self.assertEqual(self.b.weather,'clear')
    def test_jsonl_task_response_contains_no_batch_progress_lines(self):
        import io
        from contextlib import redirect_stderr
        from run_agent_json import serve_jsonl
        value=self.f.req('run_scene_task',{'task':self.task()},'stream_task');output=io.StringIO()
        with redirect_stderr(io.StringIO()):serve_jsonl(self.b,io.StringIO(json.dumps(value)+'\n'),output)
        lines=output.getvalue().splitlines();self.assertEqual(len(lines),1)
        result=_hugsim_expand(json.loads(lines[0]));self.assertEqual(result['status'],'ok',result);self.assertEqual(result['result']['status'],'completed')
    def test_unavailable_weather_rejected_without_closing_scene(self):
        self.f.load();backend=self.b.session.backend
        r=self.f.send('load_scene',dict(scene_id='552',weather='rainy',expected_generation=self.b.session.generation,expected_revision=0))
        self.assertEqual(r['error']['code'],'unsupported_weather');self.assertIs(self.b.session.backend,backend);self.assertFalse(backend.closed)

class TaskDrivingTests(unittest.TestCase):
    def setUp(self):
        import test_ego_execution
        self.f=test_ego_execution.EgoExecutionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.b=self.f.bridge;self.client=SceneAgentClient(self.b.handle)
    def test_task_runs_actual_ilqr_then_reset_and_replay_without_second_control(self):
        task=dict(schema='scene_task_v1',task_id='drive_task',scene_id='552',edits=[self.f.fixture.insert()],previews=[dict(kind='camera',args=dict(source_frame=80))],ego_execution=dict(driver='reference_tracking',trajectory=self.f.short_path()),verification_frames=[80],reset_after=True)
        r=self.client.run_task(task,request_id='drive_once');self.assertEqual(r['status'],'ok',r)
        self.assertEqual(r['controls_executed'],4);self.assertTrue(r['control_executed']);self.assertEqual(r['result']['execution']['executed_controls'],4)
        self.assertEqual(r['result']['execution']['actual_renders'],8);self.assertTrue(r['result']['acceptance']['checks']['ego_execution_verified'])
        self.assertFalse(r['result']['execution']['uniad_inference_executed']);self.assertEqual(self.b.context()['phase'],'pre_run')
        again=self.client.run_task(task,request_id='drive_once');self.assertTrue(again['replayed']);self.assertEqual(self.b.control_count,4)

class TaskActorLightTests(unittest.TestCase):
    def setUp(self):
        import test_actor_motion
        from test_traffic_lights import LightModel
        self.f=test_actor_motion.SmoothTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.b=self.f.b;self.client=SceneAgentClient(self.b.handle);backend=self.b.session.backend
        light=LightModel();backend.trainer.models['TrafficLightNodes']=light
        render=self.f.manager.original_render
        def combined(pose,timestamp):
            light.cur_frame=int(getattr(backend,'_actor_source_frame',round(timestamp*10)))
            light.get_gaussians();return render(pose,timestamp)
        self.f.manager.original_render=combined
    def test_task_combines_assets_native_light_and_smooth_pedestrian(self):
        edits=[self.f.f.insert(),dict(operation='translate',object_id='native:RigidNodes:0',offset_world_m=[0,.3,0]),dict(operation='set_signal',object_id='native:TrafficLightNodes:7',state='red'),self.f.command(self.f.smooth())]
        task=dict(schema='scene_task_v1',task_id='all_modules',scene_id='552',edits=edits,previews=[dict(kind='camera',args=dict(source_frame=80)),dict(kind='actor_trajectory',args=dict(object_id='native:SMPLNodes:0',start_frame=80,end_frame=82,camera_source_frame=80,render_fps=30))],verification_frames=[80],reset_after=True)
        r=self.client.run_task(task);self.assertEqual(r['status'],'ok',r)
        result=r['result'];self.assertEqual(result['acceptance']['coverage']['edits'],4);self.assertEqual(result['execution']['actual_renders'],17)
        self.assertEqual(result['context']['phase'],'pre_run');self.assertFalse(self.b.session.editor.actor_paths.programs);self.assertFalse(self.b.session.editor.traffic.has_programs)
        self.assertEqual(self.b.load_count,1);self.assertTrue(any(a['kind']=='video' for a in result['artifacts']))

if __name__=='__main__':unittest.main()
