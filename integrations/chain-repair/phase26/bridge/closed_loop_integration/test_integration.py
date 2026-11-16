"""CPU orchestration tests; doubles never count as GPU acceptance."""

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
pass  # Path is provided by the portable boundary above
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import run_edited_closed_loop as runner
sys.path.insert(0,str(runner.PHASE/'bridge/framework_backend'))
from sensor_backend import sensor_plane

class SensorTests(unittest.TestCase):
    def test_singleton_channel_and_z_preserved(self):
        z=np.array([[[2.],[4.]]],dtype=np.float32)
        np.testing.assert_array_equal(sensor_plane(z,(1,2),'depth'),[[2,4]])
    def test_invalid_not_silently_filled(self):
        for value,name in (([[float('nan')]],'depth'),([[-1]],'depth'),([[1.1]],'alpha')):
            with self.assertRaises(ValueError):sensor_plane(value,(1,1),name)
        with self.assertRaises(ValueError):sensor_plane(np.ones((1,2)),(2,1),'alpha')

class StartupTests(unittest.TestCase):
    def test_readiness_failure_retains_handle_for_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);process=object();cleaned=[]
            def fail(*args,**kwargs):raise RuntimeError('worker boot failed')
            h=types.SimpleNamespace(SEED=1,WORKER='frozen_worker.py',wait_json=fail,
                _stop_worker=lambda w,p,log:(cleaned.append(p),log.close()))
            with patch('subprocess.Popen',return_value=process):
                with self.assertRaisesRegex(RuntimeError,'worker boot failed'):runner.start_worker(h,folder,folder)
            self.assertEqual(cleaned,[process])

class ContractTests(unittest.TestCase):
    def test_reject_external_driver_and_unpinned_insertion(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'task.json'
            task=_hugsim_expand(json.loads((runner.HERE/'examples/source_insert551.json').read_text()))
            for key,value in (('driver','arbitrary.py'),('edits',[]),('edits',[{'source_insert':False}])):
                bad=copy.deepcopy(task);bad[key]=value;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):runner.read_task(path)

    def test_only_registered_profiles_are_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'task.json'
            for profile in runner.PROFILES:
                task=dict(schema='edited_hugsim_closed_loop_v1',driver=profile,edits=[dict(source_insert=True)])
                path.write_text(json.dumps(task));self.assertEqual(runner.read_task(path)['driver'],profile)
            task['driver']={};path.write_text(json.dumps(task))
            with self.assertRaises(ValueError):runner.read_task(path)

class OwnershipTests(unittest.TestCase):
    def episode(self,folder,*,failure=False,stale=False,controls=30):
        calls=[];holder={};writes={}
        class Session:
            def __init__(self,backend_factory):
                self.backend=object();self.generation='generation-1';self.revision=0;holder['session']=self
            def load_scene(self,path):calls.append('load')
            def edit(self,command):self.revision+=1;calls.append('edit')
            def edit_journal(self):return [{'revision':self.revision}]
            def render(self,pose,timestamp):
                calls.append('render')
                return dict(history_token=self.generation,edit_revision=self.revision,source_frame=0,timestamp_s=0,
                    cameras={'CAM_FRONT':dict(rgb=np.zeros((2,2,3),dtype=np.uint8),depth=np.ones((2,2)),depth_alpha=np.ones((2,2)))})
            def close(self):calls.append('close')
        h=types.SimpleNamespace(THREE_CAMERAS=['CAM_FRONT'])
        h.atomic_json=lambda path,data:writes.__setitem__(Path(path).name,copy.deepcopy(data))
        h.atomic_pickle=lambda *a: calls.append('seed')
        h._load_arm_request=lambda path:dict(arm='edited')
        spec=types.SimpleNamespace(fps=10)
        setup=dict(spec=spec,manifest={},checkpoint='checkpoint',start=0,relative=[np.eye(4)],initial_native=np.eye(4),reference={},vehicle={},insert=dict(object_id='car',operation='insert'))
        h._setup=lambda arm,out:setup
        h._implementation=lambda *a:lambda:{'frozen':'digest'}
        h.fingerprint=lambda *a:{'hash':'frozen'}
        h._base_report=lambda *a:dict(timing={},trace=[],observations=[])
        h._fresh_driver_state=lambda *a:dict(token='old')
        def packet_factory(setup,spec,state):
            self.assertEqual(state['token'],'generation-1:1')
            return lambda packet:({},dict(depth_encoding='expected_camera_z_v1'))
        h._model_packet_factory=packet_factory
        h._load_reliability_patch=lambda *a:None;h._termination_cfg=lambda *a:{}
        h.reference_check=lambda *a:dict(overlap=False)
        h._spawn_worker=lambda *a:(calls.append('worker') or (object(),object(),{},.1))
        def stop(*args):
            if failure or stale:self.assertEqual(writes['run_report.json']['status'],'failed')
            calls.append('stop')
        h._stop_worker=stop
        def loop(ctx,out,session,report,*args):
            self.assertIs(session,holder['session']);self.assertEqual(session.revision,1)
            self.assertEqual(calls,['load','edit','render','seed','worker'])
            if stale:session.revision+=1
            ctx['model_packet'](session.render(np.eye(4),0))
            if failure:raise RuntimeError('inference failed')
            report['trace']=[dict(physics_integration_verified=True,temporal=dict(history_bev_present=i>0,metadata_cache_is_tracking_cache=True)) for i in range(controls)];report['observations']=[{}]*(controls+1)
        h._control_loop=loop;h._score_run=lambda *a:None
        driver=types.SimpleNamespace(h=h,base=types.SimpleNamespace())
        launch=types.SimpleNamespace(verify=lambda:None)
        task=dict(driver='phase34_global_control551_v1',edits=[dict(source_insert=True)])
        audit=dict(adapter_implementation=runner.provenance())
        modules={'scene_edit_session':types.SimpleNamespace(UnifiedSceneSession=Session),'sensor_backend':types.SimpleNamespace(ClosedLoopFrameworkBackend=object)}
        with patch.dict(sys.modules,modules), patch.object(runner,'start_worker',side_effect=lambda h,w,o:h._spawn_worker(w,o)):
            try:result=runner.run(task,driver,launch,audit,folder/'run')
            except RuntimeError:
                self.assertTrue(failure or stale)
                result=writes['run_report.json']
        self.assertEqual(calls.count('load'),1)
        self.assertEqual(calls[-2:],['stop','close'])
        return result
    def test_one_edited_session_reaches_real_loop_entry(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.episode(Path(d));self.assertTrue(result['integration_acceptance_passed'])
    def test_inference_failure_cleans_worker_and_records_failure(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.episode(Path(d),failure=True);self.assertEqual(result['status'],'failed');self.assertFalse(result['integration_acceptance_passed'])
    def test_stale_revision_cannot_feed_driver(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.episode(Path(d),stale=True);self.assertIn('ownership/revision',result['error'])
    def test_short_run_is_not_full_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.episode(Path(d),controls=3);self.assertFalse(result['integration_acceptance_passed'])

if __name__=='__main__':unittest.main()
