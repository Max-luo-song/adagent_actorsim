"""Exercise the normal episode entry with CPU fake render/inference boundaries."""

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

import argparse
import hashlib
import json
pass  # Path is provided by the portable boundary above
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'weather_runtime'))
import run_weather_loop as runner
import paired_closed_loop_arm as harness
from scene_edit_session import saf
from shared_weather_scene import SharedWeatherScene
from test_lifecycle_probe import Backend


class WeatherBackend(Backend):
    @property
    def weather_epoch(self):return self.epoch
    def __init__(self,spec,output):
        super().__init__(spec);Path(output).mkdir(parents=True)
    def begin_weather_episode(self,output):
        self.clear_observation_history();self.weather_records=[];Path(output).mkdir(parents=True)
    def render(self,pose,timestamp):
        raw=super().render(pose,timestamp)
        record=self.weather_records[-1]
        record.update(timestamp=timestamp,ego_to_world=np.asarray(pose).tolist(),cameras={})
        for cam,c in raw['cameras'].items():
            c['rgb']=c['rgb']+5
            record['cameras'][cam]=dict(rgb_sha256=hashlib.sha256(c['rgb'].tobytes()).hexdigest(),
                actor_rgb_preserved=True,depth_unchanged=True,changed_pixels=24,
                camera_to_ego=np.eye(4).tolist(),camera_to_world=pose.tolist())
        return raw


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.kwargs=[]
        spec=SimpleNamespace(source=dict(scene=552,checkpoint='fake'),fps=10.,frames=90,scale=1.,
            camera_order=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'),summary=lambda:{})
        patches=[patch.object(saf.SceneSpec,'read',return_value=spec),
            patch.object(runner,'RuntimeWeatherBackend',WeatherBackend),patch.object(runner.fw,'run',side_effect=self.fake_drive)]
        for item in patches:item.start();self.addCleanup(item.stop)

    def fake_drive(self,request_path,output,**kwargs):
        self.kwargs.append(kwargs)
        request=_hugsim_expand(json.loads(Path(request_path).read_text()));steps=request['steps']
        output=Path(output);output.mkdir()
        observations=[]
        with harness.SceneSession() as session:
            session.load_scene(request['scene_manifest'])
            for i in range(steps+1):
                stamp=8.+i*.1;pose=np.eye(4)
                packet=session.render(pose,stamp)
                observations.append(dict(closed_ego_to_world=pose.tolist(),timestamp=stamp))
                if i<steps:
                    folder=output/f'uniad_worker/step_{i:03d}';folder.mkdir(parents=True)
                    harness.atomic_pickle(folder/'observation.pkl',(
                        dict(rgb={cam:c['rgb'] for cam,c in packet['cameras'].items()}),dict(timestamp=stamp)))
                    (folder.parent/f'done_{i:03d}.json').write_text(json.dumps(dict(status='ok')))
        report=dict(status='completed',trace=[{}]*steps,observations=observations,
            control_actions_executed=True,recorded_future_ego_used_for_closed_motion=False,formal_evaluation_admitted=False)
        if kwargs:
            report.update(inference_service='resident_across_driving_episodes',worker_ready={'uniad_model_loads':1},
                          driver_state_initial={'checks':{'fake_contract_test':True}})
        (output/'run_report.json').write_text(json.dumps(report))

    def options(self,name):
        return argparse.Namespace(output=self.root/name,steps=2,native_edit=None,edit_command=None,
            articulation_report=None,verify_edit_lifecycle=False)

    def test_default_entry_retains_independent_path_and_restores_harness_hooks(self):
        original=harness.SceneSession
        result=runner.run_episode(self.options('default'))
        self.assertTrue(result['passed']);self.assertEqual(self.kwargs,[{}])
        self.assertIs(harness.SceneSession,original)

    def test_rotation_uses_unified_edit_probe_weather_receipts_and_reset_audit(self):
        import torch
        original_init=WeatherBackend.__init__
        backends=[]
        def init(backend,spec,output):
            original_init(backend,spec,output)
            model=backend.trainer.models['RigidNodes']
            model.instances_quats=torch.zeros(90,1,4);model.instances_quats[...,0]=1
            backend.env.models['RigidNodes']['instances_quats']=model.instances_quats
            backends.append(backend)
        def probe(backend,edit,output,render_clear,apply_edit):
            receipt=apply_edit(edit)
            self.assertEqual(receipt['command']['operation'],'rotate')
            self.assertGreater(float(backend.trainer.models['RigidNodes'].instances_quats[80,0,3]),0.)
            return dict(passed=True)
        command=self.root/'rotate.json'
        command.write_text(json.dumps(dict(schema='scene_edit_v1',mode='diagnostic_preview',
            operation='rotate',object_id='native:RigidNodes:0',yaw_degrees=15.)))
        body=self.root/'body.json'
        body.write_text(json.dumps(dict(schema='dynamic_articulation_acceptance_v1',passed=True,
            groups={g:dict(passed=True) for g in ('SMPLNodes','DeformableNodes')})))
        options=self.options('rotation');options.edit_command=command
        options.articulation_report=body;options.verify_edit_lifecycle=True
        with patch.object(WeatherBackend,'__init__',init),patch.object(runner,'probe_native_rotation',side_effect=probe) as called:
            result=runner.run_episode(options)
        self.assertTrue(result['passed']);called.assert_called_once()
        self.assertTrue(result['checks']['native_edit_revision_consistent'])
        audit=_hugsim_expand(json.loads((options.output/'unified_reset_audit.json').read_text()))[0]
        self.assertTrue(audit['native_rotations_restored'])
        q=backends[0].trainer.models['RigidNodes'].instances_quats
        self.assertTrue(torch.all(q[...,0]==1));self.assertTrue(torch.all(q[...,1:]==0))
        self.assertEqual(_hugsim_expand(json.loads((options.output/'unified_edit_journal.json').read_text()))[0]['command']['operation'],'rotate')

    def test_same_entry_passes_service_and_reuses_reset_scene_for_next_episode(self):
        with SharedWeatherScene(self.root/'host',backend_factory=WeatherBackend) as host:
            service=object()
            for name in ('one','two'):
                options=self.options(name);options.verify_edit_lifecycle=True
                result=runner.run_episode(options,inference_service=service,scene_host=host)
                self.assertTrue(result['passed']);self.assertFalse(host.active)
                self.assertIs(self.kwargs[-1]['inference_service'],service)
            self.assertEqual(host.loads,1)
            self.assertTrue(host.transitions[1]['same_backend_reset'])


if __name__=='__main__':unittest.main()
