
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

pass  # Path is provided by the portable boundary above
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared_weather_scene import SharedWeatherScene,saf
from test_lifecycle_probe import Backend
from test_edit_api import command


class WeatherBackend(Backend):
    @property
    def weather_epoch(self):return self.epoch
    def begin_weather_episode(self,output):
        if self._last is not None:raise RuntimeError('not reset')
        Path(output).mkdir(parents=True,exist_ok=False)
        self.clear_observation_history();self.weather_records=[]


class SharedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        def spec(name):
            return SimpleNamespace(source=dict(scene=552 if name=='first' else 551,checkpoint=name),
                fps=10.,frames=90,scale=1.,camera_order=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'),
                summary=lambda:{})
        reader=patch.object(saf.SceneSpec,'read',side_effect=spec);reader.start();self.addCleanup(reader.stop)
        self.host=SharedWeatherScene(self.root/'host',backend_factory=lambda spec,out:WeatherBackend(spec))
        self.addCleanup(self.host.close)

    def test_reset_reuses_session_backend_and_scene_switch_replaces_only_backend(self):
        session,_=self.host.acquire('first',self.root/'one/weather')
        backend=session.backend;old_editor=session.editor
        session.edit(command('RigidNodes'));session.reset();self.host.release()
        next_session,transition=self.host.acquire('first',self.root/'two/weather')
        self.assertIs(next_session,session);self.assertIs(next_session.backend,backend)
        self.assertTrue(transition['same_backend_reset'])
        with self.assertRaises(RuntimeError):old_editor.objects()
        next_session.reset();self.host.release()
        new_session,transition=self.host.acquire('other',self.root/'three/weather')
        self.assertIs(new_session,session);self.assertIsNot(new_session.backend,backend)
        self.assertTrue(backend.closed);self.assertTrue(transition['scene_switched'])
        self.assertEqual(self.host.loads,2)

    def test_dirty_release_prevents_further_reuse(self):
        session,_=self.host.acquire('first',self.root/'one/weather')
        session.edit(command('RigidNodes'))
        with self.assertRaisesRegex(RuntimeError,'did not reset'):self.host.release()
        with self.assertRaises(RuntimeError):self.host.acquire('first',self.root/'two/weather')

    def test_concurrent_acquire_rejected(self):
        self.host.acquire('first',self.root/'one/weather')
        with self.assertRaises(RuntimeError):self.host.acquire('other',self.root/'two/weather')


if __name__=='__main__':unittest.main()
