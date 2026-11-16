
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

import json
pass  # Path is provided by the portable boundary above
import tempfile
from types import SimpleNamespace
import unittest

from runtime_backend import RuntimeWeatherBackend


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.backend=RuntimeWeatherBackend.__new__(RuntimeWeatherBackend)
        self.backend._last=None;self.backend.weather_epoch=4
        self.backend.weather_records=[dict(old_episode=True)]
        self.backend.client=SimpleNamespace(request=lambda request:{'reset':True})

    def test_weather_worker_reused_but_output_and_history_are_fresh(self):
        client=self.backend.client;old_records=self.backend.weather_records
        self.backend.begin_weather_episode(self.root/'weather')
        self.assertIs(self.backend.client,client)
        self.assertEqual(self.backend.weather_epoch,5)
        self.assertEqual(self.backend.weather_records,[])
        self.assertEqual(old_records,[dict(old_episode=True)])
        self.assertTrue(self.backend.last_weather_reset_verified)
        self.assertEqual(_hugsim_expand(json.loads((self.root/'weather/report.json').read_text()))['records'],[])
        with self.assertRaises(FileExistsError):self.backend.begin_weather_episode(self.root/'weather')

    def test_failed_worker_reset_does_not_create_episode_artifacts(self):
        self.backend.client.request=lambda request:{'reset':False}
        with self.assertRaises(RuntimeError):self.backend.begin_weather_episode(self.root/'weather')
        self.assertFalse((self.root/'weather').exists())
        self.assertTrue(self.backend.weather_records)

    def test_cannot_change_output_mid_observation(self):
        self.backend._last=(8.,None)
        with self.assertRaises(RuntimeError):self.backend.begin_weather_episode(self.root/'weather')
        self.assertEqual(self.backend.weather_epoch,4)


if __name__=='__main__':unittest.main()
