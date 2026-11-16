"""Exercise the actual pipe protocol with CPU subprocesses, no weather model."""

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
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from runtime_backend import WeatherClient


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def client(self, code):
        client = WeatherClient.__new__(WeatherClient)
        client.folder = self.folder
        client.startup_status = self.folder / 'startup_status.json'
        client.log = (self.folder / 'worker.log').open('w')
        client.process = subprocess.Popen([sys.executable, '-u', '-c', code],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=client.log, text=True, bufsize=1)
        self.addCleanup(client.close)
        return client

    def test_slow_start_can_use_separate_budget_and_request_stays_300(self):
        client = self.client('import time; time.sleep(.15); print(\'{"ready":true}\',flush=True)')
        self.assertTrue(client.receive(timeout=2, phase='startup')['ready'])
        self.assertEqual(WeatherClient.receive.__defaults__, (300, 'request'))

    def test_timeout_retains_loading_stage_and_elapsed_time(self):
        client = self.client('import time; time.sleep(10)')
        client.startup_status.write_text(json.dumps(dict(stage='construct_and_load_weather_model')))
        with self.assertRaisesRegex(TimeoutError, 'startup timed out.*construct_and_load_weather_model'):
            client.receive(timeout=.05, phase='startup')
        result = _hugsim_expand(json.loads((self.folder / 'timeout.json').read_text()))
        self.assertEqual(result['phase'], 'startup')
        self.assertGreaterEqual(result['elapsed_seconds'], .05)
        self.assertIsNone(result['worker_returncode'])

    def test_exited_worker_is_not_misreported_as_a_timeout(self):
        client = self.client('raise SystemExit(7)')
        with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
            client.receive(timeout=2, phase='startup')
        self.assertFalse((self.folder / 'timeout.json').exists())

    def test_constructor_uses_900_startup_default_and_accepts_override(self):
        for override, expected in [(None, 900), ('1200', 1200)]:
            env = {} if override is None else {'PHASE26_WEATHER_STARTUP_TIMEOUT': override}
            with self.subTest(override=override), patch.dict('os.environ', env, clear=True):
                child = subprocess.Popen([sys.executable, '-u', '-c', 'import time; time.sleep(10)'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1)
                with patch('runtime_backend.subprocess.Popen', return_value=child), patch.object(
                        WeatherClient, 'receive', return_value={'ready': True}) as receive:
                    client = WeatherClient(self.folder / str(expected))
                    try:
                        receive.assert_called_once_with(timeout=expected, phase='startup')
                    finally:
                        client.close()

    def test_invalid_timeout_rejected_before_launch(self):
        for value in ('0', '-1', 'nan', 'inf', 'bad'):
            with self.subTest(value=value), patch.dict('os.environ', {'PHASE26_WEATHER_STARTUP_TIMEOUT': value}):
                with patch('runtime_backend.subprocess.Popen') as launch:
                    with self.assertRaises(ValueError): WeatherClient(self.folder / 'unused')
                    launch.assert_not_called()


if __name__ == '__main__': unittest.main()
