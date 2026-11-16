
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
import unittest
from unittest.mock import patch
import service

class GPUQueueTests(unittest.TestCase):
    def test_busy_race_waits_then_retries_without_erasing_failed_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);flag=root/'next'
            script="import pathlib,sys;p=pathlib.Path(sys.argv[1]);busy=not p.exists();p.touch();print('GPU lock busy' if busy else 'done');sys.exit(75 if busy else 0)"
            args=[sys.executable,'-B','-c',script,str(flag)]
            with patch.object(service,'OUT',root),patch.object(service,'stopped',return_value=False),patch.object(service.time,'sleep',return_value=None):
                self.assertFalse(service.command('pair','original_off',args,30))
                original=(root/'pair/jobs/original_off/receipt.json').read_bytes()
                self.assertEqual(service.read(root/'pair/status.json')['stage'],'waiting_gpu')
                self.assertTrue(service.command('pair','original_off',args,30))
                self.assertEqual(original,(root/'pair/jobs/original_off/receipt.json').read_bytes())
                self.assertTrue((root/'pair/jobs/original_off_retry001/receipt.json').exists())
                self.assertTrue(service.command('pair','original_off',args,30))
                self.assertFalse((root/'pair/jobs/original_off_retry002').exists())
    def test_non_busy_exit_75_remains_a_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(service,'OUT',Path(temp)),patch.object(service,'stopped',return_value=False),patch.object(service.time,'sleep',return_value=None):
                with self.assertRaises(RuntimeError):service.command('pair','x',[sys.executable,'-c',"print('other failure');raise SystemExit(75)"],30)
    def test_ambiguous_partial_episode_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'pair/jobs/x';folder.mkdir(parents=True);(folder/'output.log').write_text('partial')
            with patch.object(service,'OUT',root):
                with self.assertRaises(RuntimeError):service.command('pair','x',[sys.executable,'-c','pass'],30)
