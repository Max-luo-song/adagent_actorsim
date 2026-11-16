
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

from contextlib import redirect_stdout
import io
pass  # Path is provided by the portable boundary above
import sys
import tempfile
import unittest

from run_edit_completion import run_stage


class StageTests(unittest.TestCase):
    def test_worker_progress_reaches_terminal_and_full_log_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stage.log'
            output = io.StringIO()
            with redirect_stdout(output):
                run_stage([sys.executable, '-u', '-c',
                    'print("internal detail"); print("Weather startup: stage=ready")'], path)
            self.assertIn('Weather startup: stage=ready', output.getvalue())
            self.assertNotIn('internal detail', output.getvalue())
            self.assertIn('internal detail', path.read_text())

    def test_failure_surfaces_actual_worker_error_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rigid.log'
            with self.assertRaisesRegex(RuntimeError, 'weather startup failed'):
                run_stage([sys.executable, '-u', '-c', 'raise RuntimeError("weather startup failed")'], path)
            with self.assertRaises(FileExistsError):
                run_stage([sys.executable, '-c', 'pass'], path)


if __name__ == '__main__': unittest.main()
