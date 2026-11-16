"""CPU tests for the framework-backend session shim (no GPU)."""

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

import sys
import unittest
pass  # Path is provided by the portable boundary above

HERE = Path(__file__).resolve().parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
for p in (str(HERE), str(ROOT / 'integrations/sr-hugsim')):
    if p not in sys.path:
        sys.path.insert(0, p)

import scene_api_fwbackend as saf


class TestShim(unittest.TestCase):
    def test_reexport_completeness(self):
        missing = [n for n in dir(saf._real)
                   if not n.startswith('_') and n != 'SceneSession' and not hasattr(saf, n)]
        self.assertEqual(missing, [])

    def test_default_factory_is_framework_backend(self):
        # 不构造（构造需 GPU）；只验证默认工厂指向
        import backend as fb
        seen = {}
        class Probe(saf.SceneSession):
            def __init__(self, backend_factory=None):
                seen['factory'] = backend_factory
                # 不调 super().__init__（避免默认工厂导入即触发 GPU 依赖）
        # 直接走 shim 的 __init__ 逻辑但拦截
        orig = fb.FrameworkSceneBackend
        try:
            class FakeBackend:
                def __init__(self, spec): pass
            fb.FrameworkSceneBackend = FakeBackend
            inst = saf.SceneSession.__new__(saf.SceneSession)
            saf.SceneSession.__init__(inst)
            from scene_api import SceneSession as _  # noqa
        finally:
            fb.FrameworkSceneBackend = orig
        self.assertIs(seen.get('factory') or getattr(inst, 'factory', None), FakeBackend)


if __name__ == '__main__':
    unittest.main(verbosity=2)
