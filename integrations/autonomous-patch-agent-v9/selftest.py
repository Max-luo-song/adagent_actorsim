"""Only synthetic datasets; never discover/read actual private validation results."""

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

import sys,unittest
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;sys.path[:0]=[str(LOCAL),str(LOCAL/'tests'),str(LOCAL/'tests_hardening')]
suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(LOCAL/'tests_hardening')),*[unittest.defaultTestLoader.loadTestsFromName(n) for n in ['test_governance','test_convergence_rotation','test_validation_lifecycle','test_blind_validation','test_generalization']]])
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(not r.wasSuccessful())
