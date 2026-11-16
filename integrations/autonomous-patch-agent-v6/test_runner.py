"""Run causal tests and pin the actual controller import; failures are data."""

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

import argparse,importlib.util,sys,time,unittest
pass  # Path is provided by the portable boundary above
from core import ROOT,asset,write

def run(code,files,output):
 code=Path(code).resolve();sys.path[:0]=[str(code/'tests'),str(code),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
 # Use the same reviewed import bootstrap as registered controller regressions.
 from common import configure
 configure()
 # configure prepends framework paths; keep the tested controller authoritative.
 sys.path[:0]=[str(code/'tests'),str(code)]
 loader=unittest.TestLoader();suite=unittest.TestSuite()
 for i,file in enumerate(files):
  spec=importlib.util.spec_from_file_location('_registered_causal_'+str(i),file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);suite.addTests(loader.loadTestsFromModule(module))
 began=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
 import joint_safety
 actual=Path(joint_safety.__file__).resolve()
 if actual!=code/'joint_safety.py':raise ValueError('Tests imported the wrong controller: '+str(actual))
 write(output,dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),expected_failures=len(result.expectedFailures),unexpected_successes=len(result.unexpectedSuccesses),passed=result.wasSuccessful(),controller=asset(actual),seconds=time.monotonic()-began,GPU_executed=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--test',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.code,a.test,a.output)
