"""Actual CPU constructor preflight using the candidate's registered runtime."""

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

import argparse,json,sys
pass  # Path is provided by the portable boundary above
from core import ROOT,write
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.code),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
 from common import configure
 configure();sys.path.insert(0,str(a.code))
 if not (a.code/'smpl_runtime.py').exists():
  from core import verify,read
  import importlib.util
  selected=read(ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/global_patch.json');verify(selected['runtime_support'])
  if str(a.code.resolve())!=selected['policy']['code']:raise ValueError('Missing runtime support for an unregistered policy')
  sp=importlib.util.spec_from_file_location('smpl_runtime',selected['runtime_support']['path']);m=importlib.util.module_from_spec(sp);sys.modules['smpl_runtime']=m;sp.loader.exec_module(m)
 from smpl_runtime import preflight
 write(a.output,preflight());print('Actual canonical SMPL constructor passed; no GPU scene loaded',flush=True)
