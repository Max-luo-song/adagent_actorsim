"""CPU-only comparison of the actual preceding champion and candidate episodes."""

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

import argparse,os,sys
pass  # Path is provided by the portable boundary above
from core import ROOT,read
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.code),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
 os.environ.setdefault('MPLCONFIGDIR',str(a.output/'presentation_cache'))
 import compare_native
 if Path(compare_native.__file__).resolve()!=a.code.resolve()/'compare_native.py':raise ValueError('Unexpected comparison renderer')
 reports={'off':read(a.baseline/'run_report.json'),'on':read(a.candidate/'run_report.json')}
 for r in reports.values():
  if r['status']!='completed' or len(r['observations'])!=189:raise ValueError('Need complete actual episodes')
 print(compare_native.render_pair(a.output,a.baseline,a.candidate,reports))
