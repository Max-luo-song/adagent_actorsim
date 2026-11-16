"""Resolve release-local modules independently of the driver's sys.path changes."""

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

import hashlib,importlib.util,sys
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent

def load_release_module(name):
 if name not in ('postprocess','native_scores'):raise ValueError('Unexpected local release module')
 file=HERE/(name+'.py');key='_cutin_release_'+hashlib.sha256(str(file).encode()).hexdigest()[:16]+'_'+name
 if key in sys.modules:
  module=sys.modules[key]
  if Path(module.__file__).resolve()!=file:raise RuntimeError('Local module binding changed')
  return module
 spec=importlib.util.spec_from_file_location(key,file);module=importlib.util.module_from_spec(spec);sys.modules[key]=module
 try:spec.loader.exec_module(module)
 except BaseException:sys.modules.pop(key,None);raise
 return module

def finish_report(output,report):
 """A postprocessing error must never hide the driving/recovery error."""
 from resume_store import atomic_json
 if report.get('error'):print('PRIMARY SIMULATION ERROR:',report['error'],flush=True)
 try:return load_release_module('postprocess').finish(output)
 except Exception as exc:
  report['postprocess_error']=repr(exc);atomic_json(Path(output)/'run_report.json',report)
  summary=dict(status=report.get('status'),controls=len(report.get('trace',[])),error=report.get('error'),postprocess_error=repr(exc),video=None)
  Path(output,'analysis').mkdir(exist_ok=True);atomic_json(Path(output)/'analysis/summary.json',summary);return summary
