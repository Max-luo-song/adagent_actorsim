"""Record actual stage outcomes; skipped postprocessing never means success."""

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

import json,subprocess,sys
pass  # Path is provided by the portable boundary above

def finish_run(out, simulation_exit, *, script, cwd, env):
    out=Path(out);run=out/'closed_loop';report=run/'run_report.json';summary=run/'analysis/summary.json'
    result=dict(simulation_exit=simulation_exit,postprocess_exit=None,
                postprocess_status='skipped_missing_report',summary=None)
    if report.is_file():
        try:
            process=subprocess.run([sys.executable,str(script),str(run)],cwd=cwd,env=env,timeout=120,check=False)
            result['postprocess_exit']=process.returncode
            if process.returncode:
                result['postprocess_status']='failed'
            elif not summary.is_file():
                result['postprocess_status']='failed_missing_summary'
            else:
                _hugsim_expand(json.loads(summary.read_text()))
                result.update(postprocess_status='completed',summary=str(summary))
        except Exception as exc:
            result.update(postprocess_status='failed',postprocess_error=repr(exc))
    result['exit_code']=simulation_exit or (0 if result['postprocess_status']=='completed' else 1)
    (out/'launch_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Simulation exit:',simulation_exit,'postprocess:',result['postprocess_status'],
          'postprocess exit:',result['postprocess_exit'],'summary:',result['summary'],flush=True)
    return result
