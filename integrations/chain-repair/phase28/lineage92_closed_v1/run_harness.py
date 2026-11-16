"""Reuse the pinned heading-corrected harness; fresh record only."""

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
import sys
HERE=Path(__file__).resolve().parent
P28=HERE.parent
sys.path.insert(0,str(P28))
import heading_scene_run_v1 as harness

def main():
    request=_hugsim_expand(json.loads(Path(sys.argv[1]).read_text()))
    if request['mode']!='record':raise ValueError('Fresh run required')
    harness.WORKER=HERE/'worker.py'
    harness._PATCH_MODULE_PATHS.extend([p for p in HERE.iterdir() if p.suffix in ('.py','.json','.md','.sh')])
    harness._PATCH_MODULE_PATHS.extend(P28/'heading_fix_v1'/name for name in ('origin_worker.py','origin_state.py'))
    original_loop=harness._control_loop
    def loop(ctx,output,session,report,*args,**kwargs):
        report.update(experiment='posthoc_neural_lineage92_removal',generalizable_fix=False,
            diagnostic_intervention=dict(neural_track_id=92,stage='before_external_memory',
                raw_model_outputs_modified=False,posthoc_selected=True),
            termination_policy='117 steps to source end / proxy overlap / 18000s wall cap; pass and return assessed posthoc; no low-speed early cutoff')
        return original_loop(ctx,output,session,report,*args,**kwargs)
    harness._control_loop=loop
    harness.run_record(request)

if __name__=='__main__':main()
