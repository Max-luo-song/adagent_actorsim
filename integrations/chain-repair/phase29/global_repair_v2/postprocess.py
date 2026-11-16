
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

import json,sys
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from global_repair_v1 import postprocess as original

def main():
    original.main()
    run=Path(sys.argv[1]);path=run/'analysis/summary.json';result=_hugsim_expand(json.loads(path.read_text()));events=[]
    for folder in sorted((run/'uniad_worker').glob('step_[0-9][0-9][0-9]')):
        p=folder/'guard_memory_report.json'
        if not p.exists():continue
        g=_hugsim_expand(json.loads(p.read_text())).get('unknown_visibility_geometry',{})
        if g.get('generation_policy')=='support_bounded_unknown_boundary_v2':
            events.append(dict(step=folder.name,hard_boundary_points=g['points'],full_unresolved_cells=g['full_unresolved_cells'],epistemic_only_points=g['epistemic_only_points']))
    result.update(unknown_boundary_events=events,full_unknown_hypotheses_removed=False,
        measured_depth_and_occupancy_checks_preserved=True,global_repair_complete=False)
    path.write_text(json.dumps(result,indent=2)+'\n');print('Unknown boundary frames:',len(events),flush=True)
if __name__=='__main__':main()
