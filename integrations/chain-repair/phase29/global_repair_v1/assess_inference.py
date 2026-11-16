
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

import json,pickle
pass  # Path is provided by the portable boundary above
from .screening import retained_reference_check

def assess(output):
    output=Path(output);frames=[];references=0;changes=0
    for folder in sorted((output/'baseline').glob('step_[0-9][0-9][0-9]')):
        with (folder/'observation.pkl').open('rb') as f:_,info=pickle.load(f)
        before=_hugsim_expand(json.loads((folder/'scene_reliability.json').read_text()))
        after=_hugsim_expand(json.loads((output/'global'/folder.name/'scene_reliability.json').read_text()))
        check=retained_reference_check(before,after,info)
        references+=check['reference_observations'];changes+=len(check['changed_or_removed'])
        frames.append(dict(frame=folder.name,**check))
    return dict(frames=len(frames),visible_reference_observations=references,
                visible_reference_changed_or_removed=changes,details=frames,
                physical_existence_certified=False)
