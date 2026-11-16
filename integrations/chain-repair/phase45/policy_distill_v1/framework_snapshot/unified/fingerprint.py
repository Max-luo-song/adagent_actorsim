
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

import hashlib
import json
import sys
pass  # Path is provided by the portable boundary above
here = Path(__file__).resolve().parent
bridge = here.parent
files = [p for d in (here, bridge/'smpl_dynamic', bridge/'framework_backend', bridge/'fast_control')
         for p in d.glob('*.py')]
files += [here/'run_phase26.sh', here/'native_pedestrian_edit.json',
          bridge/'framework_backend/requests/fastprobe_fwbackend.json']
files += list((bridge.parent/'framework_repo/models/nodes').glob('*.py'))
result = dict(diagnostic_uncertified=True, formal_evaluation_admitted=False,
              n1_certified=False, training=False,
              files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(Path(sys.argv[1])/'implementation.json').write_text(json.dumps(result,indent=2))
