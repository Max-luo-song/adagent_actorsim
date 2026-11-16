"""Read-only local status; does not initialize models or schedule GPU work."""

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

import fcntl
import importlib.metadata as md
import json
import socket
pass  # Path is provided by the portable boundary above
here=Path(__file__).resolve().parent
bridge=here.parent
service=here.parents[5]
weather=bridge.parent/'framework_repo/WeatherEdit/background_editing/ckpts'
difix=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/Difix3D'))
import torch
packages={}
for name,version in [('diffusers','0.25.1'),('transformers','4.38.0'),('peft','0.9.0'),('huggingface-hub','0.25.1')]:
    try: found=md.version(name)
    except md.PackageNotFoundError: found=None
    packages[name]=dict(required=version,installed=found,match=found==version)
lock=service/'worker.lock'
try:
    with lock.open() as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); lock_status='available (released after inspection)'
        except BlockingIOError: lock_status='busy'
except OSError as exc: lock_status=str(exc)
print(json.dumps(dict(host=socket.gethostname(),cuda_available=torch.cuda.is_available(),
    worker_lock=lock_status,worker_heartbeat=_hugsim_expand(json.loads((service/'heartbeat.json').read_text())),
    weather={str(weather/p):(weather/p).exists() for p in ('sd-turbo','with_seg.pkl')},
    difix=dict(source=(difix/'src').is_dir(),weights=(difix/'difix/model_index.json').is_file(),packages=packages),
    existing_dynamic_render=(bridge/'smpl_dynamic/dynamic_node_validation/report.json').exists(),
    independent_articulation_reports=[str(p) for p in (here/'runs').glob('*/articulation/report.json')],
    formal_evaluation_admitted=False),indent=2))
