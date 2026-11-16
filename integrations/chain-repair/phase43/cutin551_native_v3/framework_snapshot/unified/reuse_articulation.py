"""Reuse completed fixed-scene body evidence after checking its recorded hashes."""

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

import argparse
import hashlib
import json
pass  # Path is provided by the portable boundary above

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent


def verify(run):
    run = Path(run).resolve()
    report = run / 'articulation/report.json'
    data = _hugsim_expand(json.loads(report.read_text()))
    if data.get('passed') is not True or set(data.get('groups', {})) != {'SMPLNodes', 'DeformableNodes'}:
        raise ValueError('completed body evidence required')
    driver_sources = _hugsim_expand(json.loads((run / 'joint/implementation.json').read_text()))
    original_sources = _hugsim_expand(json.loads((run / 'implementation.json').read_text()))['files']
    required = [report, BRIDGE / 'smpl_dynamic/validate_articulation.py']
    sources = dict(original_sources, **driver_sources)
    required += list((BRIDGE / 'smpl_dynamic').glob('*.py'))
    required += list((BRIDGE.parent / 'framework_repo/models/nodes').glob('*.py'))
    checks = {}
    for path in required:
        key = str(path.resolve())
        expected = sources.get(key)
        checks[key] = expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    if not all(checks.values()):
        raise ValueError('body evidence or body implementation changed; do not reuse')
    return dict(passed=True, body_report=str(report), source_checks=checks,
        scope='existing scene552 frame80/85 body experiment; verified recorded body-source subset',
        no_gpu_rerender=True, formal_evaluation_admitted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.run)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2))
