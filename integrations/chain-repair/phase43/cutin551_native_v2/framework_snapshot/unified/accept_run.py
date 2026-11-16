"""Refuse to equate a completed baseline or a video with edited-chain acceptance."""

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
import numpy as np
import sys
pass  # Path is provided by the portable boundary above
out = Path(sys.argv[1])
a = _hugsim_expand(json.loads((out/'articulation/report.json').read_text()))
r = _hugsim_expand(json.loads((out/'edited_closed_loop/run_report.json').read_text()))
observations=_hugsim_expand(json.loads((out/'edited_closed_loop/render_observations.json').read_text()))
harness_obs=r.get('observations',[])
poses_match=(len(observations)==len(harness_obs)==5 and all(
    np.allclose(o['ego_to_world'],h['closed_ego_to_world'],atol=1e-6) and
    abs(o['timestamp']-h['timestamp'])<1e-6
    for o,h in zip(observations,harness_obs)))
mounts_match=bool(observations) and all(np.allclose(
    np.asarray(o['ego_to_world']) @ np.asarray(c['camera_to_ego']),c['camera_to_world'],atol=1e-6)
    for o in observations for c in o['cameras'].values())
images_change=bool(observations) and any(len({o['cameras'][cam]['rgb_sha256'] for o in observations})>1
    for cam in observations[0]['cameras'])
checks = dict(executed_poses_match_renders=poses_match,
              camera_mounts_match=mounts_match, images_change=images_change,
              edit_revision_consistent=bool(observations) and all(o['edit_revision']==1 for o in observations),
articulation=a.get('passed') is True,
              completed=r.get('status')=='completed',
              five_executed_steps=len(r.get('trace',[]))==5,
              native_edit=bool(r.get('native_edits')),
              executed=r.get('control_actions_executed') is True,
              no_formal_admission=r.get('formal_evaluation_admitted') is False,
              no_recorded_future=r.get('recorded_future_ego_used_for_closed_motion') is False)
result = dict(checks=checks, passed=all(checks.values()),
              formal_evaluation_admitted=False, diagnostic_uncertified=True,
              safety_equivalence_to_mpc=False)
(out/'acceptance.json').write_text(json.dumps(result,indent=2))
if not result['passed']: raise SystemExit('edited-chain acceptance failed; inspect acceptance.json')
