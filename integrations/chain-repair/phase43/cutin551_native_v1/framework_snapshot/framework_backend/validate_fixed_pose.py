"""Fixed-pose validation: framework backend render at the recorded frame-80
ego pose must reproduce the s1 dataset-render frame (same pose pipeline).

If this matches, the online render path (extrinsics, conventions, intrinsics)
is verified for closed-loop use. Output: per-camera max/mean abs diff + PNG
pairs. Diagnostic, uncertified.
"""

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
import sys
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
P11 = JOB / 'chain_repair/phase11'
for p in (str(HERE), str(P11), str(ROOT / 'integrations/sr-hugsim')):
    if p not in sys.path:
        sys.path.insert(0, p)

from scene_api import SceneSpec  # real one, read-only use
from backend import FrameworkSceneBackend

MANIFEST = JOB / 'chain_repair/phase4/dynamic552_frame80/scene_manifest.json'
S1 = Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/videos_s1_baseline/full_set_40000_rgbs'))
OUT = HERE / 'fixed_pose_validation'
FRAME = 80


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    OUT.mkdir(parents=True)
    spec = SceneSpec.read(str(MANIFEST))
    inputs = Path(spec.source['inputs'])
    raw_start = spec.source['start_timestep']
    poses = np.stack([np.loadtxt(inputs / 'ego_pose' / f'{raw_start + i:03d}.txt')
                      for i in range(spec.frames)])
    relative = np.linalg.inv(poses[0])[None] @ poses
    pose = relative[FRAME]

    backend = FrameworkSceneBackend(spec)
    try:
        raw = backend.render(pose, FRAME / spec.fps)
    finally:
        backend.close()

    from PIL import Image
    report = dict(frame=FRAME, pose='recorded relative[80]',
                  reference='s1 dataset render (full_set_40000)', cameras={})
    cam_ids = {'CAM_FRONT': 0, 'CAM_FRONT_LEFT': 1, 'CAM_FRONT_RIGHT': 2}
    for name, cam_id in cam_ids.items():
        got = np.asarray(raw['cameras'][name]['rgb'])
        ref = np.asarray(Image.open(S1 / f'{FRAME:06d}_{cam_id:03d}.png').convert('RGB'))
        if got.shape != ref.shape:
            report['cameras'][name] = dict(error=f'shape {got.shape} vs {ref.shape}')
            continue
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        report['cameras'][name] = dict(max_abs=int(d.max()), mean_abs=float(d.mean()),
                                       pixel_identical=bool(d.max() == 0))
        pair = Image.new('RGB', (ref.shape[1] * 2 + 8, ref.shape[0]), (0, 0, 0))
        pair.paste(Image.fromarray(ref), (0, 0))
        pair.paste(Image.fromarray(got), (ref.shape[1] + 8, 0))
        pair.save(OUT / f'{name}_ref_vs_online.png')
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
