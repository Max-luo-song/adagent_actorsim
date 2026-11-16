"""Matched-geometry ablation: rescale an exported asset to another support wlh.

Diagnostic ablation only, required when two compared assets differ by >10% in
any support dimension (project rule). This is NOT the true-size asset:
geometry means/scales are scaled per asset-frame axis; SH appearance and
opacity are unchanged; trajectory poses are recomputed from the rescaled
calibration recentering. Clearly labelled in provenance.
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

import argparse
import json
import sys
pass  # Path is provided by the portable boundary above

import numpy as np
import torch

BASE = Path(_hugsim_expand('${HUGSIM_ROOT}'))
sys.path.insert(0, str(BASE/'waymo23-native-import'))
sys.path.insert(0, str(BASE/'integrations/sr-hugsim'))
import asset_calibration  # noqa: E402
from waymo_adapter import rigid  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-asset', required=True)
    p.add_argument('--match-calibration', required=True,
                   help='calibration JSON of the asset whose support wlh is matched')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    source = Path(args.source_asset).resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Ablation asset directory is immutable: '+str(output))
    output.mkdir(parents=True)

    params, step = torch.load(source/'gs.pth', map_location='cpu', weights_only=False)
    src_cal = asset_calibration.load(source/'asset_calibration.json', source)
    target_cal = _hugsim_expand(json.loads(Path(args.match_calibration).read_text()))
    src_wlh = np.asarray(src_cal['support']['wlh'])      # [width, length, height]
    tgt_wlh = np.asarray(target_cal['support']['wlh'])
    # asset frame: x=length, y=down(height), z=lateral(width)
    factor = np.array([tgt_wlh[1]/src_wlh[1], tgt_wlh[2]/src_wlh[2], tgt_wlh[0]/src_wlh[0]])
    rel = np.abs(tgt_wlh-src_wlh)/src_wlh
    if not (rel > 0.10).any():
        raise ValueError('Ablation only justified when a support dimension differs by >10%')

    params[1] = params[1]*torch.tensor(factor, dtype=params[1].dtype)          # means
    params[5] = params[5]+torch.tensor(np.log(factor), dtype=params[5].dtype)  # log scales
    torch.save((params, step), output/'gs.pth')
    (output/'wlh.json').write_text((source/'wlh.json').read_text())

    calibration = asset_calibration.create(output, output/'asset_calibration.json')
    support = calibration['support']
    c_new = np.array([support['raw_recenter_xz'][0], 0., support['raw_recenter_xz'][1]])
    c_old = np.array([src_cal['support']['raw_recenter_xz'][0], 0., src_cal['support']['raw_recenter_xz'][1]])
    # recentered scaled point p' = F@(p-c_old) = F@p - c_new; the world target of
    # the ORIGINAL unscaled recentered point is unchanged when the pose shifts by
    # t -> t + R@(c_new - F@c_old). Recompute the trajectory with that offset so
    # the ablation asset keeps the same world footprint centre trajectory.
    traj = np.load(source/'trajectory.npz')
    poses = traj['asset_to_world'].copy()
    for f in range(len(poses)):
        P = rigid(poses[f])
        poses[f, :3, 3] = P[:3, 3] + P[:3, :3] @ (c_new - factor*c_old)
    np.savez_compressed(output/'trajectory.npz', asset_to_world=poses,
                        visible=traj['visible'], heading_source_radians=traj['heading_source_radians'],
                        fps=float(traj['fps']), frame_period_s=float(traj['frame_period_s']))

    provenance = _hugsim_expand(json.loads((source/'provenance.json').read_text()))
    provenance.update(schema='sr_rigid_export_provenance_v1',
                      matched_geometry_ablation=True, true_size=False,
                      ablation_base_asset=str(source),
                      support_scale_factors_xyz_length_height_width=factor.tolist(),
                      matched_support_wlh=tgt_wlh.tolist(),
                      matched_calibration=args.match_calibration,
                      diagnostic_uncertified=True, n1_certified=False,
                      formal_evaluation_admitted=False)
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(ablation=str(output), factors=factor.tolist(),
                          new_support_wlh=calibration['support']['wlh']), indent=2))


if __name__ == '__main__':
    main()
