"""Audit actual saved previews and sensor-plane placement; never certify edits."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from edit_trust import _affected_roi, EditPolicy, assert_edit_evaluable, EditTrustError
from asset_placement import terrain_aligned_pose


def audit(preview, output):
    preview, output = Path(preview).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    asset = json.loads((preview/'asset_report.json').read_text())
    sensor = json.loads(Path(asset['sensor_audit']).read_text())
    sequence = json.loads((preview/'sequence_report.json').read_text())
    policy = EditPolicy()
    old = np.asarray(asset['placements'][0]['asset_to_source'])
    width, length, height = asset['asset_wlh']
    # Legacy previews had no explicit bottom_y; their insertion was horizontal.
    bottom = asset.get('local_bottom_y', old[2, 3]-sensor['proposed_anchor_source_xyz'][2])
    contacts = np.array([[x, bottom, z, 1.] for x in (-length/2, length/2) for z in (-width/2, width/2)])
    corners = np.array([[x, y, z, 1.] for x in (-length/2, length/2)
                        for y in (bottom-height, bottom) for z in (-width/2, width/2)])
    new = terrain_aligned_pose(sensor['proposed_anchor_source_xyz'], sensor['heading_source_radians'],
                                sensor['ground_fit'], bottom)
    fit = sensor['ground_fit']
    def residual(pose):
        points = (contacts@pose.T)[:, :3]
        ground = (points[:, :2]-fit['origin_xy'])@np.asarray(fit['plane'][:2])+fit['plane'][2]
        return (points[:, 2]-ground).tolist()
    affected = (corners@old.T)[:, :3]
    request = dict(kind='insert', new_bounds=np.stack([affected.min(0), affected.max(0)]).tolist())
    records = []
    for frame in sequence['frames']:
        folder = Path(frame['packet']).parent
        with np.load(folder/'render_evidence.npz', allow_pickle=False) as data:
            for camera in ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'):
                depth, alpha = data[camera+'__depth'], data[camera+'__alpha']
                try:
                    roi = _affected_roi(request, data[camera+'__K'], data[camera+'__camera_to_world'], depth.shape, policy.halo_m)
                except EditTrustError as exc:
                    records.append(dict(frame=frame['frame'], camera=camera, affected_roi_valid=False,
                                        reason=str(exc), independent_reference_compared=False))
                    continue
                valid = np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha >= policy.min_alpha) & (alpha <= 1)
                records.append(dict(frame=frame['frame'], camera=camera, affected_roi_pixels=int(roi.sum()),
                                    valid_opaque_fraction=float(valid[roi].mean()) if roi.any() else None,
                                    independent_reference_compared=False))
    admission_rejected = False
    try:
        assert_edit_evaluable(sequence)
    except EditTrustError:
        admission_rejected = True
    floating = new.copy(); floating[2, 3] += .3
    penetrating = new.copy(); penetrating[2, 3] -= .3
    report = dict(scene=sensor['scene'], source_preview=str(preview), sensor_audit=asset['sensor_audit'],
                  baseline_proxy_contact_residuals_m=residual(old),
                  aligned_proxy_contact_residuals_m=residual(new),
                  proposed_asset_to_source=new.tolist(),
                  floating_proxy_residuals_m=residual(floating), penetrating_proxy_residuals_m=residual(penetrating),
                  measured_render_rois=records, unverified_edit_admission_rejected=admission_rejected,
                  geometry_model='declared size and quantile-bottom proxy; not observed wheel contacts',
                  real_sensor_local_contacts_validated=False, multiview_consistency_certified=False,
                  edited_vehicle_detection_validated=False, n1_edit_certified=False)
    (output/'preview_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit(args.preview, args.output)
    print(json.dumps(report, indent=2))
