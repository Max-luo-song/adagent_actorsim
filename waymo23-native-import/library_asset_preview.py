"""Diagnostic insertion of a library asset; never grants N1 admission."""
import json
from pathlib import Path

import numpy as np


def install(env, specification, output):
    import torch
    import gaussian_renderer as renderer
    from scene.obj_model import ObjModel
    from edit_trust_adapter import mark_unverified_edit

    asset = Path(specification["asset"])
    audit = json.loads(Path(specification["sensor_audit"]).read_text())
    params, iteration = torch.load(asset/"gs.pth", map_location="cuda", weights_only=False)
    degree, xyz, dc, rest, semantic, scaling, rotation, opacity, unused = params
    if int(degree) != 3:
        raise ValueError("Library preview currently requires degree-three assets")
    solid = torch.sigmoid(opacity).flatten() > .5
    if int(solid.sum()) < 100:
        raise ValueError("Insufficient opaque asset support")
    bounds = torch.quantile(xyz[solid], xyz.new_tensor([.005, .995]), dim=0)
    extent = (bounds[1]-bounds[0]).detach().cpu().numpy()
    wlh = np.asarray(json.loads((asset/"wlh.json").read_text()), dtype=float)
    if wlh.shape != (3,) or not np.all((extent/wlh[[1, 2, 0]] > .5) & (extent/wlh[[1, 2, 0]] < 2)):
        raise ValueError("Library dimensions disagree with Gaussian support")
    center = (bounds[0]+bounds[1])/2
    calibration_binding = None
    declared_wlh = wlh.copy()
    if specification.get("asset_calibration"):
        from asset_calibration import load as load_calibration, fingerprint
        calibration_path = Path(specification["asset_calibration"]).resolve()
        calibration = load_calibration(calibration_path, asset)
        support = calibration['support']
        wlh = np.asarray(support['wlh'], dtype=float)
        center = center.clone()
        center[0], center[2] = support['raw_recenter_xz']
        calibration_binding = dict(path=str(calibration_path), **fingerprint(calibration_path),
                                   reference_kind="gaussian_center_support_proxy", physical_certification=False)
    car = ObjModel(int(degree), feat_mutable=False)
    car.restore(params, None)
    car._xyz = xyz.clone()
    car._xyz[:, 0] -= center[0]
    car._xyz[:, 2] -= center[2]
    from asset_placement import supported_placement
    bottom_y = (float(support["bottom_y"]) if calibration_binding is not None else float(bounds[1, 1]))
    skin_path = Path(specification["sensor_audit"]).parent/'sensor_skin.npz'
    with np.load(skin_path, allow_pickle=False) as data:
        skin = dict(ground_points=data['ground_points'], occupied_points=data['occupied_points'],
                    source_id=str(skin_path.resolve()), source_kind='sensor')
    pose, placement_search = supported_placement(audit["proposed_anchor_source_xyz"], audit["heading_source_radians"],
                                                 audit["ground_fit"], bottom_y, wlh, skin)
    target = env.target.detach().cpu().numpy()
    native = target@pose
    asset_pose = xyz.new_tensor(native)
    radiance_frame = specification.get('radiance_frame', 'asset_local')
    if radiance_frame not in ('asset_local', 'legacy_world'):
        raise ValueError('Unsupported library radiance frame')
    from asset_radiance import sh_rotation
    matrix = sh_rotation(native[:3, :3], int(degree))
    if radiance_frame == 'asset_local':
        # Geometry and covariance already rotate in HUGSIM; rotate radiance too.
        # Keep DC exact and leave native scene objects/background untouched.
        coefficients = car.get_features.detach()
        transform = coefficients.new_tensor(matrix[1:, 1:])
        car._features_rest = torch.einsum('ij,njc->nic', transform, coefficients[:, 1:, :]).contiguous()
    original = renderer.render

    def render(*args, **kwargs):
        camera, pc = kwargs["viewpoint"], kwargs["pc"]
        previous = pc.active_sh_degree, pc._features_rest, camera.dynamics
        pc.active_sh_degree = 3
        pc._features_rest = pc._features_dc.new_zeros((len(pc._xyz), 15, 3))
        camera.dynamics = dict(camera.dynamics, library_car=asset_pose)
        kwargs["dynamic_gaussians"] = dict(kwargs.get("dynamic_gaussians", {}), library_car=car)
        try:
            return original(*args, **kwargs)
        finally:
            pc.active_sh_degree, pc._features_rest, camera.dynamics = previous

    mark_unverified_edit(env, dict(kind="insert", asset=str(asset), status="diagnostic_preview_only"))
    renderer.render = render
    from asset_geometry import declared_contract
    report = dict(asset_calibration=calibration_binding, original_declared_wlh=declared_wlh.tolist(),
                  geometry_contract=declared_contract(wlh, bottom_y), asset=str(asset), asset_iteration=int(iteration), asset_gaussians=len(xyz),
                  geometry_support=dict(opacity_threshold=.5, center_quantiles=[.005, .995],
                      center_bounds_local=bounds.detach().cpu().numpy().tolist(),
                      center_extent_local_xyz=extent.tolist(),
                      support_wlh=(extent[[2,0,1]]).tolist(),
                      support_to_declared_ratio=(extent[[2,0,1]]/wlh).tolist(),
                      authority="gaussian_center_support_proxy" if calibration_binding else "declared_wlh_unverified", gaussian_centers_are_not_mesh=True),
                  local_bottom_y=bottom_y, ground_alignment="sensor_plane_normal",
                  support_contact_model="quantile_bottom_proxy_not_measured_wheel_contacts",
                  placement_search=placement_search,
                  asset_wlh=wlh.tolist(), local_recenter_xz=[float(center[0]), float(center[2])],
                  placements=[dict(case="heading_A", asset_to_source=pose.tolist(), asset_to_native=native.tolist())],
                  placement_rule="shared route lookahead and independent sensor ground fit",
                  sensor_audit=specification["sensor_audit"], n1_edit_certified=False,
                  appearance_independently_validated=False, evaluation_admitted=False,
                  radiance_frame=radiance_frame, radiance_rotation=matrix.tolist(),
                  radiance_rotation_rule='Y(world_direction) M = Y(world_direction @ asset_to_native_rotation)',
                  radiance_rotation_orthogonality_error=float(np.max(np.abs(matrix.T@matrix-np.eye(len(matrix))))))
    (Path(output)/"asset_report.json").write_text(json.dumps(report, indent=2))
    return lambda: setattr(renderer, "render", original)
