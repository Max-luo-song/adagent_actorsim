"""Metric placement from sensor ground, without scene-specific offsets."""
import numpy as np


def terrain_aligned_pose(anchor, heading, ground_fit, bottom_y):
    """Asset axes: X forward, Y down, Z lateral; source coordinates Z up.

    Align a declared flat support plane to the independently fitted terrain.
    This is placement geometry, not evidence of real wheel contact or N1 trust.
    """
    anchor = np.asarray(anchor, dtype=float).copy()
    origin = np.asarray(ground_fit['origin_xy'], dtype=float)
    plane = np.asarray(ground_fit['plane'], dtype=float)
    if anchor.shape != (3,) or origin.shape != (2,) or plane.shape != (3,):
        raise ValueError('Invalid terrain metadata shape')
    if not np.isfinite(np.r_[anchor, origin, plane, heading, bottom_y]).all():
        raise ValueError('Nonfinite placement metadata')
    a, b, height = plane
    anchor[2] = (anchor[:2]-origin)@plane[:2]+height
    normal = np.array([-a, -b, 1.])
    normal /= np.linalg.norm(normal)
    c, s = np.cos(heading), np.sin(heading)
    forward = np.array([c, s, a*c+b*s])
    forward /= np.linalg.norm(forward)
    down = -normal
    lateral = np.cross(forward, down)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([forward, down, lateral])
    pose[:3, 3] = anchor-pose[:3, 1]*bottom_y
    return pose


def supported_placement(anchor, heading, ground_fit, bottom_y, wlh, skin):
    """Search a shared metric neighborhood using independent sensor support.

    Box bottom points are a diagnostic proxy, never authenticated wheel patches.
    No thresholds or offsets are selected by scene identity.
    """
    from edit_trust import check_skin, EditPolicy, EditTrustError
    from n1_cross_scene_audit import fit_observed_ground, CrossSceneProtocol
    width, length, height = np.asarray(wlh, dtype=float)
    contacts = np.array([[x, bottom_y, z, 1.] for x in (-length/2, length/2) for z in (-width/2, width/2)])
    from asset_geometry import local_corners
    corners = np.column_stack([local_corners(wlh, bottom_y), np.ones(8)])
    c, s = np.cos(heading), np.sin(heading)
    offsets = [(a, b) for a in (0, 1, -1, 2, -2, 3, -3) for b in (0, 1, -1, 2, -2, 3, -3)]
    offsets.sort(key=lambda x: x[0]*x[0]+x[1]*x[1])
    trials = []
    policy = EditPolicy()
    protocol = CrossSceneProtocol()
    for longitudinal, lateral in offsets:
        candidate = np.asarray(anchor, dtype=float).copy()
        candidate[:2] += longitudinal*np.array([c, s])+lateral*np.array([-s, c])
        ground = np.asarray(skin['ground_points'])
        near = ground[np.linalg.norm(ground[:, :2]-candidate[:2], axis=1) <= np.hypot(length, width)/2+policy.support_radius_m]
        if len(near) < protocol.minimum_ground_points:
            trials.append(dict(longitudinal_m=longitudinal, lateral_m=lateral,
                               proxy_geometry_passed=False, rejection='Insufficient local plane observations'))
            continue
        try:
            local_fit = fit_observed_ground(near, policy.support_error_m, policy.max_ground_slope_degrees)
        except ValueError as exc:
            trials.append(dict(longitudinal_m=longitudinal, lateral_m=lateral,
                               proxy_geometry_passed=False, rejection=str(exc)))
            continue
        pose = terrain_aligned_pose(candidate, heading, local_fit, bottom_y)
        surface = (corners@pose.T)[:, :3]
        request = dict(kind='insert', geometry_skin=surface, support_points=(contacts@pose.T)[:, :3],
                       new_bounds=np.stack([surface.min(0)-1e-6, surface.max(0)+1e-6]))
        try:
            measured = check_skin(request, skin, policy)
            trials.append(dict(longitudinal_m=longitudinal, lateral_m=lateral, proxy_geometry_passed=True))
            return pose, dict(trials=trials, selected_offset_m=[longitudinal, lateral],
                              selected_local_ground_fit=local_fit,
                              proxy_geometry=measured, real_wheel_contacts_verified=False,
                              n1_edit_certified=False)
        except EditTrustError as exc:
            trials.append(dict(longitudinal_m=longitudinal, lateral_m=lateral,
                               proxy_geometry_passed=False, rejection=str(exc)))
    raise EditTrustError('No sensor-supported unoccupied placement in the shared neighborhood: '+str(trials))
