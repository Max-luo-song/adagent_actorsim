"""Evidence-only heading tiers. No actor archive, annotation or future frames."""
import numpy as np


def angle_error(a, b):
    # A rectangle has pi-periodic heading; forward/backward share one envelope.
    return np.abs((np.asarray(a)-np.asarray(b)+np.pi/2) % np.pi-np.pi/2)


def corner_error(angle, radius):
    return 2*np.asarray(radius)*np.sin(np.asarray(angle)/2)


def fit_lidar(points, box, margin, bootstrap=24):
    """Yaw-independent cylindrical ROI, then independent PCA long-axis fit.

    Never rotate the crop using detector yaw. Sparse/partial views can fool PCA;
    quality is NOT authority until separately calibrated against heading error.
    ROI uses the detector center, size and bottom-z; independence concerns yaw,
    not object association. Bootstrap count is computational, not a trust cutoff.
    """
    points = np.asarray(points, float).reshape(-1, 3); b = np.asarray(box, float)
    radius = float(np.hypot(*b[3:5])/2)
    valid = (np.isfinite(points).all(1)
             & (np.linalg.norm(points[:, :2]-b[:2], axis=1) <= radius+margin)
             & (points[:, 2] >= b[2]+margin) & (points[:, 2] <= b[2]+b[5]))
    xy = points[valid, :2]
    if len(xy) < 3:
        return dict(available=False, reason='insufficient_roi_points', point_count=len(xy))
    # Bound CPU cost without selecting an angle-dependent subset.
    if len(xy) > 512: xy = xy[np.linspace(0, len(xy)-1, 512).astype(int)]
    def pca(p):
        values, vectors = np.linalg.eigh(np.cov(p.T))
        vector = vectors[:, -1]
        return float(np.arctan2(vector[0], vector[1])), values
    yaw, values = pca(xy)
    if not np.isfinite(values).all() or values.sum() <= 1e-12:
        return dict(available=False, reason='degenerate_cloud', point_count=len(xy))
    rng = np.random.default_rng(0)
    angles = [pca(xy[rng.integers(len(xy), size=len(xy))])[0] for _ in range(bootstrap)]
    dispersion = float(np.quantile(angle_error(angles, yaw), .95))
    anisotropy = float((values[1]-values[0])/values.sum())
    quality = anisotropy*max(0., 1.-dispersion/(np.pi/2))
    return dict(available=True, reason='independent_pca_fit', point_count=len(xy),
                yaw=yaw, quality=quality, anisotropy=anisotropy,
                bootstrap_spread_rad=dispersion, detector_yaw_used=False)


def select_tier(obstacle, calibration, info, packet=None):
    """Choose rectangle -> independently validated rectangle -> circle.

    Missing calibration or out-of-domain data cannot be promoted by raw score.
    Margin covers position error; the calibrated yaw range expands the shape
    once, rather than charging that yaw error again as a metric margin.
    """
    b = np.asarray(obstacle['box'], float)
    label = str(obstacle.get('label')); model = calibration.get('classes', {}).get(label, {})
    now = float(info['timestamp']); score = float(obstacle['raw_score'])
    margin = max(float(calibration['position_margin_floor_m']),
                 float(obstacle.get('geometry_position_uncertainty_m', 0.)))
    result = dict(tier='circle', reason='no_valid_calibration_for_class', yaw=float(b[6]),
                  heading_range_rad=None, score=score, effective_margin_m=margin,
                  calibration_id=calibration['calibration_id'], label=label,
                  empirical_calibration=True, certified_error_bound=False)
    if obstacle.get('state') == 'last_known_confirmed':
        return dict(result, reason='no_current_detection')
    if info.get('heading_calibration_domain') != calibration['domain']:
        return dict(result, reason='calibration_domain_mismatch')
    detector = model.get('detector', {})
    if detector.get('released') and score >= detector['score_threshold']:
        return dict(result, tier='detector_rectangle', reason='calibrated_detection_score',
                    heading_range_rad=detector['heading_bound_rad'])
    result['reason'] = 'detector_below_calibrated_threshold_or_unreleased'
    if packet is None:
        return dict(result, reason=result['reason']+';missing_independent_lidar')
    frame_id = info.get('sensor_frame_id', info.get('source_frame'))
    expected = ('measured_lidar', info.get('scene_token'), now, frame_id)
    observed = (packet.get('source_kind'), packet.get('scene_token'),
                packet.get('timestamp'), packet.get('sensor_frame_id'))
    if (observed != expected or frame_id is None
            or packet.get('coordinate_frame') != 'current_virtual_lidar_xyz_m'
            or packet.get('edited_scene_covered') is not True):
        return dict(result, reason='lidar_provenance_time_frame_or_edit_mismatch')
    fit = fit_lidar(packet['points'], b, margin)
    result['lidar_fit'] = fit
    lidar = model.get('lidar_validation', {})
    if not fit['available']:
        return dict(result, reason=fit['reason'])
    if not lidar.get('released'):
        return dict(result, reason='lidar_quality_agreement_not_calibrated')
    difference = float(angle_error(fit['yaw'], b[6])); result['detector_lidar_difference_rad'] = difference
    if fit['quality'] < lidar['quality_threshold']:
        return dict(result, reason='lidar_quality_below_calibrated_threshold')
    if difference > lidar['agreement_threshold_rad']:
        return dict(result, reason='detector_lidar_heading_disagree')
    # The branch validates DETECTOR orientation; its bound was calibrated on
    # detector error conditional on independent LiDAR agreement and quality.
    return dict(result, tier='lidar_validated_rectangle', reason='calibrated_independent_lidar_agreement',
                heading_range_rad=lidar['detector_heading_bound_rad'])
