"""CPU contracts for source metadata and the shared vehicle profile."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VehicleConfig:
    dt: float = .1
    wheelbase_m: float = 2.7
    half_width_m: float = .8
    half_length_m: float = 1.5
    body_height_m: float = 1.6
    min_acc_mps2: float = -2.
    max_acc_mps2: float = 2.
    max_steer_rate_degrees: float = 15.
    max_steer_degrees: float = 80.
    max_speed_mps: float = 30.
    emergency_braking_mps2: float = 2.
    clearance_margin_m: float = .2

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Nonfinite vehicle configuration")
        if (any(getattr(self, k) <= 0 for k in (
                "dt", "wheelbase_m", "half_width_m", "half_length_m", "body_height_m",
                "max_acc_mps2", "max_steer_rate_degrees", "max_steer_degrees", "max_speed_mps",
                "emergency_braking_mps2")) or self.min_acc_mps2 >= 0
                or self.clearance_margin_m < 0 or self.max_steer_degrees >= 90
                or self.emergency_braking_mps2 > -self.min_acc_mps2
                or not np.isclose(.5/self.dt, round(.5/self.dt))):
            raise ValueError("Invalid vehicle configuration or incompatible model horizon")


def load_vehicle_config(path=None):
    if path is None:
        path = Path(__file__).parent/"configs/vehicle.json"
    return VehicleConfig(**json.loads(Path(path).read_text()))


def model_input_profile():
    """Contract of the installed UniAD parser/checkpoint, shared by all scenes."""
    return dict(width=1600, height=900, pad_divisor=32, input_color="RGB",
                box_convention='mmdet3d_legacy_lidar_xyz_dims_clockwise_yaw_bottom_center_v1',
                parser_color="BGR", mean=[103.530, 116.280, 123.675],
                std=[1., 1., 1.], to_rgb=False, planning_bev="history")


def configured_reliability():
    # Resident UniAD workers start with only the adapter directory on sys.path.
    # Load the NumPy-only support module independently of the GPU plugin package.
    import sys
    support = Path(__file__).resolve().parents[1]/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad"
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from scene_reliability import ReliabilityConfig
    vehicle = load_vehicle_config()
    return ReliabilityConfig(ego_half_width_m=vehicle.half_width_m,
                             ego_half_length_m=vehicle.half_length_m,
                             braking_mps2=vehicle.emergency_braking_mps2)


def scene_camera_mapping(source):
    """Map source camera IDs to semantic names; calibration stays in source files.

    Current renderer backend supports Waymo metric Z-up input only. Reject an
    unsupported declaration rather than silently interpreting its axes.
    """
    metadata = source.get("scene_adapter", {})
    unknown = set(metadata)-{"coordinate_system", "camera_names"}
    if unknown:
        raise ValueError("Unsupported scene adapter metadata: "+str(sorted(unknown)))
    if metadata.get("coordinate_system", "waymo_z_up_meters") != "waymo_z_up_meters":
        raise ValueError("Unsupported source coordinate system")
    names = metadata.get("camera_names", {"0": "CAM_FRONT", "1": "CAM_FRONT_LEFT", "2": "CAM_FRONT_RIGHT"})
    mapping = {int(k): v for k, v in names.items()}
    if (not mapping or len(mapping) != len(names) or min(mapping) < 0
            or any(not isinstance(v, str) or not v for v in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("Invalid source camera mapping")
    return mapping


def preflight_source(source):
    """Validate archived sensor metadata without importing renderer/model code."""
    mapping = scene_camera_mapping(source)
    if set(mapping.values()) != {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"}:
        raise ValueError("Current closed-loop model profile requires three front cameras")
    if "camera_ids" in source and not set(mapping).issubset(source["camera_ids"]):
        raise ValueError("Camera mapping is absent from source camera_ids")
    fps = float(source.get("pose_fps_assumption", 10.))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source timestamp frequency")
    inputs = Path(source["inputs"])
    calibrations = {}
    for cid, name in mapping.items():
        intr = np.loadtxt(inputs/"intrinsics"/f"{cid}.txt").reshape(-1)
        pose = np.loadtxt(inputs/"extrinsics"/f"{cid}.txt")
        if (len(intr) < 4 or not np.isfinite(intr).all() or np.any(intr[:2] <= 0)
                or pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
            raise ValueError("Invalid sensor calibration: "+name)
        calibrations[name] = dict(source_camera_id=cid, intrinsics=intr.tolist(),
                                  camera_to_ego=pose.tolist())
    return dict(coordinate_system="waymo_z_up_meters", source_fps=fps,
                timestamp_frequency_independently_validated=False,
                cameras=calibrations)
