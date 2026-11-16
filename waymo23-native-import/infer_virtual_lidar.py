import os
import copy
import json
import pickle
import inspect
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

import five_uniad
import infer_three_pad32 as previous_runner

base = Path(os.environ["BASE"])
original_job = base / (
    "outputs/waymo172_realcar_native_20261016_144739_944912/heading_A"
)
job = original_job / "virtual_lidar_aligned"
job.mkdir(parents=True, exist_ok=True)

cfg_path = base/"code/HUGSIM/configs/sim/nuscenes_camera.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

def transform(extrinsics, prefix):
    matrix = np.eye(4)
    matrix[:3,:3] = Rotation.from_euler(
        "XYZ", extrinsics[prefix+"_rot"], degrees=True
    ).as_matrix()
    matrix[:3,3] = extrinsics[prefix+"_trans"]
    return matrix

# 参考配置的车辆坐标：X前、Y左、Z上。
# 当前自车坐标：X右、Y前、Z上。
vehicle_to_current = np.eye(4)
vehicle_to_current[:3,:3] = [[0,-1,0], [1,0,0], [0,0,1]]
translations = []
for name in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]:
    extrinsics = cfg["cams"][name]["extrinsics"]
    lidar_to_vehicle = (
        np.linalg.inv(transform(extrinsics, "v2c"))
        @ transform(extrinsics, "l2c")
    )
    translations.append(
        (vehicle_to_current @ lidar_to_vehicle)[:3,3]
    )
assert np.allclose(translations, translations[0], atol=1e-5)

# 虚拟坐标轴保持与当前自车坐标一致，只改变原点。
virtual_shift = np.asarray(translations[0])
virtual_to_ego = np.eye(4)
virtual_to_ego[:3,3] = virtual_shift
print("虚拟LiDAR原点在自车坐标中的XYZ：",
      virtual_shift.round(6).tolist(), flush=True)

with (original_job/"observation.pkl").open("rb") as f:
    obs, original_info = pickle.load(f)
info = copy.deepcopy(original_info)

for name, camera in info["cam_params"].items():
    camera["l2c"] = (
        np.asarray(original_info["cam_params"][name]["l2c"])
        @ virtual_to_ego
    )

# 检查改变坐标原点后，同一三维点在相机中的位置不变。
ego_points = np.array([
    [0.,10.,0.,1.], [1.,12.,1.,1.], [-1.,8.,0.5,1.]
]).T
virtual_points = np.linalg.inv(virtual_to_ego) @ ego_points
for name in info["cam_params"]:
    before = (
        np.asarray(original_info["cam_params"][name]["l2c"])
        @ ego_points
    )
    after = info["cam_params"][name]["l2c"] @ virtual_points
    assert np.allclose(before, after, atol=1e-8), name

info["model_lidar_frame"] = {
    "kind": "virtual lidar; axes aligned with ego",
    "virtual_to_ego": virtual_to_ego.tolist(),
    "reference_config": str(cfg_path),
    "is_waymo_sensor_calibration": False,
}
with (job/"observation.pkl").open("wb") as f:
    pickle.dump((obs, info), f, protocol=4)

original_pad = previous_runner.pad_to_32

def pad_and_update_global(data):
    original_pad(data)
    rotation = data["l2g_r_mat"]
    translation = data["l2g_t"]
    assert tuple(rotation.shape) == (3,3), rotation.shape
    assert tuple(translation.shape) == (3,), translation.shape
    shift = torch.as_tensor(
        virtual_shift, dtype=rotation.dtype, device=rotation.device
    )
    data["l2g_t"] = translation + (rotation @ shift).to(translation)
    print("相机外参和l2g_t已同步到虚拟LiDAR原点", flush=True)

# 复用已保存的三视角、Pad32推理入口。
# 只替换输入参考位置及坐标更新，不修改模型权重或检测阈值。
text = inspect.getsource(previous_runner.infer_three)
replacements = [
    (
        '(job/"prediction/uniad_report.json")',
        '(original_job/"prediction/uniad_report.json")'
    ),
    (
        'center = np.asarray(previous["asset_center_xy"])',
        'center = np.asarray(previous["asset_center_xy"]) - virtual_shift[:2]'
    ),
    (
        '(job.parent/"asset_report.json")',
        '(original_job.parent/"asset_report.json")'
    ),
]
for old, new in replacements:
    assert text.count(old) == 1, "推理入口结构变化："+old
    text = text.replace(old, new)

scope = dict(previous_runner.__dict__)
scope.update(
    original_job=original_job,
    virtual_shift=virtual_shift,
    pad_to_32=pad_and_update_global,
)
exec(compile(text, "virtual_lidar_inference", "exec"), scope)

os.environ["FIVE_JOB"] = str(job)
five_uniad.infer_five = scope["infer_three"]
five_uniad.infer()

# 坐标轴相同，规划XY通过平移转换回自车坐标。
out = job/"prediction_pad32"
data = np.load(out/"perception.npz")
plan_virtual = data["trajectory"].reshape(-1,2)
plan_ego = plan_virtual + virtual_shift[:2]
np.save(out/"planning_traj_ego.npy", plan_ego)

report_path = out/"uniad_report.json"
report = json.loads(report_path.read_text())
report.update(
    model_lidar_frame="virtual; ego-aligned axes",
    virtual_to_ego=virtual_to_ego.tolist(),
    is_waymo_sensor_calibration=False,
    camera_projection_invariance_checked=True,
    l2g_translation_updated=True,
    trajectory_virtual_xy=plan_virtual.tolist(),
    trajectory_ego_xy=plan_ego.tolist(),
    closed_loop_tested=False,
)
report_path.write_text(json.dumps(report, indent=2))
print("\n转换回自车坐标的规划：\n", plan_ego)
print("结果目录：", out)
