"""Pano-LiDAR 单帧"插车前/后"验证（诊断、未认证；近似 LiDAR，非 lidargs 等价）。

链路：MultiTrainer 渲染器（已验收）→ 8 视锥环绕 lidar 原点 → 柱面 z-buffer
→ 点云。无强度字段（无可靠依据）。编辑经框架 add_instance（复制场景内既有
车辆实例到车前新位），与相机共享同一 trainer/编辑状态。
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
import time
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
F = JOB / 'chain_repair/phase26/framework_repo'
for p in (str(HERE), str(F), str(ROOT / 'code/HUGSIM')):
    if p not in sys.path:
        sys.path.insert(0, p)

from pano_lidar import yaw_virtual_egos, pinhole_to_pano, PanoBuilder  # noqa

CKPT = Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth'))
DATA_ROOT = (_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/data/waymo/processed/training'))
OUT = HERE / 'pano_validation'
FRAME = 80
N_VIEWS = 8


def build_trainer():
    import torch
    from omegaconf import OmegaConf
    from utils.misc import import_str
    from datasets.driving_dataset_novel_view import DrivingDatasetNovelView

    cfg = OmegaConf.load(CKPT.parent / 'config.yaml')
    cfg.data.data_root = DATA_ROOT
    device = torch.device('cuda')
    dataset = DrivingDatasetNovelView(data_cfg=cfg.data)
    cams = dataset.load_specified_cameras([0], [2])  # 仅用前视相机做视锥渲染
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_full_images=len(dataset.train_indices + dataset.test_indices),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device)
    trainer.resume_from_checkpoint(ckpt_path=str(CKPT), load_only_model=True)
    trainer.set_eval()
    return trainer, dataset, cams[0]


def render_pano(trainer, dataset, cam_data, lidar_to_world, frame_id):
    """8 视锥环绕 lidar 原点 → 柱面深度 → 点云。"""
    import torch
    from models.video_utils import render_novel_views
    cam_to_ego = np.asarray(cam_data.cam_to_ego, dtype=float)
    builder = PanoBuilder(azimuth_res_deg=0.2, elev_min_deg=-25, elev_max_deg=15,
                          elev_res_deg=0.4, max_range=120.0)
    t0 = time.time()
    for ego in yaw_virtual_egos(lidar_to_world, cam_to_ego, N_VIEWS):
        traj = torch.from_numpy(ego).float().cuda().unsqueeze(0)
        rd = dataset.prepare_online_render_data(traj=traj, target_cam_data=cam_data,
                                                frame_id=frame_id)
        res = render_novel_views(trainer, rd, cam_data)
        depth = np.asarray(res['depths'][0], dtype=np.float32)
        # 该深度为相机 z 深度；pinhole_to_pano 内部转斜距
        intr = cam_data.intrinsics[0]
        K = intr.cpu().numpy() if hasattr(intr, 'cpu') else np.asarray(intr)
        az, el, slant, valid = pinhole_to_pano(depth, K, np.eye(4), None, None)
        builder.add_view(az, el, slant, valid)
    elapsed = time.time() - t0
    pts, valid_mask = builder.to_points()
    return pts, builder.depth.copy(), elapsed


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    OUT.mkdir(parents=True)
    import torch

    trainer, dataset, cam_data = build_trainer()

    # frame 80 的 ego/lidar 位姿（waymo：lidar 系==车体系）
    inputs = Path(DATA_ROOT) / '552'
    poses = np.stack([np.loadtxt(inputs / 'ego_pose' / f'{i:03d}.txt')
                      for i in range(198)])
    relative = np.linalg.inv(poses[0])[None] @ poses
    lidar_pose = relative[FRAME]

    # 基线
    pts_base, pano_base, t_base = render_pano(trainer, dataset, cam_data,
                                              lidar_pose, FRAME)
    np.save(OUT / 'points_baseline.npy', pts_base)

    # 插入：复制场景内一辆可见车（实例 0）到 ego 前方 15m，仅 frame 80 可见
    rigid = trainer.models['RigidNodes']
    from models.nodes.rigid import RigidNodes  # noqa
    insert_xy = relative[FRAME][:2, 3] + np.array([15.0, 0.0])
    ego_traj = torch.zeros(198, 4, 4, device='cuda')
    ego_traj[:, 0, 0] = 1; ego_traj[:, 1, 1] = 1; ego_traj[:, 2, 2] = 1; ego_traj[:, 3, 3] = 1
    ego_traj[:, 0, 3] = insert_xy[0]; ego_traj[:, 1, 3] = insert_xy[1]
    rigid.add_instance(target_id=-1, add_id=0, ego_traj=ego_traj,
                       time_window=[[FRAME, FRAME + 1]])
    pts_edit, pano_edit, t_edit = render_pano(trainer, dataset, cam_data,
                                              lidar_pose, FRAME)
    np.save(OUT / 'points_inserted.npy', pts_edit)

    # 对照统计
    report = dict(frame=FRAME, n_views=N_VIEWS,
                  lidar_frame='waymo lidar==vehicle frame',
                  baseline=dict(points=len(pts_base), render_seconds=round(t_base, 2)),
                  inserted=dict(points=len(pts_edit), render_seconds=round(t_edit, 2)),
                  points_delta=len(pts_edit) - len(pts_base),
                  intensity='OMITTED (no reliable basis; see README)',
                  approximate_lidar=True, lidargs_equivalent=False,
                  diagnostic_uncertified=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
