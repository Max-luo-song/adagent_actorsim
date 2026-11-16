"""FrameworkSceneBackend: 自研框架渲染器作为闭环观测后端（诊断、未认证）。

分工：自研平台=场景/编辑/观测生成（本模块）；HUGSIM=仿真状态管理与推进
（harness 的 Motion 拥有 ego 状态，本后端不推进）；UniAD 冻结。

渲染路径（复用 cam_inference_online 已验证链，但不走 JPEG 落盘）：
  prepare_online_render_data(ego_pose, cam_data, frame_id) -> render_novel_views
  -> RGB + ray depth + opacity 数组直接进包。

坐标/时间/标定/对象身份：
  - ego pose：source world（first-Waymo-ego），与 harness closed_pose 同系；
    相机位姿 = pose @ cam_to_ego（cam_to_ego 来自框架数据集标定文件）。
  - 时间：frame_id = round(timestamp*fps)，界内钳制；场景内容（动态演员/
    SMPL 姿态）跟随源时间——与原生链 source_time_only 语义一致。
  - 标定：K 取自相机数据在加载分辨率下的内参。
  - 深度语义：框架射线深度（米），标记 depth_encoding='framework_ray_depth'，
    与原生传感器编码不同；不参与跟踪直通控制（仅记录）。
  - 受管资产编辑：标定degree3 SH车辆包插入/位姿/移除；共同高斯栅格化，GPU新能力待验。
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

pass  # Path is provided by the portable boundary above
import gc
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'code/HUGSIM').is_dir())
JOB = ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4'
F = Path(__file__).resolve().parents[2]/'framework_repo'

CAM_ID_TO_NAME = {0: 'CAM_FRONT', 1: 'CAM_FRONT_LEFT', 2: 'CAM_FRONT_RIGHT'}

# 552 完整训练数据（存档 inputs 缺 humanpose；该路径含全部内容）
FRAMEWORK_DATA_ROOT = (_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/data/waymo/processed/training'))


class FrameworkSceneBackend:
    """NativeSceneBackend 接口兼容面：__init__(spec)/render/state/current_view/
    advance/apply/reset/close。advance由harness负责，apply管理标定外部刚性资产。"""

    def __init__(self, spec):
        self.spec = spec
        self.closed = False
        self._last = None  # (timestamp, pose)
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')

        for p in (str(F), str(ROOT / 'code/HUGSIM')):
            if p not in sys.path:
                sys.path.insert(0, p)
        from omegaconf import OmegaConf
        from utils.misc import import_str
        from datasets.driving_dataset_novel_view import DrivingDatasetNovelView

        ckpt = Path(spec.source['checkpoint'])
        cfg = OmegaConf.load(ckpt.parent / 'config.yaml')
        cfg.data.data_root = FRAMEWORK_DATA_ROOT
        device = torch.device('cuda')
        self.dataset = DrivingDatasetNovelView(data_cfg=cfg.data)
        self.cameras_data = self.dataset.load_specified_cameras([0, 1, 2], [2, 2, 2])
        self.trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=self.dataset.num_img_timesteps,
            model_config=cfg.model,
            num_full_images=len(self.dataset.train_indices + self.dataset.test_indices),
            test_set_indices=self.dataset.test_timesteps,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3),
            device=device,
        )
        self.trainer.resume_from_checkpoint(ckpt_path=str(ckpt), load_only_model=True)
        self.trainer.set_eval()
        self.frames = int(spec.frames)

        # phase26: harness post-hoc reference checks read
        # session.backend.env.models[name]['instances_trans'/'instances_quats'/
        # 'instances_size'/'instances_fv'] (native backend internals). Expose
        # the same tensors from the framework trainer -- identical recorded
        # geometry source (both load the same checkpoint scene data).
        import types
        keys = ('instances_trans', 'instances_quats', 'instances_size', 'instances_fv')
        models = {}
        for name in ('RigidNodes', 'DeformableNodes', 'SMPLNodes'):
            model = self.trainer.models.get(name)
            if model is None:
                continue
            state = {k: getattr(model, k) for k in keys if hasattr(model, k)}
            if len(state) == len(keys):
                models[name] = state
        self.env = types.SimpleNamespace(models=models)
        from managed_assets import ManagedAssets
        self.managed_assets = ManagedAssets(self.trainer, device)
        self.supports_managed_assets = True

    def render(self, pose, timestamp):
        """pose: source-world ego_to_world (4x4)。逐相机按位姿重新渲染。"""
        if self.closed:
            raise RuntimeError('backend closed')
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('invalid ego pose')
        torch = self.torch
        from models.video_utils import render_novel_views

        frame_id = int(getattr(self, '_actor_source_frame', round(float(timestamp) * self.spec.fps)))
        frame_id = min(max(frame_id, 0), self.frames - 1)

        traj = torch.from_numpy(pose).float().cuda().unsqueeze(0)
        cameras = {}
        with torch.no_grad():
            for cam_id, name in CAM_ID_TO_NAME.items():
                cam_data = self.cameras_data[cam_id]
                render_data = self.dataset.prepare_online_render_data(
                    traj=traj, target_cam_data=cam_data, frame_id=frame_id)
                res = render_novel_views(self.trainer, render_data, cam_data)
                first = res[0] if isinstance(res, (list, tuple)) else res
                rgb = np.asarray(first['rgb'] if 'rgb' in first else first['rgbs'][0])
                depth = np.asarray(first.get('depth') if 'depth' in first else first['depths'][0])
                opacity = first.get('opacity')
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * 255., 0, 255).astype(np.uint8)
                depth = np.clip(np.nan_to_num(depth.astype(np.float32), nan=0.0,
                                              posinf=0.0, neginf=0.0), 0.0, None)
                cam_to_ego = np.asarray(cam_data.cam_to_ego, dtype=float)
                intr0 = cam_data.intrinsics[0]
                if hasattr(intr0, 'cpu'):
                    intr0 = intr0.cpu().numpy()
                K = np.asarray(intr0, dtype=float)
                cameras[name] = dict(
                    rgb=rgb, depth=depth,
                    # phase26: framework opacity shape/semantics differ from the
                # native depth_alpha contract (failed packet validation); omit
                # until a mapped contract is defined. Recorded in HANDOFF.
                    depth_encoding='framework_ray_depth',
                    K=K,
                    camera_to_world=pose @ cam_to_ego,
                    source_camera_id=cam_id,
                )
        self._last = (float(timestamp), pose.copy())
        return dict(cameras=cameras,
                    ego_state=dict(speed_mps=None, steering_rad=None,
                                   note='ego state owned by harness motion; render-only backend'))

    def advance(self, action):
        raise NotImplementedError('framework backend is render-only; '
                                  'simulation advance is owned by the harness Motion')

    def apply(self, edit):
        if self._last is not None:
            raise RuntimeError('managed assets are pre-run edits; clear preview history or reset first')
        previous = dict(self.managed_assets.assets)
        with self.torch.no_grad():
            self.managed_assets.apply(edit)
        try:
            self.clear_observation_history()
        except Exception:
            self.managed_assets.assets = previous
            raise

    def state(self):
        if self._last is None:
            raise RuntimeError('no render yet')
        ts, pose = self._last
        return dict(timestamp=ts, ego_to_world=pose)

    def current_view(self):
        if self._last is None:
            raise RuntimeError('no render yet')
        return self._last

    def reset(self):
        self.managed_assets.clear()
        self.clear_observation_history()

    def clear_observation_history(self):
        self._last = None

    def close(self):
        self.closed = True
        if hasattr(self, 'managed_assets'):
            self.managed_assets.close()
        self.trainer = None
        self.dataset = None
        self.cameras_data = None
        self.env = None
        # Restored bound collect/raster methods can form cycles on the trainer.
        # Collect them before the next scene allocates its model on the GPU.
        gc.collect()
