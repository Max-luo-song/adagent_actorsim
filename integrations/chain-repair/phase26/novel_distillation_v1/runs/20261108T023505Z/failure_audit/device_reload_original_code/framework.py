"""Gradient-capable adapter around the existing frozen framework renderer."""

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

import sys
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from contracts import ROOT, P26, pose, compose_camera_pose


def bootstrap(bundle):
    bundle = Path(bundle).resolve()
    sys.path[:0] = [str(bundle), str(bundle/'bridge/framework_backend'),
                   str(bundle/'framework_repo'), str(ROOT/'integrations/sr-hugsim'),
                   str(P26/'bridge/agent_json')]
    from full_scene_contract import verify
    verify(bundle)


class Renderer:
    def __init__(self, scene_manifest, bundle):
        bootstrap(bundle)
        from scene_api import SceneSpec
        from backend import FrameworkSceneBackend
        self.spec = SceneSpec.read(scene_manifest)
        self.backend = FrameworkSceneBackend(self.spec)
        if self.backend.dataset.num_img_timesteps != self.spec.frames or self.backend.dataset.start_timestep != self.spec.source['start_timestep']:
            raise ValueError('Checkpoint dataset clock differs from scene source')
        self.trainer = self.backend.trainer
        self.torch = self.backend.torch
        # eval mode keeps dropout/densification/training hooks disabled, gradients remain enabled.
        self.trainer.set_eval()

    def frame(self, ego_pose, frame, cam):
        if type(frame) is not int or not 0 <= frame < self.spec.frames:
            raise ValueError('Frame outside scene clock')
        camera = self.backend.cameras_data[cam]
        camera_pose = compose_camera_pose(ego_pose, camera.cam_to_ego)
        t = self.torch.tensor(camera_pose, dtype=self.torch.float32, device='cuda')[None]
        # The public helper multiplies ego @ extrinsics on GPU. Feed the already
        # composed camera pose into the SAME frozen ray/time builder instead.
        dataset = self.backend.dataset
        data = dataset.pixel_source._build_online_render_data(
            dataset_type=dataset.type, traj=t, camera_data=camera, frame_id=frame)
        if len(data) != 1:
            raise RuntimeError('Expected exactly one camera/time sample')
        actual = data[0]['cam_infos']['camera_to_world'].detach().cpu().numpy()
        pose(actual)
        if not np.array_equal(actual, camera_pose):
            raise ValueError('Ray builder changed the supplied calibrated camera pose')
        return data[0]

    def rgb(self, data, layers=False):
        # Do not call video_utils.render_novel_views: it has a no_grad context.
        old = self.trainer.render_each_class
        self.trainer.render_each_class = layers
        try:
            out, _ = self.trainer(image_infos=data['image_infos'], camera_infos=data['cam_infos'],
                                  novel_view=True, render_only=not layers, freeze_road=False)
            return out
        finally:
            self.trainer.render_each_class = old

    def valid_mask(self, out, data):
        torch = self.torch
        # Ignore actor/sky/ego pixels; no semantic labels are manufactured.
        static = sum((out[f'{name}_opacity'] for name in ['Background', 'RoadNodes']
                      if f'{name}_opacity' in out), torch.zeros_like(out['opacity']))
        dynamic = torch.zeros_like(static)
        for name in ['RigidNodes','DeformableNodes','SMPLNodes','TrafficLightNodes']:
            if f'{name}_opacity' in out:
                dynamic = torch.maximum(dynamic, out[f'{name}_opacity'])
        mask = (static.squeeze(-1) > .8) & (dynamic.squeeze(-1) < .01)
        ego = data['image_infos'].get('egocar_masks')
        if ego is not None:
            mask &= ~ego.bool()
        return mask

    def original(self, frame, cam):
        # Match the framework camera loading order: resize, then optional undistort.
        import cv2
        c = self.backend.cameras_data[cam]
        if getattr(c, 'is_fisheye', False):
            raise ValueError('First adapter supports Waymo pinhole cameras only')
        path = Path(c.img_filepaths[frame])
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB').resize((c.load_size[1], c.load_size[0]), Image.Resampling.BILINEAR))
        if c.undistort:
            rgb = cv2.undistort(rgb, c.intrinsics[frame].cpu().numpy(), c.distortions[frame].cpu().numpy())
        if rgb.shape[:2] != (c.HEIGHT, c.WIDTH):
            raise ValueError('GT and render resolutions disagree')
        return rgb, path, c.cam_to_worlds[frame].detach().cpu().numpy()

    def close(self):
        self.backend.close()
