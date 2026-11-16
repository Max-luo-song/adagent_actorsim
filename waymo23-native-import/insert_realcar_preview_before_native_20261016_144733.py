import os, sys, json, pickle, copy
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from waymo172_three import make_env, SOURCE_REPORT
from scene_math import sh_eval
from export_snapshot import C0


@torch.no_grad()
def main():
    base = Path(os.environ["BASE"])
    baseline = base/"outputs/waymo172_frame175_20261016_133450"
    asset = base/"data/3DRealCar/2024_06_04_13_44_39"
    out = base/"outputs"/(
        "waymo172_realcar_axisfixed_" +
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True)
    print("OUTPUT:", out, flush=True)

    with (baseline/"observation.pkl").open("rb") as f:
        baseline_obs, info = pickle.load(f)

    source = json.loads((base/SOURCE_REPORT).read_text())
    inputs = Path(source["inputs"])
    start = source["start_timestep"]
    frame = int(info["source_frame"])
    poses = np.stack([
        np.loadtxt(inputs/"ego_pose"/f"{start+i:03d}.txt")
        for i in range(source["frames"])
    ])
    relative = np.linalg.inv(poses[0])[None] @ poses
    route = relative[frame:, :3, 3]
    arc = np.r_[0., np.cumsum(
        np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    )]
    assert arc[-1] >= 10
    j = int(np.searchsorted(arc, 10, side="right")-1)
    a = (10-arc[j])/(arc[j+1]-arc[j])
    anchor = route[j]*(1-a)+route[j+1]*a
    direction = route[j+1]-route[j]
    heading = float(np.arctan2(direction[1], direction[0]))

    env = make_env(scale=1.0)
    import gaussian_renderer as renderer
    original_render = renderer.render

    try:
        target = env.target.detach().cpu().numpy().astype(float)
        native_pose = target @ relative[frame] @ np.linalg.inv(target)
        env.ego_pose = lambda: native_pose.copy()
        env.motion.timestamp = float(info["timestamp"])
        env.motion.speed = float(info["ego_velo"])

        restored = env.observe()
        errors = {
            name: int(np.abs(
                restored["rgb"][name].astype(np.int16) -
                baseline_obs["rgb"][name].astype(np.int16)
            ).max())
            for name in baseline_obs["rgb"]
        }
        print("基线重渲染误差：", errors, flush=True)
        if any(errors.values()):
            raise RuntimeError("基线不一致，停止插入")

        params, iteration = torch.load(
            asset/"gs.pth", map_location="cuda", weights_only=False
        )
        assert len(params) == 9
        degree, xyz, dc, rest, semantic, scaling, rotation, opacity, _ = params
        coefficients = torch.cat([dc, rest], dim=1)
        alpha = torch.sigmoid(opacity)
        solid = alpha.flatten() > 0.5
        assert int(solid.sum()) > 100

        bounds = torch.quantile(
            xyz[solid], xyz.new_tensor([0.005, 0.995]), dim=0
        )
        extent = (bounds[1]-bounds[0]).cpu().numpy()
        wlh = np.asarray(json.loads((asset/"wlh.json").read_text()))
        expected = wlh[[1, 2, 0]]  # X长、Y高、Z宽
        print("局部XYZ范围：", extent.round(3).tolist())
        print("尺寸文件对应长高宽：", expected.round(3).tolist())
        if not np.all((extent/expected > 0.5) & (extent/expected < 2)):
            raise RuntimeError("尺度检查未通过")

        center = (bounds[0]+bounds[1])/2
        local_xyz = xyz.clone()
        local_xyz[:, 0] -= center[0]
        local_xyz[:, 2] -= center[2]
        bottom_y = float(bounds[1, 1])

        road = env.models["RoadNodes"]
        points = road["_means"]
        d2 = ((points[:, :2]-points.new_tensor(anchor[:2]))**2).sum(1)
        valid = (d2 < 1) & (
            torch.sigmoid(road["_opacities"].flatten()) > 0.5
        )
        if int(valid.sum()) < 30:
            raise RuntimeError("放置点附近路面点不足")
        ground_z = float(points[valid, 2].median())
        print("估计路面高度：", ground_z)

        asset_pose = None

        def render_with_asset(*args, **kwargs):
            camera = kwargs["viewpoint"]
            pose = asset_pose
            camera_position = torch.as_tensor(
                camera.c2w[:3, 3], device=xyz.device, dtype=xyz.dtype
            ).reshape(3)
            local_camera = pose[:3, :3].T @ (
                camera_position-pose[:3, 3]
            )
            rgb = (
                sh_eval(coefficients, F.normalize(
                    local_xyz-local_camera, dim=-1
                )) + 0.5
            ).clamp(0, 1)

            proxy = SimpleNamespace(
                get_xyz=local_xyz,
                get_rotation=F.normalize(rotation, dim=-1),
                get_opacity=alpha,
                get_scaling=torch.exp(scaling),
                get_features=((rgb-0.5)/C0)[:, None, :],
                get_3D_features=semantic,
            )
            previous = camera.dynamics
            camera.dynamics = dict(previous, library_car=pose)
            kwargs["dynamic_gaussians"] = dict(
                kwargs.get("dynamic_gaussians", {}),
                library_car=proxy
            )
            try:
                return original_render(*args, **kwargs)
            finally:
                camera.dynamics = previous

        renderer.render = render_with_asset
        cases = [("baseline", baseline_obs)]
        placements = []

        for name, offset in [("heading_A", 0.), ("heading_B", np.pi)]:
            angle = heading+offset
            c, s = np.cos(angle), np.sin(angle)

            # 局部X沿路线，Y向下，Z横向。
            source_pose = np.eye(4)
            source_pose[:3, :3] = [
                [c, 0, -s],
                [s, 0,  c],
                [0, -1, 0]
            ]
            assert np.isclose(np.linalg.det(source_pose[:3, :3]), 1)
            source_pose[:3, 3] = [
                anchor[0], anchor[1], ground_z+bottom_y
            ]
            native_asset_pose = target @ source_pose
            asset_pose = xyz.new_tensor(native_asset_pose)

            obs = env.observe()
            folder = out/name
            folder.mkdir()
            with (folder/"observation.pkl").open("wb") as f:
                pickle.dump((obs, copy.deepcopy(info)), f, protocol=4)
            Image.fromarray(obs["rgb"]["CAM_FRONT"]).save(
                folder/"front.png"
            )
            cases.append((name, obs))
            placements.append({
                "case": name,
                "asset_to_native": native_asset_pose.tolist(),
                "asset_to_source": source_pose.tolist(),
            })

        order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        canvas = Image.new("RGB", (1440, 1044), "white")
        draw = ImageDraw.Draw(canvas)
        for row, (label, obs) in enumerate(cases):
            for col, camera in enumerate(order):
                canvas.paste(
                    Image.fromarray(obs["rgb"][camera]).resize((480,320)),
                    (col*480, row*348+28)
                )
                draw.text(
                    (col*480+8, row*348+8),
                    label+": "+camera, fill="black"
                )
        canvas.save(out/"asset_placement_comparison.png")
        report = {
            "baseline": str(baseline),
            "asset": str(asset),
            "asset_iteration": int(iteration),
            "asset_gaussians": len(xyz),
            "asset_wlh": wlh.tolist(),
            "asset_axis_order": "length,height,width",
            "frame": frame,
            "distance_along_route_m": 10.,
            "estimated_ground_source_z": ground_z,
            "local_bottom_y": bottom_y,
            "local_recenter_xz": [float(center[0]), float(center[2])],
            "placements": placements,
            "baseline_rerender_max_error": errors,
            "original_scene_vehicles_preserved": True,
            "original_checkpoint_written": False,
            "placement_quality_verified": False,
            "uniad_inference_tested": False,
        }
        (out/"asset_report.json").write_text(json.dumps(report, indent=2))
        print("三组预览：", out/"asset_placement_comparison.png")
        print("OUTPUT:", out)
    finally:
        renderer.render = original_render
        env.close()


if __name__ == "__main__":
    main()
