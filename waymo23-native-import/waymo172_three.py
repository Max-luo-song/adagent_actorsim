import json
import math
import os
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from scene_math import embed, rigid_geometry

SOURCE_REPORT = (
    "outputs/waymo172_three_setup_20261016_132050_295391/source_report.json"
)
ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]


def make_env(scale=1.0):
    base = Path(os.environ["BASE"])
    source = json.loads((base / SOURCE_REPORT).read_text())
    path = base / "waymo23-native-import/live_env.py"
    text = path.read_text()

    # 独立运行时适配，不修改已有 live_env.py。
    old = "'SMPLNodes':11}"
    if text.count(old) != 1:
        raise RuntimeError("live_env.py 节点定义已变化，请停止并检查")
    text = text.replace(old, "'SMPLNodes':11,'TrafficLightNodes':6}")

    old = (
        "coeff=torch.cat([state['_features_dc'][idx,None,:],"
        "state['_features_rest'][idx]],1)"
    )
    if text.count(old) != 1:
        raise RuntimeError("live_env.py 颜色处理代码已变化，请停止并检查")
    text = text.replace(
        old,
        old + "\n"
        "                        if part['name']=='TrafficLightNodes':\n"
        "                            coeff=coeff+traffic_delta("
        "state,self.cfg['model']['TrafficLightNodes'],"
        "frame,self.frames,self.source_step)[1][idx]"
    )

    cache = {}

    def traffic_delta(state, cfg, frame, frames, step):
        key = (id(state), frame)
        if key in cache:
            return cache[key]

        means = state["_means"]
        coeff = torch.cat([
            state["_features_dc"][:, None, :],
            state["_features_rest"]
        ], dim=1)
        ctrl = cfg.get("ctrl", {})
        enabled = (
            ctrl.get("use_deformgs_for_nonrigid", False)
            and step > ctrl.get("use_deformgs_after", 4000)
        )

        if not enabled:
            result = (torch.zeros_like(means), torch.zeros_like(coeff))
        else:
            network = cfg["networks"]
            if network.get("deform_quat", False):
                raise RuntimeError("此入口尚未启用交通灯旋转变形")
            if network.get("deform_scale", False):
                raise RuntimeError("此入口尚未启用交通灯尺度变形")

            ids = state["points_ids"].flatten().long()
            heights = state["instances_size"][ids, 2:3]
            if not bool((heights > 0).all()):
                raise RuntimeError("交通灯实例高度无效")

            x = means.detach() / heights * 2
            t = x.new_full((len(x), 1), frame / max(frames-1, 1))
            original = torch.cat([
                embed(x, network["x_multires"]),
                embed(t, network["t_multires"]),
                state["instances_embedding"][ids]
            ], dim=-1)

            h = original
            for i in range(network["D"]):
                h = F.relu(F.linear(
                    h.float(),
                    state[f"deform_network.linear.{i}.weight"],
                    state[f"deform_network.linear.{i}.bias"]
                ))
                if i == network["D"] // 2:
                    h = torch.cat([original, h], dim=-1)

            delta_xyz = F.linear(
                h,
                state["deform_network.gaussian_warp.weight"],
                state["deform_network.gaussian_warp.bias"]
            )
            delta_color = torch.zeros_like(coeff)
            if network.get("deform_shs", False):
                delta_color = F.linear(
                    h,
                    state["deform_network.gaussian_color.weight"],
                    state["deform_network.gaussian_color.bias"]
                ).view_as(coeff)

            result = (delta_xyz, delta_color)

        if not all(bool(torch.isfinite(v).all()) for v in result):
            raise RuntimeError("交通灯变形出现非有限值")
        cache.clear()
        cache[key] = result
        return result

    scope = {
        "__name__": "waymo172_live_runtime",
        "__file__": str(path),
        "traffic_delta": traffic_delta,
    }
    exec(compile(text, str(path), "exec"), scope)
    original_geometry = scope["node_geometry"]

    def geometry(name, state, cfg, frame, frames, step):
        if name == "TrafficLightNodes":
            delta_xyz, _ = traffic_delta(state, cfg, frame, frames, step)
            return rigid_geometry(state, frame, delta_xyz)
        return original_geometry(name, state, cfg, frame, frames, step)

    scope["node_geometry"] = geometry
    return scope["make_env"](
        base / "code/HUGSIM",
        source["checkpoint"],
        source["inputs"],
        cameras=(0, 1, 2),
        scale=scale,
        start_speed=0.0
    )


def observation(env):
    from prepare_uniad_input import lidar_to_camera

    obs, info = env.observe(), env.info()
    info["cam_rect"] = np.eye(4)
    info["cam_params"] = {}

    for cid, camera in env.cameras.items():
        name = env.cam_names[cid]
        w, h, k = camera["width"], camera["height"], camera["K"]
        info["cam_params"][name] = {
            "intrinsic": {
                "W": w, "H": h,
                "cx": float(k[0, 2]), "cy": float(k[1, 2]),
                "fovx": 2*math.atan(w/(2*k[0, 0])),
                "fovy": 2*math.atan(h/(2*k[1, 1])),
            },
            "l2c": lidar_to_camera(camera["rig"]),
        }

    assert set(obs["rgb"]) == set(ORDER)
    return obs, info


def main():
    base = Path(os.environ["BASE"])
    source = json.loads((base / SOURCE_REPORT).read_text())
    inputs = Path(source["inputs"])
    scene = Path(source["checkpoint"]).parent
    out = base / "outputs" / (
        "waymo172_three_input_" +
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True)
    print("OUTPUT:", out, flush=True)

    env = make_env(scale=1.0)
    try:
        # 首帧采用原始位姿，以后续帧估计初始速度。
        # 10Hz 为当前数据时间间隔假设，记录在报告中。
        start = source["start_timestep"]
        p0 = np.loadtxt(inputs / "ego_pose" / f"{start:03d}.txt")
        p1 = np.loadtxt(inputs / "ego_pose" / f"{start+1:03d}.txt")
        env.motion.speed = float(
            np.linalg.norm(p1[:3, 3]-p0[:3, 3]) * 10
        )
        obs, info = observation(env)

        # 仅供首帧诊断；闭环转弯指令稍后单独核对。
        info["command"] = 2
        with (out / "observation.pkl").open("wb") as f:
            pickle.dump((obs, info), f, protocol=4)

        order = [(1, "CAM_FRONT_LEFT"), (0, "CAM_FRONT"),
                 (2, "CAM_FRONT_RIGHT")]
        canvas = Image.new("RGB", (1440, 696), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (cid, name) in enumerate(order):
            rgb = obs["rgb"][name]
            Image.fromarray(rgb).save(out / f"{name}.png")
            canvas.paste(
                Image.fromarray(rgb).resize((480, 320)),
                (col*480, 28)
            )
            draw.text((col*480+8, 8), "Adapter: "+name, fill="black")

            saved = (
                scene / "videos/full_set_40000_rgbs" /
                f"000000_{cid:03d}.png"
            )
            if saved.is_file():
                with Image.open(saved) as image:
                    canvas.paste(
                        image.convert("RGB").resize((480, 320)),
                        (col*480, 376)
                    )
            draw.text(
                (col*480+8, 356),
                "Archived render: "+name, fill="black"
            )
            print(name, rgb.shape, flush=True)

        canvas.save(out / "adapter_vs_archived.png")
        report = {
            "scene": 172,
            "checkpoint": source["checkpoint"],
            "inputs": source["inputs"],
            "frame": 0,
            "camera_order": ORDER,
            "checkpoint_loads": env.checkpoint_loads,
            "estimated_initial_speed_mps": info["ego_velo"],
            "pose_sampling_fps_assumption": 10.0,
            "command": 2,
            "command_scope": "provisional first-frame diagnostic only",
            "traffic_light_deformation_enabled": True,
            "sky_and_affine_enabled": False,
            "camera_refinement": False,
            "scene_edits": [],
            "rendering_quality_validated": False,
            "uniad_inference_tested": False,
        }
        (out / "input_report.json").write_text(
            json.dumps(report, indent=2)
        )
        print("初始速度估计：", info["ego_velo"], "m/s")
        print("三视角对照图：", out / "adapter_vs_archived.png")
        print("观测输入包：", out / "observation.pkl")
        print("OUTPUT:", out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
