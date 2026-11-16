"""
LiDAR 编辑效果 BEV（俯视图）可视化视频生成工具。

与在线渲染路径完全解耦：只读取已保存到磁盘的逐帧点云 .pcd 文件和
ego_positions.json（sim_render/lidar/render.py 的 CLI 编辑渲染路径产出），
不 import sim_render/lidar/render.py，避免给在线渲染模块引入
cv2/matplotlib 依赖。

ego 位置必须从 ego_positions.json 读取，不能用 lidar_pose 原始位姿文件
独立重算——两者不是同一坐标系，配对会导致自车轨迹和点云对不上。

既可以当库用（配合 render.py --output_path 下的 renders_edit/）：
    frames = collect_rendered_frames(render_output_dir)
    ego_positions = load_ego_positions(render_output_dir)
    make_bev_video(frames, ego_positions, out_path="edit_bev.mp4")

也可以直接当命令行工具用（见文件末尾 _main）：
    python utils/lidar_edit_video.py --render_dir .../renders_edit --out_dir .../rendervideo
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

import glob
import json
import os
import re

import numpy as np


def read_pcd_xyzi_binary(pcd_path: str) -> np.ndarray:
    """读取 render.py save_pcd_binary 写出的 x,y,z,intensity 二进制 PCD，返回 (N,4) 数组。"""
    with open(pcd_path, "rb") as f:
        raw = f.read()
    marker = b"DATA binary\n"
    header_end = raw.index(marker) + len(marker)
    header_text = raw[:header_end].decode("ascii")
    n = int(re.search(r"POINTS (\d+)", header_text).group(1))
    data = np.frombuffer(raw[header_end:], dtype=np.float32, count=n * 4)
    return data.reshape(n, 4).copy()


def load_ego_positions(render_dir: str) -> dict:
    """读取 render.py 写出的 render_dir/ego_positions.json，返回 {frame_id: (3,) ndarray}。

    render_dir 是 collect_rendered_frames 用的同一个目录（即 render.py
    --output_path 下的 renders_edit/）。找不到该文件时返回空 dict——
    调用方应把它当作"没有 ego 轨迹信息，BEV 窗口退化为固定在原点"处理，
    而不是自己去拿 lidar_pose 重算（坐标系对不上，见模块顶部说明）。
    """
    json_path = os.path.join(render_dir, "ego_positions.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        raw = _hugsim_expand(json.load(f))
    return {int(fid): np.array(pos, dtype=np.float64) for fid, pos in raw.items()}


def collect_rendered_frames(render_dir: str, lidar_id: int = 0) -> dict:
    """扫描 render.py 输出目录里的 ldr{lidar_id}_{frame_id:06d}.pcd，返回 {frame_id: (N,4) array}。"""
    pattern = os.path.join(render_dir, f"ldr{lidar_id}_*.pcd")
    name_re = re.compile(rf"^ldr{lidar_id}_(\d+)\.pcd$")
    frames = {}
    for path in sorted(glob.glob(pattern)):
        m = name_re.match(os.path.basename(path))
        if not m:
            continue
        frames[int(m.group(1))] = read_pcd_xyzi_binary(path)
    return frames


def make_bev_video(
    frames: dict,
    ego_positions: dict,
    out_path: str,
    gt_frames: dict = None,
    bev_xlim=(-60.0, 60.0),
    bev_ylim=(-50.0, 50.0),
    panel_w: int = 1200,
    panel_h: int = 900,
    fps: int = 10,
    pt_size: float = 2.0,
) -> None:
    """
    生成 BEV mp4：每帧以 ego 位置为中心裁剪显示范围，散点按强度(jet)着色。

    - frames / gt_frames: {frame_id: (N,4) x,y,z,intensity}，gt_frames 提供时双栏 GT|Rendered。
    - bev_xlim/bev_ylim: 相对 ego 位置的偏移范围（米），xlim 对应行驶方向、ylim 对应横向。
    - 横轴=横向 Y，雷达系 +Y 指向左，故 invert_xaxis 让左侧点显示在图像左侧，
      否则整幅 BEV 会与真实场景/图像侧左右镜像。
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_ids = sorted(frames.keys())
    if not frame_ids:
        raise ValueError("frames 为空，无法生成视频")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    dual_panel = gt_frames is not None
    n_panels = 2 if dual_panel else 1
    total_w = panel_w * n_panels
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (total_w, panel_h))

    for frame_id in frame_ids:
        ren_pts = frames[frame_id]
        gt_pts = gt_frames.get(frame_id) if dual_panel else None
        ego_pos = ego_positions.get(frame_id, np.zeros(3))
        ego_x, ego_y = float(ego_pos[0]), float(ego_pos[1])
        frame_xlim = (ego_y + bev_ylim[0], ego_y + bev_ylim[1])
        frame_ylim = (ego_x + bev_xlim[0], ego_x + bev_xlim[1])

        panels = [("GT", gt_pts), ("Rendered (edit)", ren_pts)] if dual_panel else [("Rendered (edit)", ren_pts)]
        fig, axs = plt.subplots(1, n_panels, figsize=(panel_w * n_panels / 100, panel_h / 100), dpi=100)
        axs = [axs] if n_panels == 1 else list(axs)
        fig.patch.set_facecolor("white")

        for ax, (title, pts) in zip(axs, panels):
            ax.set_facecolor("white")
            ax.set_xlim(frame_xlim)
            ax.set_ylim(frame_ylim)
            ax.invert_xaxis()
            ax.set_aspect("auto")
            ax.axis("off")
            ax.set_title(title, fontsize=13, pad=6, color="#222222")
            if pts is not None and len(pts) > 0:
                ax.scatter(
                    pts[:, 1], pts[:, 0],
                    c=np.clip(pts[:, 3], 0.0, 1.0),
                    s=pt_size,
                    cmap="jet",
                    vmin=0.0, vmax=1.0,
                    linewidths=0,
                    rasterized=True,
                )
            ax.plot(ego_y, ego_x, "r+", markersize=10, markeredgewidth=1.5)

        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (total_w, panel_h), interpolation=cv2.INTER_LINEAR)
        vw.write(bgr)
        plt.close(fig)

    vw.release()


def _main():
    """CLI：从 render.py 落盘的 renders_edit/ 目录直接画 BEV mp4，不经过任何中间
    py 脚本——render.py 负责渲染，这个文件负责画图，各自是独立可单独调用的
    命令行工具，批量/单 clip 脚本用 shell 直接串起来即可。

    用法：
        python utils/lidar_edit_video.py \
            --render_dir /path/to/output/renders_edit \
            --out_dir /path/to/output/rendervideo \
            --lidar_id 0 --video_fps 10 --bev_point_size 0.5 \
            --bev_xmin -30 --bev_xmax 30 --bev_ymin -30 --bev_ymax 30
    """
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--render_dir", required=True, help="render.py --output_path 下的 renders_edit/ 目录")
    parser.add_argument("--out_dir", required=True, help="视频输出目录（生成 lidar_edit_bev.mp4 / _h264.mp4）")
    parser.add_argument("--lidar_id", type=int, default=0)
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--bev_width", type=int, default=1200)
    parser.add_argument("--bev_height", type=int, default=900)
    parser.add_argument("--bev_xmin", type=float, default=-30.0)
    parser.add_argument("--bev_xmax", type=float, default=30.0)
    parser.add_argument("--bev_ymin", type=float, default=-30.0)
    parser.add_argument("--bev_ymax", type=float, default=30.0)
    parser.add_argument("--bev_point_size", type=float, default=0.5)
    args = parser.parse_args()

    frames = collect_rendered_frames(args.render_dir, lidar_id=args.lidar_id)
    if not frames:
        print(f"[lidar_edit_video] {args.render_dir} 下未找到 ldr{args.lidar_id}_*.pcd，跳过视频生成")
        return
    ego_positions = load_ego_positions(args.render_dir)
    if not ego_positions:
        print(f"[lidar_edit_video] {args.render_dir} 下没有 ego_positions.json，BEV 窗口将固定在原点")

    out_path = os.path.join(args.out_dir, "lidar_edit_bev.mp4")
    make_bev_video(
        frames,
        ego_positions,
        out_path=out_path,
        bev_xlim=(args.bev_xmin, args.bev_xmax),
        bev_ylim=(args.bev_ymin, args.bev_ymax),
        panel_w=args.bev_width,
        panel_h=args.bev_height,
        fps=args.video_fps,
        pt_size=args.bev_point_size,
    )
    print(f"[lidar_edit_video] BEV 视频已保存: {out_path}")

    h264_path = os.path.join(args.out_dir, "lidar_edit_bev_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-i", out_path, "-vcodec", "libx264", "-crf", "22", "-y", h264_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[lidar_edit_video] H.264 版本: {h264_path}")
    except Exception as exc:
        print(f"[lidar_edit_video] ffmpeg 转码失败（原始 mp4 仍可用 VLC 播放）: {exc}")


if __name__ == "__main__":
    _main()
