import os, json, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import five_closed_bev_reset as closed
from vehicle_edit import relocate

parser = argparse.ArgumentParser()
parser.add_argument("--distance", type=float, required=True)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--preview", action="store_true")
args = parser.parse_args()
if not 1 <= args.steps <= 198 or not 6 <= args.distance <= 30:
    raise ValueError("steps需为1～198，distance需为6～30米")

base = Path(os.environ["BASE"])
original_factory = closed.five_factory
edit = {}

def front_factory(*a, **kw):
    env = original_factory(*a, **kw)
    try:
        # 自车坐标的正前方点，转换到源场景坐标。
        source_ego = (
            np.linalg.inv(env.target.detach().cpu().numpy())
            @ env.ego_pose()
        )
        target = source_ego @ np.array([0., 0., args.distance, 1.])
        edit.update(relocate(env, 10, float(target[0]), float(-target[1]), preview_only=args.preview))
        edit.update(
            initial_forward_distance_m=args.distance,
            initial_lateral_offset_m=0.,
            original_motion_preserved=True,
            orientation_changed=False,
            height_changed=False,
            placement_quality_verified=False
        )
        print("FRONT EDIT:", json.dumps(edit), flush=True)
        return env
    except Exception:
        env.close()
        raise

closed.five_factory = front_factory

if args.preview:
    from five_layout import compose_five
    out = base/"outputs"/(
        f"front_vehicle_{args.distance:g}m_preview_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir()
    env = front_factory(base/"code/HUGSIM", None, None, scale=.5)
    try:
        obs, info = env.reset()
        for name, rgb in obs["rgb"].items():
            Image.fromarray(rgb).save(out/(name+".png"))
        compose_five(out)
        (out/"edit.json").write_text(json.dumps(edit, indent=2))
    finally:
        env.close()
    print("正前方图片：", out/"CAM_FRONT.png")
    print("五视角图片：", out/"five_views.png")
else:
    scope = closed.runtime()
    out = scope["run"](args.steps, None)
    closed.make_video(out)
    path = Path(out)/"closed_report.json"
    report = json.loads(path.read_text())
    report["vehicle_position_edit"] = edit
    path.write_text(json.dumps(report, indent=2))
    print("OUTPUT:", out)
