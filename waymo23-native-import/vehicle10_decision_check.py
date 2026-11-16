import os, json
from pathlib import Path
import numpy as np
import five_closed_bev_reset as closed
from vehicle_edit import relocate

original_factory = closed.five_factory
edit_report = {}

def edited_factory(*args, **kwargs):
    env = original_factory(*args, **kwargs)
    try:
        position = env.models["RigidNodes"]["instances_trans"][0,10]
        position = position.detach().cpu().numpy()
        native_to_source = np.linalg.inv(
            env.target.detach().cpu().numpy()
        )
        camera = native_to_source @ env.ego_pose() @ env.cameras[0]["rig"]
        camera_xy = camera[:2,3]
        target = camera_xy + .6*(position[:2]-camera_xy)
        edit_report.update(
            relocate(env, 10, float(target[0]), float(-target[1]))
        )
        edit_report.update(
            distance_ratio=.6,
            orientation_changed=False,
            height_changed=False,
            visual_quality_verified=False
        )
        print("VEHICLE EDIT:", json.dumps(edit_report), flush=True)
        return env
    except Exception:
        env.close()
        raise

closed.five_factory = edited_factory
scope = closed.runtime()
steps = int(os.environ.get("VEHICLE_STEPS", "1"))
if not 1 <= steps <= 10:
    raise ValueError("VEHICLE_STEPS 必须为 1～10")
folder = scope["run"](steps, None)
closed.make_video(folder)

path = Path(folder) / "closed_report.json"
report = json.loads(path.read_text())
report["vehicle_position_edit"] = edit_report
path.write_text(json.dumps(report, indent=2))
(Path(folder) / "vehicle_edit.json").write_text(
    json.dumps(edit_report, indent=2)
)
print("OUTPUT:", folder)
