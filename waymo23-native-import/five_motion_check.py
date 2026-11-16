import os, sys, pickle, types, json
from pathlib import Path
import numpy as np

tools = Path(__file__).resolve().parent
sys.path.insert(0, str(tools))
import five_ablation_policy as policy

original_policy = policy.apply_policy

def apply(model, mode):
    original_policy(model, "history" if mode == "motion_fixed" else mode)
    core = getattr(model, "module", model)
    if getattr(core, "_motion_check_installed", False):
        return
    original = core.get_bevs

    def measured(self, *args, **kwargs):
        if os.environ["ABLATION_MODE"] == "motion_fixed":
            metas = kwargs.get("img_metas")
            if metas is None:
                metas = args[1]
            meta = metas[0]
            packet = Path(os.environ["UNIAD_PACKET"])
            with packet.open("rb") as f:
                _, info = pickle.load(f)

            # native: right/down/forward -> IMU: forward/left/up
            position = np.array(info["ego_pos"], dtype=float)[[2, 0, 1]]
            position *= np.array([1., -1., -1.])
            yaw = -float(info["ego_rot"][1])
            token = meta["scene_token"]
            previous = getattr(self, "_motion_check_previous", None)
            delta = np.zeros(3)
            angle = 0.
            if previous is not None and previous[0] == token:
                delta = position - previous[1]
                angle = np.degrees(
                    (yaw - previous[2] + np.pi) % (2*np.pi) - np.pi
                )
            self._motion_check_previous = (token, position.copy(), yaw)
            meta["can_bus"][:3] = delta
            meta["can_bus"][-1] = angle
            row = dict(packet=str(packet), delta_m=delta.tolist(),
                       delta_yaw_degrees=float(angle))
            log = Path(os.environ["REPLAY_JOB"]) / "motion_applied.jsonl"
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return original(*args, **kwargs)

    core.get_bevs = types.MethodType(measured, core)
    core._motion_check_installed = True

policy.apply_policy = apply

path = tools / "five_ablation.py"
source = path.read_text()
old = "for mode in " + repr(policy.MODES) + ":"
assert source.count(old) == 1, "脚本结构不同，请贴出报错"
source = source.replace(
    old,
    "for mode in ('history','motion_fixed','bev_reset','new_scene_each_frame'):"
)
source = source.replace("five_history_ablation_", "five_motion_check_")
exec(compile(source, str(path), "exec"),
     {"__name__": "__main__", "__file__": str(path)})
