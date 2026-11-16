import os, json, copy, inspect
from pathlib import Path

import five_uniad
import infer_car_sequence_bev_aligned as runner

source_root = Path(os.environ["SEQUENCE_JOB"])
out = source_root/"planning_current_bev_experiment"
out.mkdir(exist_ok=True)
runner.root = out


def experimental_infer(model,namespace):
    core = getattr(model,"module",model)
    head = core.planning_head
    assert not head.training

    original_bevs = core.get_bevs
    original_planning = head.forward
    current = {}
    used_frames = []

    def get_bevs(*args,**kwargs):
        bound = inspect.signature(original_bevs).bind(*args,**kwargs)
        bound.apply_defaults()

        # 正常历史BEV返回给检测、跟踪和运动预测。
        history_result = original_bevs(*args,**kwargs)

        fresh_args = dict(bound.arguments)
        fresh_args["prev_bev"] = None
        fresh_args["prev_img"] = None
        fresh_args["prev_img_metas"] = None
        metas = copy.deepcopy(fresh_args["img_metas"])
        for metadata in metas:
            bus = metadata["can_bus"].copy()
            bus[:3] = 0
            bus[-1] = 0
            metadata["can_bus"] = bus
        fresh_args["img_metas"] = metas

        current["bev"],current["pos"] = original_bevs(**fresh_args)
        return history_result

    def planning(*args,**kwargs):
        bound = inspect.signature(original_planning).bind(*args,**kwargs)
        arguments = dict(bound.arguments)
        arguments["bev_embed"] = current["bev"]
        arguments["bev_pos"] = current["pos"]
        used_frames.append(int(runner.current_info["source_frame"]))

        # 自车查询、占据和碰撞优化设置沿用原流程。
        return original_planning(**arguments)

    core.get_bevs = get_bevs
    head.forward = planning
    try:
        runner.infer_sequence(model,namespace)
    finally:
        core.get_bevs = original_bevs
        head.forward = original_planning

    assert used_frames == [165,170,175], used_frames
    path = out/"temporal_prediction_bev_aligned/sequence_summary.json"
    report = json.loads(path.read_text())
    report.update(
        experimental_configuration=True,
        perception_bev="history fused",
        planning_bev="current frame only",
        tracking_and_motion_queries="retained from history-enabled pipeline",
        collision_optimization="unchanged",
        model_weights_modified=False,
        closed_loop_tested=False,
        avoidance_validated=False,
    )
    path.write_text(json.dumps(report,indent=2))
    print("实验报告：",path)


if __name__ == "__main__":
    os.environ["FIVE_JOB"] = str(Path(runner.records[0]["packet"]).parent)
    five_uniad.infer_five = experimental_infer
    five_uniad.infer()
