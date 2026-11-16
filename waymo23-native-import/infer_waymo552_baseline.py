import os
import sys
import io
import csv
import json
import copy
import pickle
import inspect
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
import torch

base = Path(os.environ["BASE"])
sys.path[:0] = [
    str(base/"waymo23-native-import"),
    str(base/"code/UniAD_SIM"),
]
import five_uniad
from uniad_single import TAIL
from infer_three_pad32 import pad_to_32

job = Path((
    base/"outputs/latest_waymo552_baseline_sequence.txt"
).read_text().strip())
sequence = json.loads((job/"sequence_report.json").read_text())
records = sequence["frames"]
assert [int(r["frame"]) for r in records] == list(range(0,151,5))

out = job/(
    "continuous_baseline_"+datetime.now().strftime("%Y%m%d_%H%M%S")
)
out.mkdir()
print("OUTPUT：",out,flush=True)

setup = Path((base/"outputs/latest_waymo552_setup.txt").read_text().strip())
source = json.loads((setup/"source_report.json").read_text())
inputs = Path(source["inputs"])
start = int(source["start_timestep"])
count = int(source["frames"])

poses = np.stack([
    np.loadtxt(inputs/"ego_pose"/f"{start+i:03d}.txt")
    for i in range(count)
])
relative = np.linalg.inv(poses[0])[None]@poses
C = np.asarray(sequence["virtual_to_ego"],dtype=float)
B = np.array([
    [1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]
],dtype=float)
T = np.array([
    [0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]
],dtype=float)
virtual_poses = np.stack([
    B@T@p@np.linalg.inv(T)@np.linalg.inv(B)@C
    for p in relative
])

def array(value):
    if hasattr(value,"tensor"):
        value = value.tensor
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def recorded_future(frame,info):
    future_ids = frame+np.arange(1,7)*5
    assert future_ids[-1] < len(virtual_poses)
    current_pose = np.eye(4)
    current_pose[:3,:3] = info["explicit_l2g_r_mat"]
    current_pose[:3,3] = info["explicit_l2g_t"]
    assert np.allclose(current_pose,virtual_poses[frame],atol=1e-5)
    return np.stack([
        (np.linalg.inv(current_pose)@virtual_poses[i])[:2,3]
        for i in future_ids
    ])

def infer(model,namespace):
    core = getattr(model,"module",model)
    head = core.planning_head
    assert not head.training
    assert core.video_test_mode
    assert core.test_track_instances is None
    assert np.allclose([
        core.track_base.score_thresh,
        core.track_base.filter_score_thresh
    ],[0.4,0.35])

    for name,module in list(model.named_modules()):
        if hasattr(module,"num_cams"):
            assert module.num_cams == 6,name
            module.num_cams = 3
        if hasattr(module,"cams_embeds"):
            old = module.cams_embeds
            assert old.shape[0] == 6
            module.cams_embeds = torch.nn.Parameter(
                old.detach()[:3].clone(),requires_grad=False
            )

    core.prev_bev = None
    core.prev_frame_info["prev_bev"] = None
    original_bevs = core.get_bevs
    original_forward = head.forward
    old_collision_setting = bool(head.use_col_optim)

    # 对比规划头原始输出，排除碰撞后处理的影响。
    head.use_col_optim = False
    current = {}

    def get_bevs(*args,**kwargs):
        bound = inspect.signature(original_bevs).bind(*args,**kwargs)
        bound.apply_defaults()
        history = original_bevs(*args,**kwargs)

        fresh_args = dict(bound.arguments)
        fresh_args.update(
            prev_bev=None,prev_img=None,prev_img_metas=None
        )
        metas = copy.deepcopy(fresh_args["img_metas"])
        for metadata in metas:
            bus = metadata["can_bus"].copy()
            bus[:3] = 0
            bus[-1] = 0
            metadata["can_bus"] = bus
        fresh_args["img_metas"] = metas
        fresh = original_bevs(**fresh_args)

        current["fresh_bev"],current["fresh_pos"] = fresh
        h,f = history[0].float(),fresh[0].float()
        current["bev_relative_difference"] = float(
            torch.linalg.vector_norm(h-f)/
            torch.linalg.vector_norm(h).clamp_min(1e-9)
        )
        return history

    def planning(*args,**kwargs):
        bound = inspect.signature(original_forward).bind(*args,**kwargs)
        arguments = dict(bound.arguments)
        history_result = original_forward(**arguments)

        fresh_arguments = dict(arguments)
        fresh_arguments["bev_embed"] = current["fresh_bev"]
        fresh_arguments["bev_pos"] = current["fresh_pos"]
        fresh_result = original_forward(**fresh_arguments)

        current["history_plan"] = array(
            history_result["sdc_traj"]
        ).reshape(6,2).copy()
        current["fresh_plan"] = array(
            fresh_result["sdc_traj"]
        ).reshape(6,2).copy()

        # 原流水线继续使用历史BEV规划结果。
        return history_result

    core.get_bevs = get_bevs
    head.forward = planning

    old = (
        "_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n"
        "            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']"
    )
    marker = "print('Input shape:',tuple(_data['img'].shape),flush=True)"
    assert TAIL.count(old) == TAIL.count(marker) == 1
    tail = TAIL.replace(
        old,"_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']"
    ).replace("virtual_rear_views=True","virtual_rear_views=False")
    tail = tail.replace(marker,"_prepare(_data)\n"+marker)
    code = compile(tail,"waymo552_continuous_baseline","exec")

    summaries = []
    completed = False

    def save_report():
        path = out/"baseline_report.json"
        path.write_text(json.dumps({
            "scene":552,
            "sequence":str(job),
            "completed":completed,
            "frames":summaries,
            "uniad_model_loads":1,
            "perception_bev":"history enabled",
            "comparison":"same SDC queries; only planning BEV changed",
            "planning_outputs":"raw; collision postprocessor disabled",
            "original_use_col_optim":old_collision_setting,
            "thresholds":[0.4,0.35],
            "navigation_command":2,
            "source_fps_assumption":10.,
            "observation_sampling_hz":2.,
            "scene_edits":[],
            "sky_and_affine_enabled":False,
            "closed_loop_tested":False,
        },indent=2))
        return path

    print(
        "帧号 速度 录制终点Y 历史终点Y 当前终点Y "
        "历史ADE 当前ADE 保留车辆数",
        flush=True
    )
    try:
        for record in records:
            frame = int(record["frame"])
            packet = Path(record["packet"])
            with packet.open("rb") as f:
                _,info = pickle.load(f)

            def prepare(data):
                pad_to_32(data)
                rotation = np.asarray(info["explicit_l2g_r_mat"])
                translation = np.asarray(info["explicit_l2g_t"])
                data["l2g_r_mat"] = data["l2g_r_mat"].new_tensor(
                    rotation.T.copy()
                )
                data["l2g_t"] = data["l2g_t"].new_tensor(translation)
                metadata = data["img_metas"][0][0]
                metadata["scene_token"] = "waymo552_"+out.name
                bus = metadata["can_bus"]
                bus[:3] = [translation[1],-translation[0],0.]
                yaw = float(np.arctan2(
                    rotation[1,0],rotation[0,0]
                ))%(2*np.pi)
                bus[-2],bus[-1] = yaw,np.degrees(yaw)

            folder = out/f"frame_{frame:03d}"
            folder.mkdir()
            os.environ["UNIAD_PACKET"] = str(packet)
            os.environ["UNIAD_SMOKE_OUTPUT"] = str(folder)
            core.prev_frame_info["prev_bev"] = core.prev_bev
            current.clear()

            scope = dict(namespace,model=model,_prepare=prepare)
            log = io.StringIO()
            try:
                with redirect_stdout(log),torch.no_grad():
                    exec(code,scope)
            finally:
                (folder/"inference.log").write_text(log.getvalue())

            result = scope["_results"][0]
            (folder/"scene_reliability.json").write_text(json.dumps(
                result["scene_reliability"], indent=2, allow_nan=False))
            history_plan = current["history_plan"]
            fresh_plan = current["fresh_plan"]
            truth = recorded_future(frame,info)
            assert np.isfinite(history_plan).all()
            assert np.isfinite(fresh_plan).all()

            labels = array(result["labels_3d"]).reshape(-1)
            scores = array(result["scores_3d"]).reshape(-1)
            boxes = array(result["boxes_3d"])
            occ = result.get("occ",{}).get("seg_out")
            assert occ is not None,"没有占据输出"
            occ = array(occ)
            assert occ.size == 5*200*200,occ.shape
            counts = [
                int(np.count_nonzero(g))
                for g in occ.reshape(5,200,200)
            ]

            # car/truck/construction_vehicle/bus/trailer
            vehicle_count = int(np.isin(labels,[0,1,2,3,4]).sum())
            h_error = np.linalg.norm(history_plan-truth,axis=1)
            f_error = np.linalg.norm(fresh_plan-truth,axis=1)

            np.savez_compressed(
                folder/"diagnostic.npz",
                history_plan=history_plan,
                current_frame_plan=fresh_plan,
                recorded_future_lidar_xy=truth,
                occupancy=occ,boxes=boxes,scores=scores,labels=labels,
                candidate_boxes=array(result["boxes_3d_det"]),
                candidate_scores=array(result["scores_3d_det"]),
                candidate_labels=array(result["labels_3d_det"])
            )
            row = {
                "frame":frame,
                "time_s":float(info["timestamp"]),
                "speed_mps":float(info["ego_velo"]),
                "recorded_last_forward_m":float(truth[-1,1]),
                "history_last_forward_m":float(history_plan[-1,1]),
                "current_last_forward_m":float(fresh_plan[-1,1]),
                "history_ade_m":float(h_error.mean()),
                "current_ade_m":float(f_error.mean()),
                "history_fde_m":float(h_error[-1]),
                "current_fde_m":float(f_error[-1]),
                "bev_relative_difference":current["bev_relative_difference"],
                "retained_vehicle_count":vehicle_count,
                "retained_total_count":int(len(labels)),
                "occupancy_counts":counts,
            }
            summaries.append(row)
            save_report()
            print(
                f'{frame:3d} {row["speed_mps"]:5.2f} '
                f'{truth[-1,1]:9.3f} {history_plan[-1,1]:9.3f} '
                f'{fresh_plan[-1,1]:9.3f} '
                f'{h_error.mean():7.3f} {f_error.mean():7.3f} '
                f'{vehicle_count:4d}',
                flush=True
            )
            del result,scope
            current.clear()

        completed = True

    finally:
        core.get_bevs = original_bevs
        head.forward = original_forward
        head.use_col_optim = old_collision_setting
        path = save_report()
        print("\n报告：",path,flush=True)

    fields = [key for key in summaries[0] if key != "occupancy_counts"]
    with (out/"baseline_metrics.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    print("\n=== 全序列汇总 ===",flush=True)
    print("已完成帧数：",len(summaries),flush=True)
    print("历史BEV平均ADE(m)：",np.mean([
        r["history_ade_m"] for r in summaries
    ]),flush=True)
    print("当前帧BEV平均ADE(m)：",np.mean([
        r["current_ade_m"] for r in summaries
    ]),flush=True)
    print("逐帧指标：",out/"baseline_metrics.csv",flush=True)


if __name__ == "__main__":
    os.environ["FIVE_JOB"] = str(Path(records[0]["packet"]).parent)
    five_uniad.infer_five = infer
    print("开始单独加载UniAD；本进程不加载场景重建模型。",flush=True)
    five_uniad.infer()
