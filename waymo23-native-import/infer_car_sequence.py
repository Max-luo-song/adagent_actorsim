import os, json, pickle, io
from pathlib import Path
from contextlib import redirect_stdout
from itertools import product

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mmdet3d.core.bbox import LiDARInstance3DBoxes

import five_uniad
from uniad_single import TAIL
from infer_three_pad32 import pad_to_32
from five_ablation_policy import apply_policy

root = Path(os.environ["SEQUENCE_JOB"])
sequence = json.loads((root/"sequence_report.json").read_text())
asset = json.loads((root/"asset_report.json").read_text())
records = sequence["frames"]
current_info = None


def array(value):
    if hasattr(value, "tensor"):
        value = value.tensor
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def hull(points):
    points = np.array(points, dtype=np.float32, copy=True).reshape(-1,2)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("无效的俯视角点")
    # 强制连续内存，并采用OpenCV标准点集形状。
    points = np.ascontiguousarray(points.reshape(-1,1,2))
    return cv2.convexHull(points)


def overlap(a,b):
    intersection, _ = cv2.intersectConvexConvex(a,b)
    union = abs(cv2.contourArea(a))+abs(cv2.contourArea(b))-intersection
    return float(intersection/max(union,1e-8))


def reference(info):
    ego = np.eye(4)
    ego[:3,:3] = Rotation.from_euler("XYZ",info["ego_rot"]).as_matrix()
    ego[:3,3] = info["ego_pos"]
    basis = np.eye(4)
    basis[:3,:3] = [[1,0,0],[0,0,1],[0,-1,0]]
    c = np.asarray(info["model_lidar_frame"]["virtual_to_ego"])
    placement = next(
        p for p in asset["placements"] if p["case"] == "heading_A"
    )
    transform = (
        np.linalg.inv(c) @ basis @ np.linalg.inv(ego)
        @ np.asarray(placement["asset_to_native"])
    )
    w,length,height = asset["asset_wlh"]
    bottom = asset["local_bottom_y"]
    points = np.array([
        [x,y,z,1.]
        for x,y,z in product(
            [-length/2,length/2],
            [bottom-height,bottom],
            [-w/2,w/2]
        )
    ]) @ transform.T
    return points[:,:3].mean(0), hull(points[:,:2])


def summarize(boxes,scores,labels,truth,center):
    indices = np.flatnonzero(labels == 0)
    if not len(indices):
        return []
    corners = LiDARInstance3DBoxes(
        torch.as_tensor(boxes[indices],dtype=torch.float32),
        box_dim=boxes.shape[1]
    ).corners.cpu().numpy()
    items = []
    for index,points in zip(indices,corners):
        distance = float(np.linalg.norm(boxes[index,:2]-center[:2]))
        if distance <= 6:
            items.append({
                "index":int(index),
                "score":float(scores[index]),
                "distance_m":distance,
                "bev_iou":overlap(hull(points[:,:2]),truth),
                "box":boxes[index].tolist(),
            })
    return sorted(items,key=lambda x:x["bev_iou"],reverse=True)


def prepare(data):
    pad_to_32(data)
    data["l2g_r_mat"] = data["l2g_r_mat"].new_tensor(
        current_info["explicit_l2g_r_mat"]
    )
    data["l2g_t"] = data["l2g_t"].new_tensor(
        current_info["explicit_l2g_t"]
    )
    data["img_metas"][0][0]["scene_token"] = "waymo172_static_asset_sequence"


def infer_sequence(model,namespace):
    global current_info
    core = getattr(model,"module",model)
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
    if not hasattr(core,"prev_bev"):
        core.prev_bev = None

    old = (
        "_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n"
        "            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']"
    )
    assert TAIL.count(old) == 1
    tail = TAIL.replace(
        old,"_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']"
    ).replace("virtual_rear_views=True","virtual_rear_views=False")
    marker = "print('Input shape:',tuple(_data['img'].shape),flush=True)"
    assert tail.count(marker) == 1
    tail = tail.replace(marker,"_prepare(_data)\n"+marker)
    code = compile(tail,"three_camera_sequence","exec")
    summaries = []

    print("帧号  保留框最佳IoU/分数  候选最佳IoU/分数  附近占据",flush=True)
    for record in records:
        packet = Path(record["packet"])
        with packet.open("rb") as f:
            _,current_info = pickle.load(f)
        frame = record["frame"]
        output = root/"temporal_prediction_fixed"/f"frame_{frame:03d}"
        output.mkdir(parents=True,exist_ok=True)
        os.environ["UNIAD_PACKET"] = str(packet)
        os.environ["UNIAD_SMOKE_OUTPUT"] = str(output)

        apply_policy(model,"bev_reset")
        scope = dict(namespace,model=model,_prepare=prepare)
        log = io.StringIO()
        try:
            with redirect_stdout(log):
                exec(code,scope)
        finally:
            (output/"inference.log").write_text(log.getvalue())

        result = scope["_results"][0]
        saved = {
            "boxes":array(result["boxes_3d"]),
            "scores":array(result["scores_3d"]),
            "labels":array(result["labels_3d"]),
            "candidate_boxes":array(result["boxes_3d_det"]),
            "candidate_scores":array(result["scores_3d_det"]),
            "candidate_labels":array(result["labels_3d_det"]),
            "trajectory":array(
                result["planning"]["result_planning"]["sdc_traj"]
            ).reshape(-1,2),
        }
        occ = result.get("occ",{}).get("seg_out")
        if occ is not None:
            saved["occupancy"] = array(occ)
        np.savez_compressed(output/"perception.npz",**saved)

        center,truth = reference(current_info)
        kept = summarize(
            saved["boxes"],saved["scores"],saved["labels"],truth,center
        )
        candidates = summarize(
            saved["candidate_boxes"],saved["candidate_scores"],
            saved["candidate_labels"],truth,center
        )
        counts = None
        if "occupancy" in saved:
            mask = np.zeros((200,200),dtype=np.uint8)
            pixels = np.ascontiguousarray(
                np.rint(truth.reshape(-1,2)*2+99.5).astype(np.int32)
            )
            cv2.fillConvexPoly(mask,pixels,1)
            mask = cv2.dilate(mask,np.ones((5,5),np.uint8)).astype(bool)
            counts = [
                int(g[mask].sum())
                for g in saved["occupancy"].reshape(5,200,200)
            ]

        def short(items):
            return (f"{items[0]['bev_iou']:.3f}/{items[0]['score']:.3f}"
                    if items else "-")
        print(frame,short(kept),short(candidates),counts,flush=True)
        summaries.append({
            "frame":frame,
            "target_center_virtual_xyz":center.tolist(),
            "nearby_retained_boxes":kept,
            "nearby_candidates":candidates,
            "occupancy_near_footprint":counts,
            "trajectory_model_xy":saved["trajectory"].tolist(),
        })

    report = {
        "frames":summaries,
        "uniad_model_loads":1,
        "bev_history_enabled":False,
        "tracking_queries_retained":True,
        "full_recorded_l2g_used":True,
        "thresholds":[0.4,0.35],
        "association_method":"geometric overlap; identity not independently verified",
        "planning_reference_validated":False,
        "closed_loop_tested":False,
        "avoidance_validated":False,
    }
    path = root/"temporal_prediction_fixed/sequence_summary.json"
    path.write_text(json.dumps(report,indent=2))
    print("连续推理报告：",path)


if __name__ == "__main__":
    # 加载模型前，先验证本次出错的几何统计。
    with Path(records[0]["packet"]).open("rb") as f:
        _,first_info = pickle.load(f)
    center,truth = reference(first_info)
    assert abs(overlap(truth,truth)-1) < 1e-4

    old_result = root/"temporal_prediction/frame_165/perception.npz"
    if old_result.is_file():
        old = np.load(old_result)
        summarize(old["boxes"],old["scores"],old["labels"],truth,center)
        summarize(
            old["candidate_boxes"],old["candidate_scores"],
            old["candidate_labels"],truth,center
        )
    print("连续内存与几何统计检查通过",flush=True)

    os.environ["FIVE_JOB"] = str(Path(records[0]["packet"]).parent)
    five_uniad.infer_five = infer_sequence
    five_uniad.infer()
