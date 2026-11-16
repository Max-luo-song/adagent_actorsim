import os, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import five_uniad
from uniad_single import TAIL


def pad_to_32(data):
    image = data["img"]
    h, w = image.shape[-2:]
    padded_h = (h+31)//32*32
    padded_w = (w+31)//32*32
    data["img"] = F.pad(
        image, (0, padded_w-w, 0, padded_h-h),
        mode="constant", value=0
    )

    for metadata in data["img_metas"][0]:
        shapes = metadata["img_shape"]
        original = [list(shape) for shape in shapes]
        padded = [
            [padded_h, padded_w, int(shape[2])]
            for shape in shapes
        ]
        metadata.setdefault("ori_shape", original)
        metadata["img_shape"] = padded
        metadata["pad_shape"] = padded

    if padded_h <= 0 or padded_w <= 0:
        raise ValueError("Invalid input image dimensions")
    print(
        f"Pad32: {h}x{w} -> {padded_h}x{padded_w}; "
        "intrinsics unchanged", flush=True
    )


def infer_three(model, namespace):
    job = Path(os.environ["FIVE_JOB"])
    previous = json.loads(
        (job/"prediction/uniad_report.json").read_text()
    )
    center = np.asarray(previous["asset_center_xy"])
    asset = json.loads((job.parent/"asset_report.json").read_text())
    out = job/"prediction_pad32"
    out.mkdir(exist_ok=True)
    os.environ["UNIAD_SMOKE_OUTPUT"] = str(out)

    core = getattr(model, "module", model)
    thresholds = [
        float(core.track_base.score_thresh),
        float(core.track_base.filter_score_thresh)
    ]
    assert np.allclose(thresholds, [0.4, 0.35])
    assert core.test_track_instances is None

    for name, module in list(model.named_modules()):
        if hasattr(module, "num_cams"):
            assert module.num_cams == 6, name
            module.num_cams = 3
        if hasattr(module, "cams_embeds"):
            old = module.cams_embeds
            assert old.shape[0] == 6
            module.cams_embeds = torch.nn.Parameter(
                old.detach()[:3].clone(), requires_grad=False
            )
    core.prev_bev = None
    core.prev_frame_info["prev_bev"] = None

    order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]
    old = (
        "_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n"
        "            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']"
    )
    assert TAIL.count(old) == 1
    tail = TAIL.replace(old, "_cameras = "+repr(order))
    tail = tail.replace("virtual_rear_views=True", "virtual_rear_views=False")

    marker = "print('Input shape:',tuple(_data['img'].shape),flush=True)"
    assert tail.count(marker) == 1
    tail = tail.replace(marker, "_pad_to_32(_data)\n"+marker)
    scope = dict(namespace, model=model, _pad_to_32=pad_to_32)
    exec(compile(tail, "three_camera_pad32", "exec"), scope)
    result = scope["_results"][0]

    def array(value):
        if hasattr(value, "tensor"):
            value = value.tensor
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    saved = {
        "boxes": array(result["boxes_3d"]),
        "scores": array(result["scores_3d"]),
        "labels": array(result["labels_3d"]),
        "trajectory": array(
            result["planning"]["result_planning"]["sdc_traj"]
        ).reshape(-1,2),
    }
    for key, name in [
        ("boxes_3d_det", "candidate_boxes"),
        ("scores_3d_det", "candidate_scores"),
        ("labels_3d_det", "candidate_labels"),
    ]:
        if key in result:
            saved[name] = array(result[key])
    occ = result.get("occ", {}).get("seg_out")
    if occ is not None:
        saved["occupancy"] = array(occ)
    np.savez_compressed(out/"perception.npz", **saved)

    b, s, l = saved["boxes"], saved["scores"], saved["labels"]
    distance = np.linalg.norm(b[:,:2]-center, axis=1)
    near = np.flatnonzero((l == 0) & (distance < 4))
    print("\n目标中心XY：", center.round(3).tolist())
    print("目标4米内保留车辆框数：", len(near))
    for i in near:
        print("中心XY：", b[i,:2].round(3).tolist(),
              "分数：", round(float(s[i]),4),
              "距离：", round(float(distance[i]),3))

    candidates = []
    if "candidate_boxes" in saved:
        cb, cs, cl = (
            saved["candidate_boxes"],
            saved["candidate_scores"],
            saved["candidate_labels"]
        )
        d = np.linalg.norm(cb[:,:2]-center, axis=1)
        ids = np.flatnonzero((cl == 0) & (d < 4))
        ids = ids[np.argsort(-cs[ids])[:5]]
        print("目标4米内车辆候选：")
        for i in ids:
            candidates.append({
                "index": int(i), "score": float(cs[i]),
                "distance_m": float(d[i]), "box": cb[i].tolist()
            })
            print("编号", int(i), "分数", round(float(cs[i]),4),
                  "距离", round(float(d[i]),3),
                  "中心XY", cb[i,:2].round(3).tolist(),
                  "yaw", round(float(cb[i,6]),3))

    counts = None
    if "occupancy" in saved:
        grid = saved["occupancy"].reshape(5,200,200)
        yy, xx = np.mgrid[:200,:200]
        width, length, _ = asset["asset_wlh"]
        roi = (
            (np.abs((xx-100)*0.5+0.25-center[0]) < width/2+1)
            & (np.abs((yy-100)*0.5+0.25-center[1]) < length/2+1)
        )
        counts = [int(g[roi].sum()) for g in grid]

    print("目标附近占据格数：", counts)
    print("规划轨迹：\n", saved["trajectory"])

    report = dict(scope["_report"])
    report.update(
        status="three-camera pad32 diagnostic completed",
        inference_camera_count=3,
        pretrained_camera_count=6,
        pad_size_divisor=32,
        padding_after_normalization=True,
        padding_value=0,
        detection_thresholds=thresholds,
        asset_center_xy=center.tolist(),
        nearby_retained_car_count=len(near),
        nearby_candidates=candidates,
        occupancy_roi_counts=counts,
        fresh_tracking_state=True,
        bev_history_used=False,
        trained_or_finetuned_on_waymo=False,
        avoidance_validated=False,
    )
    (out/"uniad_report.json").write_text(json.dumps(report, indent=2))
    print("结果目录：", out)


if __name__ == "__main__":
    five_uniad.infer_five = infer_three
    five_uniad.infer()
