"""运行中动态轨迹接入验证探针（pose_track_probe_v1）。

A/B 两臂同一短窗口：A 不编辑；B 在观测 1→2 帧边界 insert 一辆跟踪车，
之后每个良性 step 后按观测包 timestamp_s 推进轨迹（不用墙钟），逐步做
expected/registry/backend 三方位姿读回核对。最后验证越界拒绝、重复插入拒绝、
remove 回退渲染与确定性重渲染。

纯渲染循环：本探针不运行 UniAD 推理，不验证模型历史连续性；cache token
序列仅作机制级证据。不驾驶、无 MPC、不出证书；全部输出标记 diagnostic。
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
Path = _HugsimPath

import argparse
import hashlib
import json
import signal
import sys
import time
from datetime import datetime, timezone
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path(_hugsim_expand('${HUGSIM_ROOT}'))
JOB = BASE / "outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4"
sys.path[:0] = [str(BASE / "integrations/sr-hugsim"), str(BASE / "waymo23-native-import"),
                str(JOB / "chain_repair/phase26/bridge/pose_track")]
BENIGN_ACTION = {"acc_mps2": 0.0, "steer_rate_radps": 0.0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_sources(sources):
    for path, want in sources.items():
        p = Path(path)
        if not p.is_file() or sha(p) != want:
            raise ValueError("Pinned source missing or changed: " + str(path))


def packet_digest(packet):
    cams = {}
    for name, view in sorted(packet["cameras"].items()):
        cams[name] = {k: hashlib.sha256(np.ascontiguousarray(v).view(np.uint8)).hexdigest()
                      for k, v in sorted(view.items()) if isinstance(v, np.ndarray)}
    meta = {k: packet[k] for k in ("timestamp_s", "source_frame", "history_token",
                                   "edit_revision", "model_cache_token") if k in packet}
    timing = {k: packet[k] for k in ("timestamp_s", "source_frame") if k in packet}
    return {"cameras": cams, "meta": meta, "timing": timing,
            "arrays_digest": digest(cams),
            "digest": digest({"cameras": cams, "timing": timing})}


def build_trajectory(plan):
    from pose_track import PoseTrajectory
    spec = plan["trajectory"]
    if spec["unit"] != "meter" or spec["bounds"] not in ("strict", "hold"):
        raise ValueError("trajectory unit/bounds contract violated")
    times = [k["t"] for k in spec["keyframes"]]
    poses = [np.asarray(k["asset_to_world"], dtype=float) for k in spec["keyframes"]]
    traj = PoseTrajectory(times, poses, bounds=spec["bounds"])
    if (trajectory_window := spec["time_window_s"]) != [traj.t_start, traj.t_end]:
        raise ValueError("declared window does not match keyframes")
    return traj


def pose_readback(pts, backend, object_id, t):
    """expected vs registry vs backend (source_pose + native tensor)."""
    from waymo_adapter import SR_TO_HUGSIM as T
    expected = pts.registry.render_pose(object_id)
    collision = pts.registry.collision_pose(object_id)
    asset = backend.assets[object_id]
    native = np.asarray(asset.native_pose.cpu().numpy() if hasattr(asset.native_pose, "cpu")
                        else asset.native_pose, dtype=float)
    return {
        "t": t,
        "max_abs_coordinate_m": float(np.max(np.abs(expected[:3, 3]))),
        "same_array_render_collision": expected is collision,
        "registry_vs_backend_source_max_abs": float(np.max(np.abs(
            expected - np.asarray(asset.source_pose, dtype=float)))),
        "native_vs_expected_max_abs": float(np.max(np.abs(native - T @ expected))),
    }


def run_arm(session, arm, plan, pose_applier=None):
    """单臂流程。pose_applier 仅 GPU 路径注入；CPU 夹具注入记录型假实现。"""
    if arm not in ("A", "B"):
        raise ValueError("arm must be A or B")
    from session_adapter import PoseTrackSession
    events, observations, readbacks = [], [], []

    def log(event, **data):
        events.append({"event": event, "utc": datetime.now(timezone.utc).isoformat(),
                       "perf_s": time.perf_counter(), **data})

    prefix = plan["protocol"]["prefix_observations"]
    total = plan["protocol"]["total_observations"]
    log("arm_start", arm=arm)
    init_pose, init_ts = None, None
    pts = None
    result = {"arm": arm, "observations": observations, "events": events}
    for i in range(total):
        t0 = time.perf_counter()
        packet = session.observe() if i == 0 else session.step(BENIGN_ACTION)
        wall = time.perf_counter() - t0
        d = packet_digest(packet)
        d["observation_index"] = i
        observations.append(d)
        log("observation", index=i, wall_s=wall, digest=d["digest"], meta=d["meta"])
        if i == 0:
            init_pose, init_ts = packet["ego_to_world"], packet["timestamp_s"]
        if arm == "B":
            if i == prefix - 1:
                traj = build_trajectory(plan)
                applier = pose_applier if pose_applier is not None else _null_applier
                pts = PoseTrackSession(session, applier)
                t0 = time.perf_counter()
                ack = pts.insert_tracked("tracked_car", traj, edit_request=dict(plan["edit"]))
                log("insert_committed", wall_s=time.perf_counter() - t0, ack=ack)
                result["insert_ack"] = ack
            elif pts is not None:
                t0 = time.perf_counter()
                poses = pts.advance_pose_track(packet["timestamp_s"])
                adv_wall = time.perf_counter() - t0
                rb = pose_readback(pts, session.backend, "tracked_car", packet["timestamp_s"])
                rb["advance_wall_s"] = adv_wall
                readbacks.append(rb)
                log("advance", index=i, wall_s=adv_wall, **rb)
    if arm == "B":
        # 1) strict out-of-range must raise
        try:
            pts.advance_pose_track(plan["trajectory"]["time_window_s"][1] + 0.5)
            oor = {"rejected": False, "unexpected_success": True}
        except ValueError as exc:
            oor = {"rejected": True, "error": str(exc)}
        log("out_of_range_advance", **oor)
        result["out_of_range_advance"] = oor
        # 2) duplicate insert must be rejected
        try:
            session.edit(dict(plan["edit"]))
            dup = {"rejected": False, "unexpected_success": True}
        except ValueError as exc:
            dup = {"rejected": True, "error": str(exc)}
        log("duplicate_insert_attempt", **dup)
        result["duplicate_insert"] = dup
        # 3) deterministic double render at identical state
        r1 = packet_digest(session.render(init_pose, init_ts))
        r2 = packet_digest(session.render(init_pose, init_ts))
        det = {"identical": r1["cameras"] == r2["cameras"]}
        log("double_render", **det)
        result["double_render"] = det
        # 4) remove + revert render vs baseline initial
        t0 = time.perf_counter()
        rack = pts.remove_tracked("tracked_car",
                                  edit_request={"operation": "remove",
                                                "object_id": "tracked_car",
                                                "mode": "diagnostic_preview"})
        log("remove_committed", wall_s=time.perf_counter() - t0, ack=rack)
        back = packet_digest(session.render(init_pose, init_ts))
        result["revert"] = {"remove_ack": rack, "packet": back}
        log("revert_render")
    result["readbacks"] = readbacks
    result["initial_pose"] = np.asarray(init_pose).tolist()
    result["initial_timestamp_s"] = init_ts
    log("arm_end", arm=arm)
    return result


def _null_applier(object_id, pose):
    raise RuntimeError("CPU fixture must inject its own recording applier")


def evaluate(results, prefix=2):
    """验收评估（纯函数，CPU 可测）。返回 criteria dict + overall passed。"""
    a, b = results["A"], results["B"]
    criteria = {}
    criteria["prefix_identical"] = all(
        a["observations"][i]["digest"] == b["observations"][i]["digest"]
        for i in range(prefix))
    rb = b["readbacks"]
    # native_pose lives in a float32 torch tensor; the tolerance is the
    # float32 representation floor at the stored magnitudes (eps*|x|, with a
    # small factor for the T combination), not a loosened quality threshold.
    def native_tol(r):
        return 8 * 1.19e-7 * max(1.0, r.get("max_abs_coordinate_m", 20.0))
    criteria["readback_exact"] = bool(rb) and all(
        r["same_array_render_collision"]
        and r["registry_vs_backend_source_max_abs"] == 0.0
        and r["native_vs_expected_max_abs"] <= native_tol(r) for r in rb)
    revs = [o["meta"].get("edit_revision") for o in b["observations"]]
    toks = [o["meta"].get("model_cache_token") for o in b["observations"]]
    # insert between obs1 and obs2: revision 0,0,1,1,...; token must follow revision
    criteria["revision_bumps_only_on_edits"] = (
        revs[:2] == [0, 0] and all(r == 1 for r in revs[2:]))
    criteria["token_tracks_revision"] = len(set(toks)) == 2
    post = [b["observations"][i] for i in range(prefix, len(b["observations"]))]
    criteria["motion_visible"] = all(
        post[i]["arrays_digest"] != post[i + 1]["arrays_digest"]
        for i in range(len(post) - 1))
    criteria["out_of_range_rejected"] = b["out_of_range_advance"]["rejected"]
    criteria["duplicate_insert_rejected"] = b["duplicate_insert"]["rejected"]
    criteria["double_render_deterministic"] = b["double_render"]["identical"]
    criteria["revert_matches_baseline"] = (
        b["revert"]["packet"]["cameras"] == a["observations"][0]["cameras"])
    passed = all(criteria.values())
    return {"criteria": criteria, "passed": passed}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    plan = _hugsim_expand(json.loads(Path(args.plan).read_text()))
    if plan.get("schema") != "pose_track_probe_v1":
        raise ValueError("Unexpected plan schema")
    verify_sources(plan["sources"])  # verify BEFORE creating any output
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)

    def interrupted(number, frame):
        raise RuntimeError(f"probe interrupted by signal {number}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    results, failed = {}, None
    for arm in ("A", "B"):
        session = None
        try:
            from scene_api import SceneSession
            session = SceneSession()
            session.load_scene(plan["scene_manifest"])
            applier = None
            if arm == "B":
                from gpu_applier import make_backend_pose_applier
                applier = make_backend_pose_applier(session.backend)
            results[arm] = run_arm(session, arm, plan, pose_applier=applier)
        except BaseException as exc:
            failed = {"arm": arm, "error": repr(exc)}
            write(out / f"failure_arm_{arm}.json",
                  {"state": "discarded", "error": repr(exc),
                   "formal_evaluation_admitted": False, "certificate": None})
            break
        finally:
            if session is not None:
                session.close()
    summary = {"schema": "pose_track_probe_result_v1", "diagnostic_only": True,
               "n1_certified": False, "formal_evaluation_admitted": False,
               "uniad_history_continuity_validated": False,
               "uniad_note": "no model inference ran; cache-token stability is "
                             "mechanism-level evidence only",
               "plan_hash": digest(plan), "failed": failed}
    if failed is None:
        summary["evaluation"] = evaluate(results, prefix=plan["protocol"]["prefix_observations"])
        summary["readbacks_B"] = results["B"]["readbacks"]
        summary["events_B"] = results["B"]["events"]
    write(out / "summary.json", summary)
    print(json.dumps({"output": str(out), "failed": failed,
                      "passed": None if failed else summary["evaluation"]["passed"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
