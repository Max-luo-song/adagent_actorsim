"""One metadata-driven UniAD runner for any calibrated observation sequence.

No scene-specific threshold, BEV override, vehicle identity or edited asset is
used. Input is a sequence_report.json with frames[].packet and optional
pretrained_camera_slots. Packets must include explicit column-vector l2g pose.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
from queue import Queue
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
from runtime_configuration import model_input_profile, configured_reliability

SLOTS = {"CAM_FRONT": 0, "CAM_FRONT_RIGHT": 1, "CAM_FRONT_LEFT": 2,
         "CAM_BACK": 3, "CAM_BACK_LEFT": 4, "CAM_BACK_RIGHT": 5,
         "CAM_SIDE_LEFT": 4, "CAM_SIDE_RIGHT": 5}


def synchronize_temporal_state(core, reset=False):
    """Handle UniAD's lazily created tracking BEV, preserving later history."""
    if reset:
        core.test_track_instances = None
        core.prev_bev = None
        core.scene_reliability.reset()
        for field in ("_adapter_scene_token", "_adapter_timestamp"):
            if hasattr(core, field):
                delattr(core, field)
    core.prev_frame_info["prev_bev"] = getattr(core, "prev_bev", None)


def prepare_temporal_frame(core, scene_token, timestamp, expected_dt=None):
    """Initialize once; reset BEV, tracks and evidence on scene change or gap."""
    timestamp = float(timestamp)
    if not scene_token or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Invalid scene token/timestamp")
    previous_scene = getattr(core, "_adapter_scene_token", None)
    previous_time = getattr(core, "_adapter_timestamp", None)
    reset = previous_scene != scene_token or previous_time is None
    if not reset:
        delta = timestamp-previous_time
        if delta <= 0:
            raise ValueError("Sequence timestamps must increase within a scene")
        reset = delta > core.scene_reliability.config.max_gap_seconds
        if not reset and expected_dt is not None and not np.isclose(delta, expected_dt, atol=1e-6):
            raise ValueError("Non-contiguous simulation timestamp")
    synchronize_temporal_state(core, reset=reset)
    core._adapter_scene_token, core._adapter_timestamp = scene_token, timestamp
    return reset


def packet_path(record, source):
    p = Path(record["packet"])
    return p if p.is_absolute() else source.parent/p


def inspect_packet(path, slots, *, perception_preview=False, resize_mode='stretch'):
    with path.open("rb") as stream:
        obs, info = pickle.load(stream)
    from edit_trust import assert_edit_evaluable
    if not perception_preview:
        assert_edit_evaluable(info)
    names = list(obs["rgb"])
    if not names or any(name not in slots for name in names):
        raise ValueError("Supply pretrained_camera_slots metadata for unknown camera names")
    names.sort(key=slots.__getitem__)
    rows = [slots[name] for name in names]
    if len(set(rows)) != len(rows) or any(not isinstance(row, int) or not 0 <= row < 6 for row in rows):
        raise ValueError("Camera embedding slots must be distinct integers in [0,5]")
    profile = model_input_profile()
    from input_geometry import camera_projection
    projections, intrinsics = [], []
    for name in names:
        image = np.asarray(obs["rgb"][name])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB image: " + name)
        camera = info["cam_params"][name]
        k, projection = camera_projection(camera, image.shape, (profile['width'], profile['height']), resize_mode)
        projections.append(projection)
        intrinsics.append(k)
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(info["explicit_l2g_r_mat"])
    pose[:3, 3] = np.asarray(info["explicit_l2g_t"])
    from scene_reliability import _validate_pose
    _validate_pose(pose)
    if info["command"] not in (0, 1, 2) or not math.isfinite(float(info["timestamp"])):
        raise ValueError("Invalid command/timestamp")
    return obs, info, names, rows, projections, intrinsics, pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-report", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = Path(args.sequence_report).resolve()
    sequence = json.loads(source.read_text())
    from model_profiles import resolve_model_profile
    profile_name, effective_profile = resolve_model_profile(sequence)
    from edit_trust_adapter import assert_edit_evaluable
    from perception_preview import preview_requested, QuarantinedPerceptionObserver
    perception_preview = preview_requested(sequence)
    from diagnostic_variants import requested_variants
    variants = requested_variants(sequence, perception_preview)
    if not perception_preview:
        assert_edit_evaluable(sequence)
    records = sequence["frames"]
    if args.max_frames is not None:
        if args.max_frames <= 0:
            parser.error("--max-frames must be positive")
        records = records[:args.max_frames]
    if not records:
        parser.error("Empty sequence")
    slots = sequence.get("pretrained_camera_slots", SLOTS)
    sys.path.insert(0, str(BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad"))
    first = packet_path(records[0], source)
    _, _, names, rows, _, _, _ = inspect_packet(first, slots, perception_preview=perception_preview, resize_mode=effective_profile['resize_mode'])
    for record in records:
        if not packet_path(record, source).is_file():
            raise FileNotFoundError(packet_path(record, source))
    if args.check_only:
        print(json.dumps(dict(status="metadata_preflight_passed", frames=len(records),
                              diagnostic_preview_only=perception_preview, evaluation_admitted=not perception_preview,
                              first_packet=str(first), camera_order=names, embedding_rows=rows,
                              model_profile_name=profile_name, effective_model_input=effective_profile,
                              inference_run=False), indent=2))
        return
    from perception_run_binding import input_binding, implementation_binding
    binding = input_binding(source, records)
    os.environ["MPLCONFIGDIR"] = str(BASE/"outputs/gpu_test_service/matplotlib_cache")
    print("Sequence preflight passed; importing CUDA inference stack; preview:", perception_preview, flush=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; metadata preflight and CPU replay remain available")
    output = Path(args.output).resolve() if args.output else BASE/"outputs"/(
        "reliable_sequence_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(BASE/"waymo23-native-import"))
    import five_uniad

    def infer(model, namespace):
        from dataparser import parse_raw
        from infer_three_pad32 import pad_to_32
        from scene_reliability import _numpy
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from box_geometry import assert_box_class_contract
        box_contract = assert_box_class_contract(LiDARInstance3DBoxes, lambda x: torch.tensor(x, dtype=torch.float32))
        core = getattr(model, "module", model)
        if perception_preview:
            core.scene_reliability = QuarantinedPerceptionObserver()
        core.scene_reliability.config = configured_reliability()
        synchronize_temporal_state(core, reset=True)
        for _, module in model.named_modules():
            if hasattr(module, "num_cams"):
                if module.num_cams != 6:
                    raise ValueError("Expected original six-camera model")
                module.num_cams = len(names)
            if hasattr(module, "cams_embeds"):
                old = module.cams_embeds
                module.cams_embeds = torch.nn.Parameter(old.detach()[rows].clone(), requires_grad=False)
        from model_profiles import apply_model_profile
        apply_model_profile(core, effective_profile)
        def replay(run_output, variant):
            completed = []
            try:
                previous = None
                for index, record in enumerate(records):
                    resize_mode = (effective_profile['resize_mode'] if variants is None else
                                   ('letterbox' if variant.startswith('letterbox') else 'stretch'))
                    obs, info, current_names, current_rows, projections, intrinsics, pose = inspect_packet(packet_path(record, source), slots, perception_preview=perception_preview, resize_mode=resize_mode)
                    if (current_names, current_rows) != (names, rows):
                        raise ValueError("Camera rig changed inside one sequence")
                    stamp = float(info["timestamp"])
                    scene_token = str(record.get("scene_token", info.get("scene_token", source)))
                    if variant == "fresh_frame":
                        synchronize_temporal_state(core, reset=True)
                    prepare_temporal_frame(core, scene_token, stamp)
                    bgr = dict(rgb={name: obs["rgb"][name][..., ::-1].copy() for name in names})
                    profile = effective_profile
                    norm = dict(mean=np.array(profile["mean"]), std=np.array(profile["std"]), to_rgb=profile["to_rgb"])
                    data = parse_raw((bgr, info), names, norm, Queue(), perception_preview=perception_preview, resize_mode=resize_mode)
                    data.pop("raw_imgs", None)
                    pad_to_32(data)
                    data["l2g_r_mat"] = data["l2g_r_mat"].new_tensor(pose[:3, :3].T.copy())
                    data["l2g_t"] = data["l2g_t"].new_tensor(pose[:3, 3])
                    meta = data["img_metas"][0][0]
                    if not (np.allclose(meta['lidar2img'], projections, atol=1e-6)
                            and np.allclose(meta['intrinsics'], intrinsics, atol=1e-6)):
                        raise ValueError('Parser/preflight projection contract differs')
                    meta.update(scene_token=scene_token, lidar2img=projections, intrinsics=intrinsics)
                    bus = meta["can_bus"]
                    bus[:3] = [pose[1, 3], -pose[0, 3], 0.]
                    yaw = float(np.arctan2(pose[1, 0], pose[0, 0])) % (2*np.pi)
                    bus[-2], bus[-1] = yaw, np.degrees(yaw)
                    synchronize_temporal_state(core)
                    from input_geometry import model_input_audit
                    input_audit = model_input_audit(data)
                    began = time.monotonic()
                    from pseudo_lidar_adapter import bind_frame
                    if not perception_preview:
                        bind_frame(core, obs, info)
                    try:
                        with torch.no_grad():
                            result = model(return_loss=False, rescale=True, **data)[0]
                    finally:
                        core.geometry_candidate_refiner = None
                    folder = run_output/f"frame_{index:04d}"
                    folder.mkdir()
                    (folder/'input_geometry.json').write_text(json.dumps(input_audit, indent=2, allow_nan=False))
                    reliability = result["scene_reliability"]
                    report_name = "quarantined_preview.json" if perception_preview else "scene_reliability.json"
                    (folder/report_name).write_text(json.dumps(reliability, indent=2, allow_nan=False))
                    arrays = {out: _numpy(result[key]) for out, key in [
                        ("boxes", "boxes_3d"), ("scores", "scores_3d"), ("labels", "labels_3d"),
                        ("candidate_boxes", "boxes_3d_det"), ("candidate_scores", "scores_3d_det"),
                        ("candidate_labels", "labels_3d_det")]}
                    for out,key in (("boxes_gravity","boxes_3d_gravity"),("candidate_boxes_gravity","boxes_3d_det_gravity")):
                        if result.get(key) is not None: arrays[out] = _numpy(result[key])
                    if "geometry_candidate_boxes" in result:
                        arrays["geometry_candidate_boxes"] = result["geometry_candidate_boxes"]
                        (folder/"geometry_refinement.json").write_text(json.dumps(result["geometry_refinement"], indent=2, allow_nan=False))
                    arrays['decoded_box_origin'] = np.asarray(result.get('decoded_box_origin', 'legacy_unspecified'))
                    arrays['box_convention'] = np.asarray(model_input_profile()['box_convention'])
                    if not perception_preview:
                        arrays["trajectory"] = _numpy(result["planning"]["result_planning"]["sdc_traj"]).reshape(-1, 2)
                    occupancy = result.get("occ", {}).get("seg_out")
                    if occupancy is not None and not perception_preview:
                        arrays["occupancy"] = _numpy(occupancy)
                    np.savez_compressed(folder/"perception.npz", **arrays)
                    completed.append(dict(index=index, frame=record["frame"], scene_token=scene_token, timestamp=stamp, action=reliability.get("action", "evaluation_blocked"),
                                          inference_seconds=time.monotonic()-began))
                    previous = stamp
            finally:
                (run_output/"run_report.json").write_text(json.dumps(dict(
                    diagnostic_variant=variant, model_profile_name=profile_name, effective_model_input=effective_profile, source=str(source), completed_frames=len(completed), requested_frames=len(records),
                    camera_order=names, embedding_rows=rows, records=completed,
                    raw_scores_modified=False, closed_loop_run=False,
                    diagnostic_preview_only=perception_preview, planning_outputs_exported=not perception_preview,
                    control_actions_executed=False, n1_edit_certified=False if perception_preview else None,
                    box_convention=model_input_profile()['box_convention'],
                    box_contract=box_contract,
                    input_binding=binding,
                    implementation_unchanged_during_run=implementation_binding()==binding['implementation_hash'],
                    confidence_calibrated=False, thresholds=[core.track_base.score_thresh, core.track_base.filter_score_thresh]), indent=2))

        if variants is None:
            replay(output, 'reference')
        else:
            from diagnostic_variants import configure_variant, restore_variant
            reports = []
            for variant in variants:
                variant_output = output/variant
                variant_output.mkdir()
                synchronize_temporal_state(core, reset=True)
                switches = configure_variant(core, variant)
                try:
                    print('Diagnostic variant:', variant, flush=True)
                    replay(variant_output, variant)
                finally:
                    restore_variant(switches)
                reports.append(dict(variant=variant, report=str(variant_output/'run_report.json')))
            (output/'variant_report.json').write_text(json.dumps(dict(
                variants=reports, diagnostic_preview_only=True, raw_scores_modified=False,
                planning_outputs_exported=False, control_actions_executed=False,
                production_variant_selected=False), indent=2))

    os.environ["BASE"] = str(BASE)
    bootstrap = output/"bootstrap"
    bootstrap.mkdir()
    (bootstrap/"observation.pkl").symlink_to(first)
    os.environ["FIVE_JOB"] = str(bootstrap)
    five_uniad.infer_five = infer
    print("OUTPUT:", output, flush=True)
    five_uniad.infer()


if __name__ == "__main__":
    main()
