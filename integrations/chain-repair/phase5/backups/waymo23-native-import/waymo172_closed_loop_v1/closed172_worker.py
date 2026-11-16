"""Resident legacy UniAD process. Launched by closed172_run.py only."""

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
import copy
import inspect
import io
import json
import os
import pickle
import sys
import time
import traceback
from contextlib import redirect_stdout
pass  # Path is provided by the portable boundary above

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from startup_timing import atomic_json,configure,mark
mark('worker_started')

if os.environ.get('UNIAD_INIT_TRACE')=='1':
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120,repeat=True)
    print('UniAD bootstrap: importing numerical/runtime dependencies',flush=True)

if __name__=='__main__':
    from runtime_bundle import activate_configured
    mark('runtime_cache_check_started')
    runtime_cache=activate_configured()
    if runtime_cache is not None:mark('runtime_cache_ready',runtime=runtime_cache)

import numpy as np
mark('numpy_loaded')

JOB = None
PLANNING_BEV = "history"
MODEL_PROFILE = "legacy"


def serve(model, namespace):
    mark('worker_configuration_started')
    if os.environ.get('UNIAD_INIT_TRACE')=='1':
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        print('UniAD bootstrap: model loaded; configuring inference',flush=True)
    import torch
    from uniad_single import TAIL
    from infer_three_pad32 import pad_to_32

    from runtime_configuration import configured_reliability
    core = getattr(model,"module",model)
    core.scene_reliability.config = configured_reliability()
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from box_geometry import assert_box_class_contract
    box_contract = assert_box_class_contract(LiDARInstance3DBoxes, lambda x: torch.tensor(x, dtype=torch.float32))
    if not np.allclose([core.track_base.score_thresh,
                        core.track_base.filter_score_thresh],[.4,.35]):
        raise RuntimeError("Detection thresholds differ from validated setup")
    if not core.video_test_mode or core.test_track_instances is not None:
        raise RuntimeError("Expected fresh video-mode model")
    adaptations = []
    for name,module in list(model.named_modules()):
        if hasattr(module,"num_cams"):
            if module.num_cams != 6:
                raise RuntimeError("Unexpected camera count: " + name)
            module.num_cams = 3
            adaptations.append(name)
        if hasattr(module,"cams_embeds"):
            old = module.cams_embeds
            if old.shape[0] != 6:
                raise RuntimeError("Unexpected camera embedding")
            module.cams_embeds = torch.nn.Parameter(old.detach()[:3].clone(),requires_grad=False)
    if not adaptations:
        raise RuntimeError("No camera modules adapted")
    core.prev_bev = None
    core.prev_frame_info["prev_bev"] = None
    from model_profiles import resolve_model_profile, apply_model_profile
    _, effective_profile = resolve_model_profile(dict(model_input_profile=MODEL_PROFILE))
    apply_model_profile(core, effective_profile)
    head = core.planning_head
    if head.training:
        raise RuntimeError("Planning head is in training mode")
    # The external motion-constrained MPC replaces, rather than duplicates,
    # the original pointwise collision postprocessor.
    head.use_col_optim = False
    original_bevs = core.get_bevs
    original_forward = head.forward
    current = {}

    def get_bevs(*args,**kwargs):
        bound = inspect.signature(original_bevs).bind(*args,**kwargs)
        bound.apply_defaults()
        history = original_bevs(*args,**kwargs)
        if PLANNING_BEV == "history":
            return history
        fresh = dict(bound.arguments)
        fresh.update(prev_bev=None,prev_img=None,prev_img_metas=None)
        metas = copy.deepcopy(fresh["img_metas"])
        for metadata in metas:
            bus = metadata["can_bus"].copy()
            bus[:3] = 0
            bus[-1] = 0
            metadata["can_bus"] = bus
        fresh["img_metas"] = metas
        current["bev"],current["pos"] = original_bevs(**fresh)
        return history

    def planning(*args,**kwargs):
        bound = inspect.signature(original_forward).bind(*args,**kwargs)
        arguments = dict(bound.arguments)
        if PLANNING_BEV == "current":
            arguments.update(bev_embed=current["bev"],bev_pos=current["pos"])
        current["planning_calls"] = current.get("planning_calls",0)+1
        return original_forward(**arguments)

    core.get_bevs = get_bevs
    head.forward = planning
    old = ("_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n"
           "            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']")
    marker = "print('Input shape:',tuple(_data['img'].shape),flush=True)"
    if TAIL.count(old) != 1 or TAIL.count(marker) != 1:
        raise RuntimeError("UniAD input template changed")
    tail = TAIL.replace(old,"_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']")
    tail = tail.replace("virtual_rear_views=True","virtual_rear_views=False")
    tail = tail.replace(marker,"_prepare(_data)\n"+marker)
    tail = tail.replace('_cameras,_norm,_Queue())', '_cameras,_norm,_Queue(),resize_mode='+repr(effective_profile['resize_mode'])+')')
    tail = tail.replace("(_profile['width'], _profile['height']))", "(_profile['width'], _profile['height']), "+repr(effective_profile['resize_mode'])+")")
    code = compile(tail,"closed172_three_camera","exec")

    def array(value):
        if hasattr(value,"tensor"):
            value = value.tensor
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    from edit_trust import diagnostic_closed_loop_enabled
    atomic_json(JOB/"ready.json",dict(model_profile=MODEL_PROFILE,diagnostic_closed_loop=diagnostic_closed_loop_enabled(),effective_model_input=dict(effective_profile,planning_bev=PLANNING_BEV),status="ready",uniad_model_loads=1,
                box_contract=box_contract,
                cameras=3,thresholds=[.4,.35],original_collision_postprocessor=False,
                planning_bev=PLANNING_BEV))
    mark('ready',torch_file=torch.__file__,numpy_file=np.__file__,optional_ui_modules_loaded=[name for name in ('open3d','dash','plotly','tools.closeloop.visualizer','visualizer') if name in sys.modules])
    index = 0
    previous_time = None
    seeded_scene = None
    try:
        while not (JOB/"stop.json").exists():
            request = JOB/f"request_{index:03d}.json"
            if not request.exists():
                time.sleep(.1)
                continue
            output = JOB/f"step_{index:03d}"
            output.mkdir(exist_ok=True)
            try:
                if index == 0:mark("first_input_started")
                req = _hugsim_expand(json.loads(request.read_text()))
                with Path(req["packet"]).open("rb") as f:
                    obs,info = pickle.load(f)
                from edit_trust_adapter import assert_edit_evaluable
                assert_edit_evaluable(info)
                if "diagnostic_seed" in req:
                    if not diagnostic_closed_loop_enabled():
                        raise ValueError("Diagnostic seeds require the scoped diagnostic runner")
                    seed = req["diagnostic_seed"]
                    scene = str(info["scene_token"])
                    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 or scene == seeded_scene:
                        raise ValueError("Seed may be reset only once at a new diagnostic scene/arm")
                    import random
                    random.seed(seed); np.random.seed(seed)
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    seeded_scene = scene
                timestamp = float(info["timestamp"])
                from run_reliable_sequence import prepare_temporal_frame, inspect_packet, SLOTS
                _, _, names, _, _, _, _ = inspect_packet(Path(req["packet"]), SLOTS, resize_mode=effective_profile["resize_mode"])
                if names != ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]:
                    raise ValueError("Camera rig differs from loaded model profile")
                scene_token = str(info.get("scene_token", "closed_loop_"+JOB.name))
                prepare_temporal_frame(core, scene_token, timestamp,
                                       expected_dt=float(info.get("simulation_dt", .1)))

                def prepare(data):
                    pad_to_32(data)
                    rotation = np.asarray(info["explicit_l2g_r_mat"],dtype=float)
                    translation = np.asarray(info["explicit_l2g_t"],dtype=float)
                    data["l2g_r_mat"] = data["l2g_r_mat"].new_tensor(rotation.T.copy())
                    data["l2g_t"] = data["l2g_t"].new_tensor(translation)
                    metadata = data["img_metas"][0][0]
                    metadata["scene_token"] = scene_token
                    bus = metadata["can_bus"]
                    bus[:3] = [translation[1],-translation[0],0.]
                    yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))%(2*np.pi)
                    bus[-2],bus[-1] = yaw,np.degrees(yaw)
                    from input_geometry import model_input_audit
                    atomic_json(output/'input_geometry.json', model_input_audit(data))

                from pseudo_lidar_adapter import bind_frame
                bind_frame(core, obs, info)
                core.prev_frame_info["prev_bev"] = core.prev_bev
                atomic_json(output/'temporal_state.json', dict(scene_token=scene_token,timestamp=timestamp,
                    history_bev_present=core.prev_bev is not None,
                    history_bev_shape=list(core.prev_bev.shape) if core.prev_bev is not None else None,
                    metadata_cache_is_tracking_cache=core.prev_frame_info['prev_bev'] is core.prev_bev,
                    perception_uses_history=True,planning_bev=PLANNING_BEV))
                os.environ["UNIAD_PACKET"] = req["packet"]
                os.environ["UNIAD_SMOKE_OUTPUT"] = str(output)
                scope = dict(namespace,model=model,_prepare=prepare)
                current["planning_calls"] = 0
                stream = io.StringIO()
                try:
                    with redirect_stdout(stream),torch.no_grad():
                        exec(code,scope)
                finally:
                    (output/"inference.log").write_text(stream.getvalue())
                if current["planning_calls"] != 1:
                    raise RuntimeError("Unexpected number of planning calls")
                result = scope["_results"][0]
                occupancy = result.get("occ",{}).get("seg_out")
                if occupancy is None:
                    raise RuntimeError("Missing predicted occupancy")
                plan = array(result["planning"]["result_planning"]["sdc_traj"]).reshape(6,2)
                if not np.isfinite(plan).all():
                    raise RuntimeError("Nonfinite neural plan")
                from runtime_configuration import model_input_profile
                saved = dict(trajectory=plan,occupancy=array(occupancy),
                             box_convention=np.asarray(model_input_profile()['box_convention']))
                reliability = result.get("scene_reliability")
                if reliability is None:
                    raise RuntimeError("Missing shared scene reliability output")
                atomic_json(output/"scene_reliability.json", reliability)
                for name,key in [("boxes","boxes_3d"),("scores","scores_3d"),
                                 ("labels","labels_3d"),("candidate_boxes","boxes_3d_det"),
                                 ("candidate_scores","scores_3d_det"),
                                 ("candidate_labels","labels_3d_det")]:
                    if result.get(key) is not None:
                        saved[name] = array(result[key])
                for name,key in (("boxes_gravity","boxes_3d_gravity"),("candidate_boxes_gravity","boxes_3d_det_gravity")):
                    if result.get(key) is not None: saved[name] = array(result[key])
                saved["decoded_box_origin"] = np.asarray(result.get("decoded_box_origin", "legacy_unspecified"))
                if "geometry_candidate_boxes" in result:
                    saved["geometry_candidate_boxes"] = result["geometry_candidate_boxes"]
                    atomic_json(output/"geometry_refinement.json", result["geometry_refinement"])
                target = output/"perception.npz"
                np.savez_compressed(target,**saved)
                atomic_json(JOB/f"done_{index:03d}.json",dict(status="ok",index=index,
                            timestamp=timestamp,perception=str(target),
                            reliability=str(output/"scene_reliability.json"),
                            inference_seconds=float(scope["_elapsed"])))
                if index == 0:mark("first_inference_completed")
                previous_time = timestamp
                core.geometry_candidate_refiner = None
                del result,saved,scope,obs
                current.clear()
                index += 1
            except BaseException as exc:
                traceback.print_exc()
                atomic_json(JOB/f"done_{index:03d}.json",dict(status="error",error=repr(exc)))
                raise
    finally:
        core.get_bevs = original_bevs
        head.forward = original_forward


def main():
    global JOB, PLANNING_BEV, MODEL_PROFILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    parser.add_argument("--planning-bev", choices=("history", "current"), default="history")
    parser.add_argument("--model-profile",choices=("legacy","calibration_geometry_v1"),default="legacy")
    parser.add_argument("--diagnostic-closed-loop",action="store_true")
    args = parser.parse_args()
    JOB = Path(args.job).resolve()
    configure(JOB/"startup.json")
    mark("arguments_parsed")
    PLANNING_BEV = args.planning_bev
    MODEL_PROFILE = args.model_profile
    base = Path(os.environ["BASE"])
    sys.path.insert(0,str(base/"waymo23-native-import"))
    if "BEV_EXPERIMENT_SEED" in os.environ:
        import random
        mark("torch_import_started")
        import torch
        mark("torch_imported")
        seed = int(os.environ["BEV_EXPERIMENT_SEED"])
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    import five_uniad
    os.environ["FIVE_JOB"] = str(JOB/"step_000")
    five_uniad.infer_five = serve
    try:
        from contextlib import nullcontext
        from edit_trust_adapter import diagnostic_closed_loop
        with diagnostic_closed_loop() if args.diagnostic_closed_loop else nullcontext():
            five_uniad.infer()
    except BaseException as exc:
        atomic_json(JOB/"worker_error.json",dict(status="error",error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
