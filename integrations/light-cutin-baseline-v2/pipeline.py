"""Separately preregistered original-scene A validation; frozen v1 tracker reused."""

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

import argparse,ast,copy,datetime,importlib.util,sys
pass  # Path is provided by the portable boundary above
import numpy as np

LOCAL=Path(__file__).resolve().parent
OLD=LOCAL.parent/'light-cutin-diagnostic'
sys.path.insert(1,str(OLD))
from common import *
spec=importlib.util.spec_from_file_location('original_light_pipeline',OLD/'pipeline.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
original_check=base.check
PREVIOUS=DEFAULT
OUTPUT=P26/'light_baseline551_v2'

def own_pins():
    return {str(p):sha(p) for p in sorted(LOCAL.iterdir()) if p.suffix in ('.py','.sh')}

def check(output):
    root,plan=original_check(output)
    if plan['light_baseline_v2_code']!=own_pins():raise ValueError('A-v2 implementation changed after preregistration')
    verify(plan['initial_context_asset']);verify(plan['baseline_input_audit'])
    context=read(plan['initial_context_asset']['path'])
    if context!=plan['initial_context']:raise ValueError('Initial context differs from registered asset')
    if plan['B_admitted'] or plan['patch_initially'] is not None:raise ValueError('A-v2 does not admit an intervention or patch')
    if read(root/'baseline_freeze.json')!=read(PREVIOUS/'baseline_freeze.json'):raise ValueError('Original fixed tracker changed')
    return root,plan

base.check=check

def prepare(output):
    root=Path(output).resolve()
    if root.exists():raise FileExistsError('Reuse existing v2 directory; do not prepare twice: '+str(root))
    _,oldplan=original_check(PREVIOUS)
    configure()
    from scenario551 import polygon
    from environment import native_kernels
    from runtime import TERMINATION
    report=read(PREVIOUS/'A/run_report.json')
    if report['status']!='completed' or report['controls_completed']!=70:raise ValueError('Expected completed original A evidence')
    with (PREVIOUS/'A/uniad_worker/step_000/sensor_metadata.pkl').open('rb') as f:
        import pickle
        info=pickle.load(f)
    geometry=read(PREVIOUS/'A/uniad_worker/step_000/input_geometry.json')
    rawplan=np.load(PREVIOUS/'A/uniad_worker/step_000/planning_traj.npy').reshape(6,2)
    with np.load(PREVIOUS/'A/uniad_worker/step_000/perception.npz') as z:np.testing.assert_allclose(rawplan,z['trajectory'],rtol=0,atol=1e-6)
    assert info['ego_velo']==geometry['can_bus'][13]==16.
    assert info['simulation_dt']==.1 and info['timestamp']==geometry['timestamp']==0.
    nodes=report['trace'][0]['control']['reference_states']
    audit=dict(speed_mps_reaches_model_unchanged=True,input_speed_mps=info['ego_velo'],model_can_bus_speed_mps=geometry['can_bus'][13],
        actual_dt_s=.1,reference_dt_s=.5,raw_prediction_equals_selected_raw=True,
        output_is_cumulative_xy=True,cumsum_source=asset(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py'),
        source_adapter=asset(ROOT/'code/UniAD_SIM/tools/closeloop/dataparser.py'),
        actual_adapter_trace=asset(PREVIOUS/'A/uniad_worker/step_000/input_geometry.json'),
        original_first_reference_speed_mps=nodes[1][3],
        conclusion='No speed-unit, raw-plan-copy or cumulative-output mismatch found in these checks; why model chooses low speed remains unresolved',
        does_not_certify_all_neural_calibration_or_domain_transfer=True)
    # One declared source start, chosen before any v2 GPU inference; no search for passing outcomes.
    frame=10;raw=Path(read(BUNDLE/'source_manifest.json')['raw_data'])
    pose0=np.loadtxt(raw/'ego_pose/000.txt');pose=np.linalg.inv(pose0)@np.loadtxt(raw/f'ego_pose/{frame:03d}.txt')
    np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),rtol=0,atol=1e-5)
    vehicle=read(ROOT/'waymo23-native-import/configs/vehicle.json')
    bg=np.load(BUNDLE/'background_sr_world.npy');road=np.load(BUNDLE/'road_sr_world.npy')
    with np.load(BUNDLE/'point_semantics/background_semantics.npz') as z:labels=z['class_id']
    ground=np.concatenate([road,bg[labels==0]])[:,:2];xy=pose[:2,3];local=ground[np.linalg.norm(ground-xy,axis=1)<4]
    tree=ast.parse(SCORER.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScoreCalculator')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_calculate_drivable_area_compliance')
    scope={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(SCORER),'exec'),scope)
    yaw=float(np.arctan2(pose[1,0],pose[0,0]));dac=scope[method.name](None,local,np.array([[*xy,yaw]]),2*vehicle['half_width_m'],2*vehicle['half_length_m'])
    if dac!=1.:raise ValueError('Declared source start lacks native known-ground support; do not silently search another start')
    with np.load(PREVIOUS/'A/actor_tracks.npz') as z:tracks={k:z[k].copy() for k in z.files}
    from environment import Evaluation
    env=Evaluation.__new__(Evaluation);env.tracks=tracks
    boxes=env.boxes(frame);ego=[*pose[:3,3],2*vehicle['half_width_m'],2*vehicle['half_length_m'],vehicle['body_height_m'],yaw]
    kernel=native_kernels()['fg_collision_det'];overlaps=[b for b in boxes if kernel(ego,[b['box']])]
    if overlaps:raise ValueError('Declared initial state overlaps a native actor')
    context=dict(source_frame=frame,source_timestamp_s=frame/10.,local_timestamp_start_s=0.,ego_to_world=pose.tolist(),
        initial_speed_mps=4.,initial_steering_rad=0.,native_known_ground_current_footprint_dac=dac,
        native_foreground_overlap=False,initial_pose_source=asset(raw/f'ego_pose/{frame:03d}.txt'),
        source_origin=asset(raw/'ego_pose/000.txt'),
        rationale='4m/s initialization is close to observed v1 reference; source frame10 has actual ground support',
        original_recorded_speed_not_preserved=True,initial_pose_set_once_only=True,
        history='fresh complete UniAD history from this declared episode start; no intervention fork or model reset during run')
    root.mkdir(parents=True);write(root/'initial_context.json',context);write(root/'baseline_input_audit.json',audit)
    for name in ['baseline_freeze.json','baseline_verification.json','scene_manifest.json']:(root/name).write_bytes((PREVIOUS/name).read_bytes())
    scenario=dict(config=dict(scene=551,initial_ego_speed_mps=4.,intervention_start_s=1.2),
        original_native_actors=True,intervention_admitted=False,target_track=None,
        reason='A only; old 16m/s cut-in script is incompatible and intentionally not copied')
    write(root/'scenario/scenario.json',scenario)
    plan=copy.deepcopy(oldplan)
    for key in ['pre_gpu_revision','startup_repair']:plan.pop(key,None)
    plan.update(schema='fixed_light_original_A_v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code=code_pins(),runtime_code=base.runtime_code_pins(),light_baseline_v2_code=own_pins(),
        common_scene_token='light551_original_A_v2',controls=70,termination=TERMINATION,
        initial_context=context,initial_context_asset=asset(root/'initial_context.json'),baseline_input_audit=asset(root/'baseline_input_audit.json'),
        B_admitted=False,patch_initially=None,gpu_executed=False,previous_A=asset(PREVIOUS/'A/run_report.json'),
        previous_plan=asset(PREVIOUS/'plan.json'),
        preregistration='Only initial speed, initial source state and source-clock origin change; tracker and criteria identical. No v2 outcomes used.',
        source_clock_contract='render, actor collision, actor scoring: frame10 + local step; model timestamps: local step*0.1; actual global ego pose remains explicit')
    plan['assets'].pop('target_track',None)
    plan['assets'].update(scene=asset(root/'scene_manifest.json'),scenario=asset(root/'scenario/scenario.json'),baseline=asset(root/'baseline_freeze.json'))
    write(root/'plan.json',plan)
    write(root/'status.json',dict(stage='A_v2_preregistered_GPU_pending',B_admitted=False,controller_changed=False,
        initial_speed_mps=4.,initial_source_frame=10,old_A_preserved=True))
    return dict(prepared=True,output=str(root),initial_context=context,tracker_reused_unchanged=True,B_admitted=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','check','baseline','inspect','render'])
    parser.add_argument('--output',default=str(OUTPUT));args=parser.parse_args()
    if args.stage=='prepare':result=prepare(args.output)
    elif args.stage=='check':
        root,plan=check(args.output);configure()
        from native_car_edit import make_types
        make_types()
        import torch
        result=dict(verified=True,cuda_available=torch.cuda.is_available(),controller_unchanged=True,source_frame_offset=10,B_admitted=False)
    elif args.stage=='baseline':
        result=base.original_baseline(args.output)
        write(Path(args.output)/'status.json',dict(stage='A_v2_gpu_admitted',baseline_result=result,B_admitted=False,
            next='Design and preregister a compatible native-car intervention before any B experiment'))
    elif args.stage=='inspect':
        root,_=check(args.output)
        from diagnostics import arm_summary
        result=arm_summary(root/'A')
    else:
        root,_=check(args.output)
        from diagnostics import artifacts
        result=artifacts(root,'A','A')
    import json
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
