"""Read-only low-speed provenance and an isolated CPU CAN-bus injection test."""

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

import ast,datetime,json,sys
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
sys.path.insert(0,str(LOCAL.parent/'light-cutin-diagnostic'))
from common import P26,BUNDLE,asset,read,write
OUTPUT=P26/'uniad_speed_audit_v1';A=P26/'light_map551_v4'
CONFIG=ROOT/'code/UniAD_SIM/projects/configs/stage2_e2e/base_e2e.py'
TRANSFORMER=ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/modules/transformer.py'
CHECKPOINT=ROOT/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'

def flags(text,name):
    return [dict(line=n.lineno,value=ast.literal_eval(n.value)) for n in ast.walk(ast.parse(text))
        if isinstance(n,ast.keyword) and n.arg==name]

def bus_injection_test(state,bus):
    import torch
    from types import SimpleNamespace
    tree=ast.parse(TRANSFORMER.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PerceptionTransformer')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_bev_features');fn.decorator_list=[]
    scope=dict(torch=torch,np=np);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(TRANSFORMER),'exec'),scope)
    mlp=torch.nn.Sequential(torch.nn.Linear(18,128),torch.nn.ReLU(inplace=True),torch.nn.Linear(128,256),torch.nn.ReLU(inplace=True))
    # Match the checkpoint's actual optional normalization, without loading a full network.
    prefix='pts_bbox_head.transformer.can_bus_mlp.'
    sub={k[len(prefix):]:v.clone() for k,v in state.items() if k.startswith(prefix)}
    if 'norm.weight' in sub:mlp.add_module('norm',torch.nn.LayerNorm(256))
    mlp.load_state_dict(sub,strict=True);mlp.eval()
    captured={}
    def encoder(query,*args,**kwargs):captured['query']=query.detach().clone();return query
    owner=SimpleNamespace(use_can_bus=False,use_shift=True,rotate_prev_bev=True,can_bus_mlp=mlp,
        use_cams_embeds=False,level_embeds=torch.zeros(1,256),encoder=encoder)
    def run(enabled,speed):
        owner.use_can_bus=enabled;b=np.asarray(bus,dtype=float).copy();b[13]=speed
        with torch.no_grad():scope['get_bev_features'](owner,[torch.zeros(1,3,256,2,2)],torch.zeros(4,256),2,2,
            bev_pos=torch.zeros(1,256,2,2),prev_bev=None,img_metas=[dict(can_bus=b)])
        return captured['query']
    off4,off16,on4,on16=run(False,4),run(False,16),run(True,4),run(True,16)
    result=dict(scope='CPU: original get_bev_features method and checkpoint CAN MLP; encoder replaced by query capture, NOT full-model inference',
        checkpoint_mlp_keys=list(sub),disabled_speed4_vs16_bitwise_identical=torch.equal(off4,off16),
        disabled_max_absolute_difference=float((off4-off16).abs().max()),
        enabled_max_absolute_difference=float((on4-on16).abs().max()),
        enabled_speed_feature_changes=not torch.equal(on4,on16),full_model_gpu_executed=False)
    if not result['disabled_speed4_vs16_bitwise_identical'] or not result['enabled_speed_feature_changes']:raise ValueError('CAN bus branch proof failed')
    return result

def audit(output=OUTPUT):
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    print('Reading checkpoint metadata and only CAN-MLP tensors with mmap; no model instantiated...',flush=True)
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    text=checkpoint['meta']['config'];(output/'checkpoint_training_config.py').write_text(text)
    trained=flags(text,'use_can_bus');current=flags(CONFIG.read_text(),'use_can_bus')
    if len(trained)!=1 or len(current)!=1:raise ValueError('Ambiguous config; inspect inherited flags')
    source_ids=['light_cutin551_v1','light_baseline551_v2','light_map551_v4'];runs={}
    for name in source_ids:
        folder=P26/name/'A';report=read(folder/'run_report.json');trace=report['trace'];selected=[0,min(10,len(trace)-1),len(trace)-1];samples=[]
        for i in selected:
            t=trace[i];raw=np.asarray(t['raw_uniad']);geometry=read(folder/f'uniad_worker/step_{i:03d}/input_geometry.json')
            sample=dict(index=i,actual_speed_mps=t['motion_state'][3],input_can_bus_speed_mps=geometry['can_bus'][13],
                first_control_reference_speed_mps=t['control']['target_speed_mps'],raw_six_nodes=raw.tolist(),
                raw_between_node_speeds_mps=(np.linalg.norm(np.diff(raw,axis=0),axis=1)/.5).tolist(),
                input_geometry=asset(folder/f'uniad_worker/step_{i:03d}/input_geometry.json'))
            motion=folder/f'uniad_worker/step_{i:03d}/motion_head_raw.npz'
            if motion.exists():
                with np.load(motion) as z:sample['inferred_SDC_bbox_velocity_xy_mps']=z['motion_query_boxes'][-1,7:9].tolist()
            samples.append(sample)
        forward_unchanged=all(np.array_equal(np.asarray(t['raw_uniad'])[:,1],np.asarray(t['adopted'])[:,1]) for t in trace)
        runs[name]=dict(report=asset(folder/'run_report.json'),controls=len(trace),samples=samples,
            target_speed_min_mps=min(t['control']['target_speed_mps'] for t in trace),
            target_speed_max_mps=max(t['control']['target_speed_mps'] for t in trace),
            longitudinal_nodes_unchanged_by_external_reference=forward_unchanged,
            road_gate_overrides=sum(bool(t.get('road_gate',{}).get('override')) for t in trace))
    bus=read(A/'A/uniad_worker/step_000/input_geometry.json')['can_bus']
    proof=bus_injection_test(checkpoint['state_dict'],bus)
    vehicle=read(ROOT/'waymo23-native-import/configs/vehicle.json')
    source=read(BUNDLE/'source_manifest.json');data=Path(source['raw_data'])
    p0=np.loadtxt(data/'ego_pose/010.txt');p1=np.loadtxt(data/'ego_pose/011.txt')
    recorded_speed=float(np.linalg.norm((np.linalg.inv(p0)@p1)[:2,3])/.1)
    result=dict(schema='uniad_low_speed_read_only_diagnosis_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        confirmed_configuration_mismatch=trained[0]['value'] is True and current[0]['value'] is False,
        checkpoint_training_use_can_bus=trained,current_runtime_config_use_can_bus=current,
        input_state_feature_gate_proof=proof,runs=runs,vehicle_max_speed_mps=vehicle['max_speed_mps'],
        source_frame10_recorded_speed_mps=recorded_speed,
        planning_head_has_no_explicit_current_speed_argument=True,
        planning_bev_policy=read(A/'A/uniad_worker/ready.json')['planning_bev'],
        inference_dt_s=.1,planning_node_dt_s=.5,output_nodes_are_cumulative_positions=True,
        known_input_incompleteness=dict(acceleration_bus_slots_7_9_are_zero=True,
            angular_rate_uses_negative_steering_rate_not_body_yaw_rate=True,
            pose_quaternion_is_yaw_only_but_actual_pose_has_pitch_and_roll=True,
            restoring_trained_flag_alone_is_diagnostic_not_a_complete_input_fix=True),
        confirmed_conclusion='Actual frozen pretrained checkpoint enables CAN-bus features; loaded runtime disables them. Reported speed reaching can_bus[13] never proved that speed entered the learned encoder. Existing raw neural trajectories are already slow before tracking/map processing.',
        not_yet_proven=['Whether restoring the flag is sufficient to recover normal-speed planning','Contribution of three-camera/Waymo domain change and current-BEV policy','Contribution of 0.1s input history versus training keyframe cadence; no automatic factor5 rescaling justified'],
        next='One resident GPU model, four isolated same-image cold-start input cases: gate off/on x speed4/16. Reproduce saved off4 and prove off-speed invariance before attributing speed changes to the gate. Then correct input state contract and validate a fresh A, before B.',
        GPU_executed=False,production_code_or_weights_changed=False,
        sources=dict(runtime_config=asset(CONFIG),transformer=asset(TRANSFORMER),checkpoint=asset(CHECKPOINT),
            checkpoint_training_config=asset(output/'checkpoint_training_config.py'),
            parser=asset(ROOT/'code/UniAD_SIM/tools/closeloop/dataparser.py'),
            planning_head=asset(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py')))
    if not result['confirmed_configuration_mismatch']:raise ValueError('Checkpoint/runtime mismatch was not reproduced')
    write(output/'report.json',result)
    print(json.dumps({k:result[k] for k in ['confirmed_configuration_mismatch','checkpoint_training_use_can_bus','current_runtime_config_use_can_bus','input_state_feature_gate_proof','source_frame10_recorded_speed_mps','GPU_executed']},indent=2))
    return result

if __name__=='__main__':audit()
