"""Freeze external road policy, reuse A-v2 off, validate only a new road-policy A."""

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

import argparse,ast,copy,importlib.util,json,pickle,sys,datetime
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parent;OLD=LOCAL.parent/'light-cutin-diagnostic'
sys.path.insert(1,str(OLD))
from common import *
spec=importlib.util.spec_from_file_location('road_original_pipeline',OLD/'pipeline.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);original_check=base.check
PREVIOUS=P26/'light_baseline551_v2';OUTPUT=P26/'light_road551_v3'
ROAD_ADMISSION=dict(max_lateral_drift_m=.5,max_observed_edge_heading_error_deg=3.,
    all_executed_stopping_certificates_pass=True,all_current_body_edge_checks_pass=True,
    same_initial_pixels=True,same_initial_raw_prediction_atol=1e-4,
    independent_white_line_ground_truth_available=False)

def own_pins():return {str(p):sha(p) for p in sorted(LOCAL.iterdir()) if p.suffix in ('.py','.sh')}

def check(output):
    root,plan=original_check(output)
    if plan['road_v3_code']!=own_pins():raise ValueError('Frozen road policy/runtime changed')
    if plan['road_admission']!=ROAD_ADMISSION:raise ValueError('Road admission rules changed')
    for key in ['initial_context_asset','A_off','A_off_summary','previous_plan']:verify(plan[key])
    if read(root/'baseline_freeze.json')!=read(PREVIOUS/'baseline_freeze.json'):raise ValueError('Fixed tracking baseline changed')
    return root,plan
base.check=check

def audit_off(folder):
    """Current-image diagnosis of recorded off; never feeds archived truth online."""
    configure()
    from road_reference import RoadReference,detect,WhiteEdge
    from runtime_configuration import load_vehicle_config
    from PIL import Image,ImageDraw
    from white_edge import geometry,project
    v=load_vehicle_config();a=read(PREVIOUS/'A/run_report.json');folder=Path(folder);rows=[]
    for o in a['observations']:
        i=o['index'];step=PREVIOUS/'A'/f'uniad_worker/step_{i:03d}'
        with (step/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
        obs={'rgb':{c:np.asarray(Image.open(PREVIOUS/'A'/f'steps/{i:03d}/{c}.png')) for c in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']}}
        packet=detect(obs,info);edge=WhiteEdge(packet,v);mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'])
        state=np.array([[-mount[0,3],-mount[1,3],0.,o['motion_state'][3],o['motion_state'][4]]])
        gap=float(edge.clearance(state)[0]);right=next((e for e in packet['edges'] if e['side']=='right'),None)
        heading=None;geometric_gap=None
        if right:
            direction=np.asarray(info['explicit_l2g_r_mat'])@np.array([right['a'],1.,0.])
            heading=float(np.degrees(np.arctan2(-direction[0],direction[1])))
            corners=state[0,:2]+np.array([[-v.half_width_m,-v.half_length_m],[-v.half_width_m,v.half_length_m],[v.half_width_m,-v.half_length_m],[v.half_width_m,v.half_length_m]])
            geometric_gap=float(np.min((right['a']*corners[:,1]+right['b']-corners[:,0])/np.hypot(1,right['a'])))
        row=dict(index=i,timestamp_s=o['timestamp'],observed_edge=packet,
            body_clearance_with_margin_m=gap if np.isfinite(gap) else None,
            geometric_white_line_clearance_without_margin_m=geometric_gap,
            world_road_heading_deg=heading,right_drift_m=o['motion_state'][0])
        rows.append(row)
        if i in (0,20,40,60,70):
            image=Image.fromarray(obs['rgb']['CAM_FRONT']);draw=ImageDraw.Draw(image)
            K,M,z=geometry(info,'CAM_FRONT',obs['rgb']['CAM_FRONT'].shape)
            for e in packet['edges']:
                ys=np.linspace(*e['observed_y_range'],80);points=np.c_[e['a']*ys+e['b'],ys,np.full(len(ys),z)]
                uv,depth=project(points,K,M);valid=(depth>0)&np.isfinite(uv).all(1)
                draw.line([tuple(map(float,p)) for p in uv[valid]],fill='red',width=5)
            draw.text((20,20),f'A-v2 OFF t={i*.1:.1f}s; estimated body clearance={gap:.3f}m',fill='red',stroke_width=1)
            folder.mkdir(parents=True,exist_ok=True);image.save(folder/f'edge_{i:03d}.png')
        if i%10==0:print('Recorded A-off white-edge diagnosis',i,'/',len(a['observations'])-1,flush=True)
    bad=[r for r in rows if r['body_clearance_with_margin_m'] is not None and r['body_clearance_with_margin_m']<0]
    crossed=[r for r in rows if r['geometric_white_line_clearance_without_margin_m'] is not None and r['geometric_white_line_clearance_without_margin_m']<0]
    headings=[r['world_road_heading_deg'] for r in rows if r['world_road_heading_deg'] is not None]
    report=dict(run=asset(PREVIOUS/'A/run_report.json'),detector=asset(CR/'phase46/framework_driver_v1/white_edge.py'),
        rows=rows,first_margin_violation_s=bad[0]['timestamp_s'] if bad else None,
        first_estimated_body_white_line_crossing_s=crossed[0]['timestamp_s'] if crossed else None,
        min_current_body_clearance_m=min(r['body_clearance_with_margin_m'] for r in rows if r['body_clearance_with_margin_m'] is not None),
        world_observed_road_heading_range_deg=[min(headings),max(headings)] if headings else None,
        physical_boundary_ground_truth=False,source='actual current images at each recorded state',
        mechanism='Raw ego-local plans retain a right bias while observed road tangent requires increasing left correction; no road-heading or white-edge term enters v2 reference/control',
        straight_command_label_source=asset(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/datasets/data_utils/trajectory_api.py'),
        straight_command_semantics='Training labels allow up to 2m terminal lateral displacement in FORWARD; this is not a lane-centering constraint',
        deeper_neural_cause='Domain transfer/calibration/temporal sensitivity not fully separated; no neural-weight fix is claimed',
        no_GPU_rerun=True)
    write(folder/'report.json',report);return report

def prepare(output):
    root=Path(output).resolve()
    if root.exists():raise FileExistsError('Existing prepared output retained; do not prepare twice')
    _,old=original_check(PREVIOUS)
    if not read(PREVIOUS/'baseline_gpu_admission.json')['passed']:raise ValueError('A-v2 not admitted')
    from road_reference import POLICY
    root.mkdir(parents=True)
    for name in ['baseline_freeze.json','baseline_verification.json','scene_manifest.json','initial_context.json','baseline_input_audit.json']:
        (root/name).write_bytes((PREVIOUS/name).read_bytes())
    write(root/'scenario/scenario.json',read(PREVIOUS/'scenario/scenario.json'))
    audit_off(root/'off_diagnosis')
    plan=copy.deepcopy(old);plan.pop('light_baseline_v2_code',None)
    plan.update(schema='light_observed_road_external_policy_v3',road_v3_code=own_pins(),
        code=code_pins(),runtime_code=base.runtime_code_pins(),road_policy=POLICY,road_admission=ROAD_ADMISSION,
        previous_plan=asset(PREVIOUS/'plan.json'),A_off=asset(PREVIOUS/'A/run_report.json'),A_off_summary=asset(PREVIOUS/'A/summary.json'),
        initial_context_asset=asset(root/'initial_context.json'),baseline_input_audit=asset(root/'baseline_input_audit.json'),
        preregistration='Same initial state, original actors, seed, frozen model/tracker and budget as A-v2. Only external road-reference and declared safety gate change.',
        B_admitted=False,patch_initially=None,gpu_executed=False,
        external_road_policy_enabled=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    plan['assets'].update(scene=asset(root/'scene_manifest.json'),scenario=asset(root/'scenario/scenario.json'),baseline=asset(root/'baseline_freeze.json'))
    write(root/'plan.json',plan);write(root/'status.json',dict(stage='road_policy_frozen_GPU_pending',A_off_reused=True,B_admitted=False))
    return dict(prepared=True,output=str(root),fixed_tracker_unchanged=True,A_off_reused=True,B_admitted=False)

def results(root):
    root=Path(root);a=read(root/'A/run_report.json');previous=read(PREVIOUS/'A/run_report.json');rows=[]
    for o in a['observations']:
        reference_file=root/'A'/f'steps/{o["index"]:03d}/reference.json'
        if not reference_file.exists():continue
        selection=read(reference_file)['selection'];packet=selection['observed_white_edge']
        from road_reference import WhiteEdge
        from runtime_configuration import load_vehicle_config
        with (root/'A'/f'uniad_worker/step_{o["index"]:03d}/sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
        mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'])
        gap=float(WhiteEdge(packet,load_vehicle_config()).clearance(np.array([[-mount[0,3],-mount[1,3],0.,0.,0.]]))[0])
        primary=next((e for e in packet['edges'] if e['side']==selection['primary_side']),None)
        rows.append(dict(index=o['index'],current_body_clearance_m=gap if np.isfinite(gap) else None,
            heading_to_road_deg=float(np.degrees(np.arctan(primary['a']))) if primary else None))
    images_same=a['observations'][0]['image_sha256']==previous['observations'][0]['image_sha256']
    raw_same=bool(a['trace'] and np.allclose(a['trace'][0]['raw_uniad'],previous['trace'][0]['raw_uniad'],rtol=0,atol=1e-4))
    ordinary=read(root/'baseline_gpu_admission.json')['passed'] if (root/'baseline_gpu_admission.json').exists() else False
    max_drift=max(abs(o['motion_state'][0]) for o in a['observations'])
    angles=[abs(r['heading_to_road_deg']) for r in rows if r['heading_to_road_deg'] is not None]
    checks=dict(original_baseline_rules=ordinary,complete_70_controls=len(a['trace'])==70,
        all_action_certificates=all(t['road_gate']['certificate']['passed'] for t in a['trace']) and bool(a['trace']),
        all_current_body_edges=len(rows)==len(a['observations']) and all(r['current_body_clearance_m'] is not None and r['current_body_clearance_m']>=0 for r in rows),
        lateral_drift=max_drift<=ROAD_ADMISSION['max_lateral_drift_m'],
        observed_road_heading=len(angles)==len(a['observations']) and max(angles)<=ROAD_ADMISSION['max_observed_edge_heading_error_deg'],
        same_initial_pixels=images_same,same_initial_raw_plan=raw_same)
    result=dict(passed=all(checks.values()),checks=checks,rows=rows,max_lateral_drift_m=max_drift,
        A_off=asset(PREVIOUS/'A/summary.json'),A_on=asset(root/'A/summary.json'),
        no_physical_ground_truth_boundary_certificate=True,
        conclusion='external_road_policy_effective_on_A_development_window' if all(checks.values()) else 'unresolved_or_failed_keep_disabled',
        causal_scope='Matched initial state/raw input; later raw outputs can differ because ego execution/viewpoints diverge',
        original_frozen_tracker_unchanged=True,adopted_as_default=False,B_admitted=False)
    write(root/'road_admission.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','check','baseline','inspect','render']);p.add_argument('--output',default=str(OUTPUT));args=p.parse_args()
    if args.stage=='prepare':out=prepare(args.output)
    elif args.stage=='check':
        root,plan=check(args.output);configure()
        import torch
        from native_car_edit import make_types
        make_types()
        from road_reference import detect
        import cv2
        out=dict(verified=True,cuda_available=torch.cuda.is_available(),opencv_available=True,road_policy_frozen=True)
    elif args.stage=='baseline':
        root,plan=check(args.output)
        try:base.original_baseline(root)
        finally:
            if (root/'A/summary.json').exists():
                out=results(root);write(root/'status.json',dict(stage='road_A_completed',road_admission=out))
        if not out['passed']:raise RuntimeError('Road-policy A did not meet frozen criteria; inspect road_admission.json')
    elif args.stage=='inspect':
        root,_=check(args.output);configure();out=results(root)
    else:
        root,_=check(args.output);configure()
        from diagnostics import artifacts
        out=artifacts(root,'A','A')
    print(json.dumps(out,indent=2,allow_nan=False))

if __name__=='__main__':main()
