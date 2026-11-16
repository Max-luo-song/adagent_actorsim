"""Frozen geometric context and measured editing feedback, with no held-out tuning data."""
import numpy as np
from common_flow import *
from scene_designer import signature,PROGRAM,PARAMETERS
from action_program import ACTIONS,LANES,phase_schema,semantics


def build(m,feedback=()):
    policy=read(m['policies']['global']);verify_pins(policy['code_pins'])
    release=Path(policy['release']);plan=read(release/'plan.json')
    bundle=P26.parent/'phase33/scene_data551_v1/runs/scene551_20261106T050218Z'
    meta=read(bundle/'source_manifest.json');config=read(release/'scenario/scenario.json')['config']
    anchor=np.asarray(plan['initial_context']['ego_to_world']);frame=int(plan['initial_context']['source_frame'])
    participants=[]
    with np.load(bundle/'actors.npz') as actors:
        for group in ('RigidNodes','DeformableNodes','SMPLNodes'):
            t=actors[group+'__instances_trans'];sizes=actors[group+'__instances_size'];valid=actors[group+'__instances_fv']
            for i in np.flatnonzero(valid[frame]):
                rel=np.linalg.inv(anchor)@np.r_[t[frame,i],1.]
                if np.linalg.norm(rel[:2])>100:continue
                velocity=(t[frame+1,i]-t[frame-1,i])/.2 if valid[frame+1,i] and valid[frame-1,i] else None
                participants.append(dict(group=group,index=int(i),relative_forward_left_up_m=rel[:3].tolist(),size_source=sizes[i].tolist(),velocity_source_mps=velocity.tolist() if velocity is not None else None))
    participants.sort(key=lambda p:np.linalg.norm(p['relative_forward_left_up_m'][:2]))
    known=[];outcomes=[]
    roots=[P26/'autonomous_edit_discovery_v1',P26/'autonomous_edit_discovery_v2',OUT]
    # Historical single-merge edits are feedback, not action-sequence duplicates.
    for root in roots:
        for file in sorted((root/'tasks').glob('*/edit_work/scenario/scenario.json')):
            audit=read(file);params={k:audit['config'][k] for k in PARAMETERS if k in audit['config']}
            if 'phases' in params:known.append(signature(params))
            else:params={k:audit['config'].get(k) for k in ('lane_change_duration_s','actor_deceleration_duration_s','target_lane_center_source_y_m')}
            item=dict(parameters=params,offline_legal=audit['passed'],failed_checks=[k for k,v in audit['checks'].items() if not v],motion=audit['metrics'])
            first=file.parent.parent.parent/'first_screen.json'
            if first.exists():item['measured_v15_finding']=read(first)
            qualification=file.parent.parent.parent/'qualification.json'
            if qualification.exists():item['qualification']={k:v for k,v in read(qualification).items() if k not in ('measurements','repeat')}
            outcomes.append(item)
    return dict(schema='scene_design_context_v1',action_semantics=semantics(),global_default=dict(version='v15',policy_sha256=sha(Path(policy['code'])/'joint_safety.py'),fixed=True),
        catalog=[dict(scene=551,editing_ready=True,program=PROGRAM,actor='native:RigidNodes:3',allowed_parameters=list(PARAMETERS),phase_schema=phase_schema(),actions=list(ACTIONS),lanes=list(LANES),speed_ratio_semantics='Derivative of retimed source trajectory clock; final physical speed is audited, not an absolute m/s command',post_sequence='Continue last lane and speed ratio; reject if source clock exceeds available track'),
                 *[dict(scene=s,editing_ready=False,reason='Cached source assets exist; matching runtime/map/scorer not admitted; do not propose this scene') for s in (172,552,621)]],
        source=dict(manifest=asset(release/'scene_manifest.json'),scene_data=asset(bundle/'source_manifest.json'),source_frames=meta['frames'],fps=10.,initial_source_frame=frame,initial_ego_speed_mps=config['initial_ego_speed_mps'],navigation='fixed straight',original_camera_count=3),
        ego_map=dict(asset=asset(release/'map/map.json'),geometry=read(release/'map/map.json'),authority='Existing user-authorized sensor-derived ego road polygon; not independently surveyed physical truth'),
        nearby_participants=participants[:32],editable_actor=dict(group='RigidNodes',index=3,activation_source_s=config['absolute_source_activation_s'],original_source=asset(bundle/'actors.npz'),insertion_allowed=False),
        constraints=dict(fps_and_initial_state_fixed=True,other_actors_fixed=True,model_and_scores_fixed=True,actor_motion_limits=config['actor_motion_limits'],minimum_other_actor_gap_m=config['minimum_other_actor_gap_m'],offline_feasible_path_required=True,patch_code_not_editable_here=True),
        known_parameter_signatures=sorted(set(known)),previous_designs_and_outcomes=outcomes[-32:],measured_host_feedback=list(feedback)[-16:],held_out_scene_data_not_provided=True,
        goal='Propose new legal, meaningfully diverse multi-stage action challenges; include changes of behavior rather than simple positional perturbations. Do not label any as a failure until actual v15 trials and independent repeat. Never require an unavoidable collision.')
