"""Reuse completed map A, freeze one native-car intervention, run and diagnose B."""

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

import argparse,copy,datetime,importlib.util,json,sys,difflib
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;V4=LOCAL.parent/'light-map551-v4';OLD=LOCAL.parent/'light-cutin-diagnostic'
sys.path[:0]=[str(LOCAL),str(V4),str(OLD)]
from common import *

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

base=module('cutin_v5_original_pipeline',OLD/'pipeline.py');original_check=base.check
v4=module('cutin_v5_frozen_map_pipeline',V4/'pipeline.py')
# The reused v4 pipeline inserts OLD at index1 during import. Restore the explicit
# release precedence so its obsolete scoring exporter cannot shadow the v4 fix.
sys.path[:0]=[str(LOCAL),str(V4),str(OLD)]
OUTPUT=P26/'light_intent_recovery_v12_r2';BASELINE=P26/'light_map551_v4'

def pins():return {str(p):sha(p) for p in LOCAL.rglob('*.py')}|{str(LOCAL/'run.sh'):sha(LOCAL/'run.sh')}

def bindings():
    # Resolve the new harness before configure() adds renderer snapshot paths.
    import runtime,scoring,local_boot,tracker
    expected={runtime:LOCAL/'runtime.py',scoring:V4/'scoring.py',local_boot:V4/'local_boot.py',tracker:OLD/'tracker.py'}
    configure()
    import joint_safety,reference_adapter,risk_memory,observation_evidence
    expected[observation_evidence]=LOCAL/'observation_evidence.py'
    expected[risk_memory]=LOCAL/'risk_memory.py'
    expected[reference_adapter]=LOCAL/'reference_adapter.py'
    expected[joint_safety]=LOCAL/'joint_safety.py'
    import map_reference
    expected[map_reference]=V4/'map_reference.py'
    for m,path in expected.items():
        if Path(m.__file__).resolve()!=path:raise ValueError('Unexpected implementation binding: '+m.__name__)
    return {m.__name__:str(path) for m,path in expected.items()}

def baseline_check(path):
    root,plan=v4.check(path)
    admission=read(root/'full_episode_admission.json')
    if not admission['passed']:raise ValueError('A was not admitted; do not run B')
    verify(admission['summary']);verify(admission['map'])
    report=read(root/'A/run_report.json')
    if report.get('experiment_plan')!=asset(root/'plan.json') or report.get('model_weights_unchanged') is not True:
        raise ValueError('A model/plan identity unavailable')
    if report['status']!='completed' or len(report['trace'])!=188:raise ValueError('A is incomplete')
    return root,plan

def check(output):
    root,plan=original_check(output)
    if plan['allowed_online_inputs']!=['static road polygon','current frozen UniAD plan','actual ego kinematics','current retained UniAD boxes/scores/labels','pose-compensated history of retained detections']:raise ValueError('Online input contract changed')
    if plan['intent_recovery_v12_code']!=pins():raise ValueError('Frozen v11 implementation changed')
    if plan['map_v4_code']!=v4.pins():raise ValueError('Frozen A map/controller sources changed')
    baseline,old=baseline_check(plan['baseline_root'])
    for entry in plan['baseline_receipts'].values():verify(entry)
    for k in ['static_map','boot_manifest','boot_archive','initial_context_asset']:verify(plan[k])
    for k in ['seed','common_scene_token','controls','source_end_frame','initial_context','termination','forbidden_online_inputs']:
        if plan[k]!=old[k]:raise ValueError('One-factor pair contract differs: '+k)
    for name in ['scene_manifest.json','baseline_freeze.json','map/map.json','boot/manifest.json','initial_context.json']:
        if sha(root/name)!=sha(baseline/name):raise ValueError('Paired frozen input differs: '+name)
    verify(plan['collision_off_baseline_report'])
    verify(plan['native_cpu_admission'])
    if plan['native_collision_policy']!=dict(enabled_after_common_index=-1,implementation='unchanged UniAD collision_optimization',external_obstacle_mpc=False):raise ValueError('Native collision policy changed')
    admission=read(root/'checks/cpu_admission.json')
    if admission['code_sha256']!=pins():raise ValueError('CPU admission does not match frozen release')
    if not admission['passed'] or not read(root/'checks/native_cpu_replay/result.json')['ipopt_solve_passed']:raise ValueError('Native collision CPU admission missing')
    scenario=read(root/'scenario/scenario.json')
    if not scenario['passed'] or scenario['config']!=__import__('cutin_scenario').CONFIG:raise ValueError('Unverified cut-in script')
    return root,plan

base.check=check

def prepare(output):
    root=Path(output).resolve()
    if (root/'plan.json').exists():
        check(root);return dict(reused=True,output=str(root),gpu_executed=False)
    bindings()
    baseline,old=baseline_check(BASELINE)
    print('Verifying existing completed A pixels, output receipts and frozen model; direct new-policy B will run; no new A...',flush=True)
    if not v4.base.completed_arm(baseline,'A'):raise ValueError('Missing valid A')
    root.mkdir(parents=True,exist_ok=True)
    for name in ['scene_manifest.json','baseline_freeze.json','baseline_verification.json','initial_context.json','baseline_input_audit.json','map/map.json','boot/manifest.json']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((baseline/name).read_bytes())
    from cutin_scenario import generate
    scenario=generate(root/'scenario',baseline)
    plan=copy.deepcopy(old)
    plan.update(schema='joint_intent_recovery_v12',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        intent_recovery_v12_code=pins(),baseline_root=str(baseline),
        native_cpu_admission=asset(root/'checks/cpu_admission.json'),
        baseline_receipts={n:asset(baseline/n) for n in ['plan.json','full_episode_admission.json','A/run_report.json','A/summary.json','A/actor_tracks.npz']},
        B_admitted=True,intervention_only='native:RigidNodes:3 pose trajectory; all other actors, scene, weather, assets, model, tracker, map and budgets unchanged',
        patch_initially=None,gpu_executed=False,
        collision_off_baseline_root=str(P26/'light_map551_cutin_v5'),
        collision_off_baseline_report=asset(P26/'light_map551_cutin_v5/B/run_report.json'),
        native_collision_policy=dict(enabled_after_common_index=-1,implementation='unchanged UniAD collision_optimization',external_obstacle_mpc=False),
        prefix_policy='Direct B per user request: fresh inference and control from observation0, full history, no archived A actions; old B is outcome comparison only',
        local_intervention_start_s=.5,source_intervention_start_s=1.5,
        B_success_not_preregistered_as_mandatory_avoidance=True,
        B_outcome_policy='Normal deceleration/yield may be correct. Use same fixed collision, map, stall, recovery, simulation and wall-clock limits as A. Insufficient response/recovery window remains unresolved; do not change criteria after run.',
        baseline_reused_without_reexecution=False,new_A_required=False,user_requested_direct_B=True,
        allowed_online_inputs=old['allowed_online_inputs']+['current retained UniAD boxes/scores/labels','pose-compensated history of retained detections'],
        joint_safety_policy=__import__('joint_safety').POLICY)
    plan['assets'].update(scene=asset(root/'scene_manifest.json'),scenario=asset(root/'scenario/scenario.json'),
        baseline=asset(root/'baseline_freeze.json'),target_track=asset(root/'scenario/target_track.npz'))
    write(root/'plan.json',plan)
    patch=''.join(difflib.unified_diff((V4/'runtime.py').read_text().splitlines(True),(LOCAL/'runtime.py').read_text().splitlines(True),fromfile='frozen_v4/runtime.py',tofile='B_v5/runtime.py'))
    (root/'runtime_changes_from_A.diff').write_text(patch)
    write(root/'status.json',dict(stage='B_prepared_cpu_audited',A_reused=False,patch=None,gpu_executed=False,
        next='check then full on idle CUDA node; inspect/render reuse completed B'))
    return dict(prepared=True,output=str(root),A_reused=False,scenario_passed=scenario['passed'],
        actor_metrics=scenario['metrics'],minimum_other_actor_gap_m=scenario['minimum_other_actor_gap_m'],gpu_executed=False)

def full(output):
    root,plan=check(output);bindings()
    b=base.run_and_score(root,'B',plan['controls'])
    result=dict(B=b,old_B_summary=read(P26/'light_risk_provenance_v11/B/summary.json'),
        new_A_executed=False,normal_scene_regression_validated=False,
        run_scope='User requested direct B; fresh complete history; no A action reuse',
        avoidance_success_not_assumed=True)
    write(root/'comparison_joint_safety.json',result)
    from compare_native import render_pair
    result['media']=render_pair(root,P26/'light_risk_provenance_v11/B',root/'B',{'off':read(P26/'light_risk_provenance_v11/B/run_report.json'),'on':read(root/'B/run_report.json')})
    write(root/'comparison_joint_safety.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','check','full','inspect','render']);parser.add_argument('--output',default=str(OUTPUT));args=parser.parse_args()
    if args.stage=='prepare':out=prepare(args.output)
    elif args.stage=='check':
        root,plan=check(args.output);resolved=bindings()
        import torch
        out=dict(verified=True,cuda_available=torch.cuda.is_available(),A_reused=False,controls=plan['controls'],tracker_solver_and_map_geometry_unchanged=True,dense_reference_adapter=True,
            source_frame_offset=10,local_intervention_start_s=.5,absolute_source_activation_s=1.5,patch=None,implementation_bindings=resolved)
        write(root/'checks/preflight.json',out)
    elif args.stage=='full':out=full(args.output)
    else:
        root,plan=check(args.output)
        if not base.completed_arm(root,'B'):raise ValueError('B not executed; only CPU preparation exists')
        out=read(root/'comparison_joint_safety.json')
    print(json.dumps(out,indent=2,allow_nan=False,ensure_ascii=False))

if __name__=='__main__':main()
