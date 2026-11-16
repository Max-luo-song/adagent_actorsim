"""Fresh registered 551 edit probes with a frozen carried controller, no old actions."""

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

import copy
import importlib.util
import os
pass  # Path is provided by the portable boundary above
import sys
from service import ROOT,P26,LOCAL,read,write,asset,verify

COPY=['scene_manifest.json','baseline_freeze.json','baseline_verification.json','initial_context.json','baseline_input_audit.json','map/map.json','boot/manifest.json','scenario/scenario.json','scenario/target_track.npz','checks/cpu_admission.json','checks/native_cpu_replay/result.json']


def bind(champion):
    for p,h in champion['code_pins'].items():
        verify(dict(path=p,sha256=h))
    code=Path(champion['code'])
    sys.path[:0]=[str(code),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
    spec=importlib.util.spec_from_file_location('_discovery_carried_pipeline',code/'pipeline.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.bindings()
    # Bind the already verified neutral-SMPL support from stable v17, in both arms.
    support=ROOT/'integrations/light-execution-horizon-v17-r2/smpl_runtime.py'
    from service import OUT
    support_pin=read(OUT/'contract.json')['policies']['stable']['code_pins'][str(support)]
    verify(dict(path=str(support),sha256=support_pin))
    support_spec=importlib.util.spec_from_file_location('smpl_runtime',support)
    support_module=importlib.util.module_from_spec(support_spec);sys.modules['smpl_runtime']=support_module;support_spec.loader.exec_module(support_module)
    return module


def preflight(champion,output):
    module=bind(champion);module.check(champion['release'])
    sys.path.insert(0,str(ROOT/'integrations/autonomous-patch-agent-v7'))
    import probe
    # The actual GPU launch separately requires this existing SMPL preflight command.
    write(output,dict(registered_release_verified=True,carried_code=champion['code_pins'],worker=asset(Path(champion['code'])/'worker.py'),gpu_executed=False))


def prepare(task,champion,work,baseline=None):
    if task['scene']!=551 or task['program']!='registered_native_cutin_duration_variants':
        raise ValueError('No admitted closed-loop adapter for requested scene/program')
    work=Path(work);work.mkdir(parents=True,exist_ok=False);template=Path(champion['release']);module=bind(champion);module.check(template)
    for name in COPY:
        dst=work/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((template/name).read_bytes())
    plan=copy.deepcopy(read(template/'plan.json'))
    if baseline:
        # Generation is an offline edit operation, never an online policy change.
        import cutin_scenario
        config=copy.deepcopy(cutin_scenario.CONFIG)
        allowed={'lane_change_duration_s','actor_deceleration_duration_s'}
        if set(task['parameters'])-allowed:raise ValueError('Unregistered edit parameter')
        for k,v in task['parameters'].items():
            if not isinstance(v,(int,float)) or not 2.<=v<=6.:raise ValueError('Edit duration outside registered bounds')
        cutin_scenario.CONFIG=copy.deepcopy(config)|task['parameters']
        try:
            cutin_scenario.generate(work/'scenario',baseline)
        finally:
            cutin_scenario.CONFIG=config
        plan['baseline_root']=str(baseline)
    # Freeze the changed scenario without lying about the original release check.
    plan['assets'].update(scenario=asset(work/'scenario/scenario.json'),target_track=asset(work/'scenario/target_track.npz'),scene=asset(work/'scene_manifest.json'),baseline=asset(work/'baseline_freeze.json'))
    plan.update(discovery_task=task['id'],discovery_only=True,fresh_history_required=True,reference_reuse=False)
    write(work/'plan.json',plan)
    contract=dict(scene=551,edit=task['id'] if baseline else 'original',seed=plan['seed'],controls=plan['controls'],initial_context=plan['initial_context'],fresh_history=True,score_kind='executed_path',carried_policy=asset(Path(champion['code'])/'joint_safety.py'),scenario=asset(work/'scenario/scenario.json'),track=asset(work/'scenario/target_track.npz'),map=asset(work/'map/map.json'),manifest=asset(work/'scene_manifest.json'),worker=asset(Path(champion['code'])/'worker.py'),native_scorer=asset(ROOT/'code/HUGSIM/sim/utils/score_calculator.py'))
    contract.update(model_weights=plan['assets']['uniad_weights']['sha256'],native_collision_policy=plan['native_collision_policy'],termination=plan['termination'],tracker=asset(ROOT/'integrations/light-cutin-diagnostic/tracker.py'),initial_speed_mps=read(work/'scenario/scenario.json')['config']['initial_ego_speed_mps'])
    write(work/'discovery_contract.json',contract)
    write(work/'input_pins.json',{str(p):asset(p)['sha256'] for p in work.rglob('*') if p.is_file()})


def check_inputs(work):
    for path,h in read(Path(work)/'input_pins.json').items():verify(dict(path=path,sha256=h))


def execute(task,champion,work,arm,mode="on"):
    os.fstat(9) # Shared GPU lock must be inherited through the launcher.
    work=Path(work);check_inputs(work);module=bind(champion)
    import runtime,scoring,diagnostics
    if mode not in ('off','on'):raise ValueError('Explicit off/on required')
    import types
    switch_path=ROOT/'integrations/patch-v15-validation-r2-v1/controller_switch.py'
    spec=importlib.util.spec_from_file_location('controller_switch',switch_path)
    switch=importlib.util.module_from_spec(spec);sys.modules['controller_switch']=switch;spec.loader.exec_module(switch);switch.MODE=mode
    source=Path(champion['code'])/'runtime.py'
    tree=switch.rewrite_runtime(source.read_text())
    active=types.ModuleType('_discovery_paired_runtime');active.__file__=str(source)
    exec(compile(tree,str(source),'exec'),active.__dict__)
    report=active.run_arm(work,arm,read(work/'plan.json')['controls'],common=None,candidate=None)
    report.update(experiment_plan=asset(work/'plan.json'),model_weights_unchanged=True,discovery_contract=asset(work/'discovery_contract.json'))
    report.update(comparison_mode=mode,controls_selected_by='native_uniad_collision_postprocessing_plus_shared_ilqr' if mode=='off' else 'frozen_external_patch_plus_shared_ilqr',ground_truth_used_by_optimizer=mode=='on')
    write(work/arm/'run_report.json',report)
    scores=scoring.score_arm(work,arm);scores.update(ground_truth_used_by_optimizer=mode=='on',comparison_mode=mode)
    write(work/arm/'native_eval/native_scores.json',scores)
    summary=diagnostics.arm_summary(work/arm)
    collisions=[o['environment'] for o in report['observations'] if o['environment'].get('collision')]
    if report.get('terminal_collision'):
        collisions.append(report['terminal_collision'])
    # Fail closed if the collision provenance is not actually recorded.
    known=any(c.get('foreground_collision') or c.get('background_collision_with_known_semantics') for c in collisions)
    unknown=bool(collisions) and all(c.get('background_collision_depends_on_unknown_semantics') for c in collisions)
    summary['collision_depends_only_on_unknown_semantics']=False if known else True if unknown else None
    write(work/arm/'summary.json',summary);check_inputs(work)
    write(work/'completion.json',dict(report=asset(work/arm/'run_report.json'),summary=asset(work/arm/'summary.json'),contract=asset(work/'discovery_contract.json'),carried_code=champion['code_pins']))
    return summary


def matched_prefix(original,edited):
    import numpy as np
    a=read(Path(original)/'A/run_report.json');b=read(Path(edited)/'B/run_report.json')
    aa={o['index']:o for o in a['observations']};bb={o['index']:o for o in b['observations']}
    prefix=list(range(6))
    if not all(i in aa and i in bb and aa[i]['image_sha256']==bb[i]['image_sha256'] and np.allclose(aa[i]['pose'],bb[i]['pose'],atol=1e-9,rtol=0) for i in prefix):return False
    # Require camera effect at the same actual ego pose, before downstream divergence.
    return any(aa[i]['image_sha256']!=bb[i]['image_sha256'] and np.allclose(aa[i]['pose'],bb[i]['pose'],atol=1e-9,rtol=0) for i in sorted(set(aa)&set(bb)) if i>5)
