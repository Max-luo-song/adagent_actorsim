"""Fixed-input sequential baseline/student-initial-guess MPC comparisons."""

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
import datetime
import hashlib
import json
pass  # Path is provided by the portable boundary above
import pickle
import sys
import time
import numpy as np
import torch
from seed_adapter import SeedProvider,build

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'offline-mpc-distillation'))
import pipeline as old
ROOT=old.ROOT;SOURCE=old.SOURCE;DRIVER=old.DRIVER
PARENT=SOURCE/'resumed_runs/20261109T084609Z'
CASES=[0,8,23]
read=old.read;write=old.write;asset=old.asset;verify=old.verify


def pins():
    result=old.pins()
    result.update({str(p):old.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh')})
    return result


def prepare(output):
    out=Path(output).resolve();trained=read(PARENT/'student/receipt.json');verify(trained['checkpoint'])
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    inputs={str(k):asset(SOURCE/'data_inputs/test_cutin'/f'step_{k:03d}.pkl') for k in CASES}
    write(out/'plan.json',dict(schema='student_mpc_expansion_initial_guess_v1',code=pins(),inputs=inputs,
        student=trained['checkpoint'],parent_training_receipt=asset(PARENT/'student/receipt.json'),cases=CASES,
        pair_order={'0':['baseline','student'],'8':['student','baseline'],'23':['baseline','student']},
        selection='preregistered initial/previous direct-feasible/expansion-event cases; reused episode, not new generalization evidence',
        new_closed_loop=False,max_extra_starts=1,maxiter=220,primary_schedule_unchanged=True,
        constraints_objective_dynamics_unchanged=True,student_direct_control=False,automatic_promotion=False))
    return dict(prepared=True,output=str(out),cases=CASES)


def load(output):
    out=Path(output).resolve();p=read(out/'plan.json')
    if p['code']!=pins():raise ValueError('Frozen experiment implementation changed')
    verify(p['student']);verify(p['parent_training_receipt'])
    for a in p['inputs'].values():verify(a)
    return out,p


def run_case(output,step,variant):
    out,p=load(output)
    if str(step) not in p['inputs']:raise ValueError('Case not preregistered')
    folder=out/f'step_{step:03d}';folder.mkdir(exist_ok=True)
    path=folder/(variant+'.json')
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(1);driver=old.configure_source();original=driver.optimized_controller
    if variant=='student':
        provider=SeedProvider(p['student']['path'],p['student']['sha256'])
        controller,migration=build(original,provider)
        write(folder/'migration.json',{k:v for k,v in migration.items() if k!='compiled_function_source'})
        (folder/'isolated_optimize_function.py').write_text(migration['compiled_function_source'])
    else:controller=original
    with open(p['inputs'][str(step)]['path'],'rb') as handle:data=pickle.load(handle)
    kwargs=copy.deepcopy(data['kwargs']);kwargs.update(_continuation=None,_check_continuation_only=False)
    # Identical saved current problem, warm control and full solver budget.
    kwargs.pop('_proposal_provider',None);kwargs.pop('_problem_callback',None);kwargs.pop('maxiter',None)
    wall=time.perf_counter();cpu=time.process_time()
    result=controller.optimize(data['reference'],data['occupancy'],data['info'],maxiter=220,**kwargs)
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    write(path,result)
    from direct_problem import make_problem,PlanningProblem
    problem=make_problem(original,data['reference'],data['occupancy'],data['info'],kwargs)
    selected=result.get('selected');validated=None
    if selected:
        if not isinstance(problem,PlanningProblem):raise ValueError('Original problem reconstruction failed')
        validated=problem.evaluate(selected['controls'],quality=False)
        if validated is None:raise ValueError('Selected controls fail original current full constraints')
        np.testing.assert_allclose(validated['states'],selected['states'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(validated['objective'],selected['objective'],rtol=1e-10,atol=1e-8)
        if selected['seed'].startswith('student_initial_'):
            if selected.get('student_post_solve_admitted') is not True or selected.get('optimization_skipped') or selected.get('solver_success') is not True:
                raise ValueError('Unoptimized student controls were admitted')
    trial=result.get('trials',[]);extra=[r for r in trial if r['seed'].startswith('student_initial_')]
    metadata=dict(completed=True,step=step,variant=variant,wall_seconds=wall,process_cpu_seconds=cpu,
        input=p['inputs'][str(step)],result=asset(path),plan=asset(out/'plan.json'),
        objective=selected['objective'] if selected else None,selected_seed=selected['seed'] if selected else None,
        first_control=selected['controls'][0] if selected else None,
        full_current_constraints_passed=validated is not None,trial_count=len(trial),
        solved_starts=sum(not r.get('optimization_skipped') for r in trial),
        extra_student_trials=len(extra),student_converged_trials=sum(r.get('solver_success') is True for r in extra),
        student_admitted_trials=sum(r.get('student_post_solve_admitted') is True for r in extra),
        primary_seeds=[r['seed'] for r in trial if r.get('search_stage')=='consensus_primary'],
        iteration_limits=[r['iteration_limit'] for r in trial],
        search_audit=result.get('guard',{}).get('search_schedule'),student_audit=result.get('guard',{}).get('student_seed'),
        proxy=selected.get('native_alignment_proxy') if selected else None,checkpoint=p['student'],
        direct_student_control=False,new_closed_loop_executed=False)
    write(folder/(variant+'_receipt.json'),metadata)
    print(json.dumps(metadata,indent=2,allow_nan=False));return metadata


def compare(output):
    out,p=load(output);rows=[]
    for step in p['cases']:
        folder=out/f'step_{step:03d}';b=read(folder/'baseline_receipt.json');s=read(folder/'student_receipt.json')
        for r in [b,s]:verify(r['input']);verify(r['result']);verify(r['plan'])
        if b['input']!=s['input']:raise ValueError('Unmatched current inputs')
        increase=None if b['objective'] is None or s['objective'] is None else (s['objective']-b['objective'])/max(abs(b['objective']),1.)
        primary_same=b['primary_seeds']==s['primary_seeds']
        checks=dict(full_current_constraints_passed=s['full_current_constraints_passed'],
            primary_starts_preserved=primary_same,full_iteration_budgets_preserved=all(i==220 for i in s['iteration_limits']),
            objective_not_more_than_1pct_worse=increase is not None and increase<=.01,
            comfort_not_regressed=bool(s['proxy'] and b['proxy'] and s['proxy']['comfort_native_kinematic_flag']>=b['proxy']['comfort_native_kinematic_flag']),
            road_proxy_not_regressed=bool(s['proxy'] and b['proxy'] and s['proxy']['dac_proxy']>=b['proxy']['dac_proxy']-1e-6))
        rows.append(dict(step=step,baseline_seconds=b['wall_seconds'],student_seconds=s['wall_seconds'],
            time_reduction_fraction=1-s['wall_seconds']/b['wall_seconds'],objective_increase_fraction=increase,
            baseline_trials=b['trial_count'],student_trials=s['trial_count'],student_initial_trials=s['extra_student_trials'],
            student_selected=s['selected_seed'].startswith('student_initial_') if s['selected_seed'] else False,
            selected_seed=s['selected_seed'],checks=checks,quality_passed=all(checks.values())))
    baseline=sum(r['baseline_seconds'] for r in rows);student=sum(r['student_seconds'] for r in rows)
    result=dict(completed=True,cases=rows,baseline_seconds=baseline,student_seconds=student,
        aggregate_time_reduction_fraction=1-student/baseline,all_quality_checks_passed=all(r['quality_passed'] for r in rows),
        performance_admitted=all(r['quality_passed'] for r in rows) and student<baseline,
        scope='three selected sequential fixed-input CPU pairs; no full-loop speed or generalization certificate',
        new_closed_loop_executed=False,automatic_promotion=False)
    write(out/'comparison.json',result);return result


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['prepare','run','compare','check']);a.add_argument('--output',required=True)
    a.add_argument('--step',type=int);a.add_argument('--variant',choices=['baseline','student']);args=a.parse_args()
    if args.stage=='run' and (args.step is None or args.variant is None):a.error('run requires --step and --variant')
    actions=dict(prepare=lambda:prepare(args.output),run=lambda:run_case(args.output,args.step,args.variant),
        compare=lambda:compare(args.output),check=lambda:dict(verified=True,cases=load(args.output)[1]['cases']))
    print(json.dumps(actions[args.stage](),indent=2,allow_nan=False))
