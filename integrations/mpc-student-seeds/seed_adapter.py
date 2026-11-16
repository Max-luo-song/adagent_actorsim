"""One student expansion start, always solved by the original full-budget MPC."""

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

import ast
import hashlib
pass  # Path is provided by the portable boundary above
import sys
import time
import types
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'offline-mpc-distillation'))
from mpc_policy import Provider,control_bounds,vehicle_hash

CALL='candidates=ranked_candidates(consensus,seeds,maxiter,current_seed_score)'
PROBLEM="_student_problem(reference=original_neural_reference,info=info,reliability=reliability,vehicle=vehicle,initial=initial,offset=offset,interval=control_interval_s,interpolation=interpolation,bounds=bounds,simulate=simulate,constraints=constraints,objective=objective,clear=clear,aligned=aligned,guard=guard,temporal=temporal_occupancy,road=road,audit=audit)"
REPLACEMENT='candidates=_student_candidates(ranked_candidates,consensus,seeds,maxiter,current_seed_score,lambda: _student_provider('+PROBLEM+',seeds,current_seed_score),audit)'
ROW="        trials.append(row)"
ROW_REPLACEMENT=ROW+"\n        if name.startswith('student_initial_'):\n            admitted=_student_admit(row,consensus)\n            row['student_initial_only']=True\n            row['student_post_solve_admitted']=admitted\n            if not admitted:\n                feasible=False\n                row['feasible']=False"


def admit(row,consensus):
    return bool(consensus is not None and row.get('solver_success') is True and not row.get('used_verified_seed')
        and not row.get('optimization_skipped') and consensus.acceptable(row))


class SeedProvider:
    def __init__(self,checkpoint,expected_sha256,max_seeds=1):
        self.base=Provider(checkpoint,expected_sha256);self.max_seeds=max_seeds
        if max_seeds!=1:raise ValueError('This experiment admits exactly one extra expansion start')
    def __call__(self,problem,original_seeds,score):
        began=time.perf_counter();audit=dict(direct_control=False,current_full_solver_required=True,
            inference_executed=False,used_ground_truth=False,max_extra_starts=1)
        def finish(seeds,reason):
            audit.update(reason=reason,provider_seconds=time.perf_counter()-began)
            return seeds,audit
        if vehicle_hash(vars(problem.vehicle))!=self.base.state['vehicle_identity']:return finish([],'vehicle_mismatch')
        from policy_features import encode,FEATURE_SCHEMA
        if FEATURE_SCHEMA!=self.base.state['feature_schema']:return finish([],'feature_schema_mismatch')
        features=np.asarray(encode(problem),np.float32)
        if features.shape!=(self.base.state['dimension'],) or not np.isfinite(features).all():return finish([],'invalid_descriptor')
        x=(features-self.base.mean)/self.base.std
        if not np.isfinite(x).all():return finish([],'invalid_normalized_descriptor')
        distance=float(np.sqrt(np.mean((self.base.training-x)**2,axis=1)).min())
        # OOD remains visible. It does not grant direct control: the solver must
        # converge and its usual current constraints/consensus still decide.
        audit.update(nearest_training_distance=distance,outside_training_support=distance>self.base.state['distance_threshold'],
            ood_role='initial_guess_only; never direct control')
        with torch.inference_mode():
            logits,controls=self.base.model(torch.from_numpy(x)[None]);prob=logits.softmax(-1)[0].numpy();controls=controls[0].numpy()
        audit['inference_executed']=True
        lo,hi=control_bounds(vars(problem.vehicle));controls=lo+(controls+1.)*.5*(hi-lo)
        from empty_controller import bounded_seed
        v=problem.vehicle;original=[bounded_seed(s,problem.initial[3],problem.initial[4],problem.interval,v) for _,s in original_seeds]
        candidates=[];ranking=[]
        for mode in self.base.state['trained_modes']:
            seed=bounded_seed(controls[mode],problem.initial[3],problem.initial[4],problem.interval,v)
            if not np.isfinite(seed).all():continue
            if any(np.allclose(seed,s,rtol=0,atol=1e-10) for s in original):continue
            name='student_initial_mode_'+str(mode);gap,cost=score(name,seed)
            finite=bool(np.isfinite([gap,cost]).all())
            ranking.append(dict(mode=mode,minimum_constraint=float(gap) if np.isfinite(gap) else None,
                objective=float(cost) if np.isfinite(cost) else None,probability=float(prob[mode]),finite=finite))
            if finite:
                key=(gap< -1e-5,cost if gap>=-1e-5 else -gap,cost,-float(prob[mode]))
                candidates.append((key,name,seed))
        candidates.sort(key=lambda x:x[0]);audit['current_seed_ranking']=ranking
        chosen=[(name,seed) for _,name,seed in candidates[:1]];audit['proposed_seeds']=[name for name,_ in chosen]
        return finish(chosen,'extra_initial_guess' if chosen else 'no_finite_distinct_initial_guess')


def candidates(original,search,seeds,maxiter,evaluate,provide,audit):
    """Preserve original primary starts and budgets; add one start only on expansion."""
    inserted=False
    for item in original(search,seeds,maxiter,evaluate):
        if item[3]=='consensus_expansion' and not inserted:
            inserted=True;extra,receipt=provide();audit['student_seed']=receipt
            receipt['original_expansion_order_preserved']=True
            for name,seed in extra:
                yield name,seed,maxiter,'student_expansion_initial_guess'
                # Exactly the original convergence / quality / 1% agreement gate.
                if search.agreement():
                    receipt['remaining_original_starts_skipped_by_original_agreement']=True
                    return
        yield item
    if not inserted:audit['student_seed']=dict(reason='original_search_finished_without_expansion',inference_executed=False,direct_control=False)


def build(original,provider):
    """Compile an isolated function with two audited learned-seed-only changes.

    Original globals are snapshotted after configure(), preserving EvidenceGuard.
    __file__ stays the source file so its existing config-relative paths retain
    their meaning. No upstream file or module global is modified.
    """
    path=Path(original.__file__).resolve();text=path.read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='optimize')
    lines=text.splitlines(keepends=True);start=min([node.lineno]+[n.lineno for n in node.decorator_list])-1
    source=''.join(lines[start:node.end_lineno])
    if source.count(CALL)!=1 or source.count(ROW)!=1:raise ValueError('Controller seed call sites changed')
    patched=source.replace(CALL,REPLACEMENT).replace(ROW,ROW_REPLACEMENT)
    # Reject any unreviewed changes to objective, constraints, physics or limits.
    if patched.replace(REPLACEMENT,CALL).replace(ROW_REPLACEMENT,ROW)!=source:raise ValueError('Unexpected controller mutation')
    from direct_problem import PlanningProblem
    module=types.ModuleType('student_seed_controller_v1');module.__dict__.update(vars(original))
    module.__name__='student_seed_controller_v1'
    module.__dict__.update(_student_provider=provider,_student_problem=PlanningProblem,_student_candidates=candidates,_student_admit=admit)
    exec(compile(patched,str(HERE/'isolated_optimize_function.py'),'exec'),module.__dict__)
    return module,dict(source_controller=str(path),source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_function_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        declared_edit='ranked-expansion call site plus converged/quality admission for extra student rows only; original objective, constraints, solver and fallbacks unchanged',
        source_files_modified=False,primary_start_budgets_unchanged=True,extra_start_budget=220,
        direct_student_control=False,compiled_function_source=patched)
