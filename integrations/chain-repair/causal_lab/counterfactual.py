"""N2 matched full-factorial evaluation and scene-cluster uncertainty."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
import random
from typing import Callable

from .core import canonical_hash


@dataclass(frozen=True)
class Outcome:
    scene_id: str
    seed: int
    factors: frozenset[str]
    policy_sha256: str
    participants_sha256: str
    start_frame: int
    failed: bool
    collision: bool
    route_passed: bool
    edit_valid: bool
    frames_executed: int

    def check(self) -> None:
        if not self.scene_id or not self.policy_sha256 or not self.participants_sha256:
            raise ValueError("missing experiment identity")
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.start_frame<0 or self.frames_executed<1:
            raise ValueError("invalid experiment time/seed")
        for name in ("failed","collision","route_passed","edit_valid"):
            if not isinstance(getattr(self,name),bool):raise ValueError("non-boolean outcome")
        if self.collision and not self.failed:raise ValueError("collision must count as failure")


def subsets(factors: tuple[str,...]) -> tuple[frozenset[str],...]:
    return tuple(frozenset(items) for n in range(len(factors)+1)
        for items in itertools.combinations(factors,n))


def _shapley(values: dict[frozenset[str],float], factors: tuple[str,...]) -> dict[str,float]:
    n=len(factors);answer={f:0. for f in factors}
    for factor in factors:
        rest=tuple(f for f in factors if f!=factor)
        for size in range(n):
            weight=1./(n*math.comb(n-1,size))
            for group in itertools.combinations(rest,size):
                key=frozenset(group)
                answer[factor]+=weight*(values[key|{factor}]-values[key])
    return answer


class CounterfactualEvaluator:
    """Runner receives identical policy, participants, seed and clock per pair."""
    def __init__(self, runner: Callable[[str,int,frozenset[str],str],Outcome]):
        self.runner=runner

    def collect(self, scene_seeds: list[tuple[str,int]], factors: tuple[str,...],
                policy_sha256: str) -> dict[tuple[str,int],dict[frozenset[str],Outcome]]:
        if not 1<=len(factors)<=4 or len(set(factors))!=len(factors) or not scene_seeds or len(set(scene_seeds))!=len(scene_seeds):
            raise ValueError("need distinct factors and scene/seed pairs; at most four factors")
        result={}
        variants=subsets(factors)
        for scene,seed in scene_seeds:
            rows={}
            for variant in variants:
                row=self.runner(scene,seed,variant,policy_sha256)
                row.check()
                if (row.scene_id,row.seed,row.factors,row.policy_sha256)!=(scene,seed,variant,policy_sha256):
                    raise ValueError("runner changed paired identity or policy")
                if not row.edit_valid:raise ValueError("N1 rejected an edited variant; do not score it")
                rows[variant]=row
            baseline=rows[frozenset()]
            if any(row.participants_sha256!=baseline.participants_sha256 or row.start_frame!=baseline.start_frame
                   for row in rows.values()):
                raise ValueError("actors or clock changed across counterfactual variants")
            result[scene,seed]=rows
        return result


def summarize(table: dict[tuple[str,int],dict[frozenset[str],Outcome]],
              factors: tuple[str,...], *, bootstrap=2000, rng_seed=172) -> dict:
    if not table:raise ValueError("empty factorial table")
    scenes=sorted({scene for scene,_ in table})
    all_factors=frozenset(factors)
    variants=set(subsets(factors))
    per_scene=defaultdict(list)
    for (scene,seed),rows in table.items():
        if set(rows)!=variants:raise ValueError("incomplete factorial design")
        vals={key:float(row.failed) for key,row in rows.items()}
        contrib=_shapley(vals,factors)
        total=vals[all_factors]-vals[frozenset()]
        if abs(sum(contrib.values())-total)>1e-9:raise AssertionError("Shapley conservation failed")
        per_scene[scene].append(dict(contrib=contrib,total=total,base=vals[frozenset()],
            full=vals[all_factors],collision_base=float(rows[frozenset()].collision),
            collision_full=float(rows[all_factors].collision)))

    def aggregate(sampled_scenes):
        records=[record for scene in sampled_scenes for record in per_scene[scene]]
        n=len(records)
        return dict(contributions={f:sum(row['contrib'][f] for row in records)/n for f in factors},
            failure_delta=sum(row['total'] for row in records)/n,
            baseline_failure_rate=sum(row['base'] for row in records)/n,
            full_failure_rate=sum(row['full'] for row in records)/n,
            collision_delta=sum(row['collision_full']-row['collision_base'] for row in records)/n)

    estimate=aggregate(scenes)
    randomizer=random.Random(rng_seed)
    replicates=[aggregate([randomizer.choice(scenes) for _ in scenes]) for _ in range(bootstrap)] if bootstrap else []
    def bounds(values):
        if not values:return [None,None]
        order=sorted(values);return [order[int(.025*(len(order)-1))],order[int(.975*(len(order)-1))]]
    ci={f:bounds([r['contributions'][f] for r in replicates]) for f in factors}
    ci['failure_delta']=bounds([r['failure_delta'] for r in replicates])
    ci['collision_delta']=bounds([r['collision_delta'] for r in replicates])
    # A percentage is only meaningful when all contributions point the same
    # way as the net effect. Opposing factors must be narrated as interactions.
    total=estimate['failure_delta']
    shares=(None if abs(total)<1e-12 or any(v*total < -1e-12 for v in estimate['contributions'].values())
        else {f:estimate['contributions'][f]/total for f in factors})
    interactions={}
    if len(factors)>=2:
        for a,b in itertools.combinations(factors,2):
            interactions[f'{a}×{b}']=sum(float(rows[frozenset((a,b))].failed)
                -float(rows[frozenset((a,))].failed)-float(rows[frozenset((b,))].failed)
                +float(rows[frozenset()].failed) for rows in table.values())/len(table)
    return dict(schema='n2_factorial_v1',factor_ids=list(factors),scene_count=len(scenes),paired_unit_count=len(table),
        policy_sha256=next(iter(table.values()))[frozenset()].policy_sha256,
        **estimate,confidence_intervals_95=ci,attribution_shares=shares,
        pair_interactions=interactions,bootstrap_scene_clusters=bootstrap,
        design_sha256=canonical_hash(dict(factors=factors,scene_seeds=sorted(table),variants=len(variants))),
        conclusion_scope='matched finite sample; estimates causal edit effect only if N1 validity and participant/clock locks hold')


def oracle_top1_accuracy(reports: list[dict], known_culprits: list[str]) -> dict:
    if len(reports)!=len(known_culprits) or not reports:raise ValueError("oracle cases missing")
    correct=0
    for report,truth in zip(reports,known_culprits):
        effects=report['contributions']
        if truth not in effects:raise ValueError("unknown oracle culprit")
        # The injected factor may remove a hazard or add one; do not reward ties.
        best=max(abs(v) for v in effects.values())
        winners=[f for f,v in effects.items() if abs(abs(v)-best)<1e-12]
        correct+=int(winners==[truth])
    return dict(oracle_cases=len(reports),top1_correct=correct,top1_accuracy=correct/len(reports))
