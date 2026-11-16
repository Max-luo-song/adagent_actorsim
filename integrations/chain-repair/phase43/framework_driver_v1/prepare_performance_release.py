
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

pass  # Path is provided by the portable boundary above
import ast,hashlib,json,shutil,numpy as np
P=Path(__file__).resolve().parent;CR=P.parents[1];SOURCE=CR/'phase37/uncertainty_closed_v1';LIVE=CR/'phase37/runs/uncertainty_closed551_20261107T044006Z_bc47a553/closed_loop/uniad_worker'
def read(name):return _hugsim_expand(json.loads((P/name).read_text()))
def write(name,data):(P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def hashes(names):return {name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in names}
checks=read('runtime_benchmarks/independent_consensus_checks.json');assert checks['passed']
old=ast.parse((SOURCE/'empty_controller.py').read_text());new=ast.parse((P/'empty_controller.py').read_text())
for name in ('integrate_state','rollout','bounded_seed','clear','robust_clear','constraints','objective'):
 nodes=[next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in (old,new)];assert ast.dump(nodes[0])==ast.dump(nodes[1]),name
core=['native_objective.py','native_contract.py','time_local_motion.py','depth_consistency.py','ground_reference.py','depth_ambiguity.py','motion_policy.py','speed_policy.py','worker.py','protocol_branch_worker.py','occupancy_protocol.py']
for name in core:assert (P/name).read_bytes()==(SOURCE/name).read_bytes(),name
rows=[]
for step in (0,1):
 a=read('runtime_benchmarks/step_%03d_baseline_summary.json'%step);b=read('runtime_benchmarks/step_%03d_consensus_summary.json'%step);check=next(row for row in checks['rows'] if row['step']==step);rows.append(dict(step=step,baseline_seconds=a['wall_seconds'],candidate_seconds=b['wall_seconds'],speedup=a['wall_seconds']/b['wall_seconds'],objective_increase_fraction=(b['objective']-a['objective'])/max(abs(a['objective']),1.),baseline_trials=a['trials'],candidate_trials=b['trials'],selected=b['selected'],forecast_recovery=b['forecast_recovery'],independent_passed=check['passed']))
held=read('runtime_benchmarks/step_002_consensus_result.json');reference=_hugsim_expand(json.loads((LIVE/'step_002/optimization.json').read_text()));assert np.array_equal(held['selected']['controls'],reference['selected']['controls']);assert {r['seed'] for r in held['trials']}=={r['seed'] for r in reference['trials']};assert all(r['iteration_limit']==220 for r in held['trials'])
shutil.copy2(LIVE/'step_002/optimization.json',P/'evidence/held_out_full_search_reference.json');shutil.copy2(P/'current_cpu_regression.json',P/'evidence/performance_cpu_regression.json')
assert read('evidence/performance_cpu_regression.json')['passed']
critical=list(read('evidence/verified_runtime_receipt.json')['implementation_sha256']);critical+=['runtime_policy.json','fast_road.py','scaled_solver.py','reference_geometry.py','consensus_search.py','performance_acceptance.py'];critical=list(dict.fromkeys(critical))
artifacts=['runtime_benchmarks/step_000_baseline_summary.json','runtime_benchmarks/step_001_baseline_summary.json','runtime_benchmarks/step_000_consensus_summary.json','runtime_benchmarks/step_001_consensus_summary.json','runtime_benchmarks/step_002_consensus_summary.json','runtime_benchmarks/step_000_consensus_result.json','runtime_benchmarks/step_001_consensus_result.json','runtime_benchmarks/step_002_consensus_result.json','runtime_benchmarks/independent_consensus_checks.json','evidence/held_out_full_search_reference.json','evidence/performance_cpu_regression.json','reference_schedule_replay.json']
d=dict(schema='mpc_runtime_acceptance_v1',passed=True,physical_math_unchanged=True,paired_frames=rows,held_out_expansion=dict(step=2,expanded=held['guard']['search_schedule']['expanded'],original_starts_preserved=True,best_controls_identical=True,original_iteration_budget_preserved=True),runtime_sha256=hashes(critical),artifact_sha256=hashes(artifacts),global_repair_verified=False,new_gpu_executed=False,scope='Two single-run paired CPU comparisons and one held-out full-expansion check; not a whole-loop speedup certificate.')
write('performance_acceptance.json',d)
original=P/'evidence/original_verified_runtime_receipt.json'
if not original.exists():shutil.copy2(P/'evidence/verified_runtime_receipt.json',original)
write('evidence/verified_runtime_receipt.json',dict(implementation_sha256=d['runtime_sha256'],verification_kind='Paired performance candidate; unchanged objective/physical constraints, changed search termination and reference scheduling',source_runtime='phase37/uncertainty_closed_v1',global_repair_verified=False))
from performance_acceptance import validate
validate(P);print(json.dumps(d['paired_frames'],indent=2))
