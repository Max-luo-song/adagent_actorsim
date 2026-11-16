
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
import ast,hashlib,json,numpy as np
P=Path(__file__).resolve().parent;CR=P.parents[1];SOURCE=CR/'phase40/mpc_runtime_v2';LIVE=CR/'phase39/runs/mpc_ten551_20261107T112411Z_83f645be/closed_loop/uniad_worker';HELD=CR/'phase37/runs/uncertainty_closed551_20261107T044006Z_bc47a553/closed_loop/uniad_worker'
def read(name):return _hugsim_expand(json.loads((P/name).read_text()))
def write(name,data):(P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def hashes(names):return {name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in names}
checks=read('runtime_benchmarks/independent_candidate_checks.json');assert checks['passed']
old=ast.parse((SOURCE/'empty_controller.py').read_text());new=ast.parse((P/'empty_controller.py').read_text())
for name in ('integrate_state','rollout','bounded_seed','clear','robust_clear','constraints','objective'):
 nodes=[next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in (old,new)];assert ast.dump(nodes[0])==ast.dump(nodes[1]),name
core=['native_objective.py','native_contract.py','time_local_motion.py','depth_consistency.py','ground_reference.py','depth_ambiguity.py','motion_policy.py','speed_policy.py','worker.py','protocol_branch_worker.py','occupancy_protocol.py']
for name in core:assert (P/name).read_bytes()==(SOURCE/name).read_bytes(),name
assert (P/'consensus_search.py').read_bytes()==(SOURCE/'consensus_search.py').read_bytes()
rows=[]
assert read('polygon_numerical_check.json')['passed'] and read('polygon_problem_equivalence.json')['passed']
for step in (8,9):
 old_result=read('runtime_benchmarks/step_%03d_baseline_result.json'%step);new_result=read('runtime_benchmarks/step_%03d_candidate_result.json'%step)
 without_timing=lambda row:{k:v for k,v in row.items() if k!='solve_wall_seconds'}
 assert without_timing(old_result['selected'])==without_timing(new_result['selected']),'Changed selected plan'
 assert [without_timing(r) for r in old_result['trials']]==[without_timing(r) for r in new_result['trials']],'Changed trials'
 a=read('runtime_benchmarks/step_%03d_baseline_summary.json'%step);b=read('runtime_benchmarks/step_%03d_candidate_summary.json'%step);check=next(row for row in checks['rows'] if row['step']==step);rows.append(dict(step=step,baseline_seconds=a['wall_seconds'],candidate_seconds=b['wall_seconds'],speedup=a['wall_seconds']/b['wall_seconds'],objective_increase_fraction=(b['objective']-a['objective'])/max(abs(a['objective']),1.),baseline_trials=a['trials'],candidate_trials=b['trials'],selected=b['selected'],forecast_recovery=b['forecast_recovery'],independent_passed=check['passed']))
held=read('held_out_benchmarks/step_002_candidate_result.json');reference=_hugsim_expand(json.loads((CR/'phase38/mpc_runtime_v1/runtime_benchmarks/step_002_consensus_result.json').read_text()));assert np.array_equal(held['selected']['controls'],reference['selected']['controls']);assert {r['seed'] for r in held['trials']}=={r['seed'] for r in reference['trials']};assert all(r['iteration_limit']==220 for r in held['trials'])
critical=list(read('evidence/verified_runtime_receipt.json')['implementation_sha256']);critical+=['baseline_consensus.py','consensus_search.py','performance_acceptance.py','fast_polygon.py','polygon_backend.py','polygon_kernel.cpp','polygon_build_manifest.json','polygon_build/libpolygon.so','build_polygon.py','empty_native_temporal.py','temporal_context.py'];critical=list(dict.fromkeys(critical))
artifacts=[]
for folder,steps,variants in [('runtime_benchmarks',(8,9),('baseline','candidate')),('held_out_benchmarks',(2,),('candidate',))]:
 for step in steps:
  for variant in variants:
   for suffix in ('summary','result'):artifacts.append('%s/step_%03d_%s_%s.json'%(folder,step,variant,suffix))
 artifacts.append(folder+'/independent_candidate_checks.json')
artifacts+=['polygon_numerical_check.json','polygon_problem_equivalence.json']
inputs={}
for root,indices in [(LIVE,(0,1,2,3,4,7,8,9)),(HELD,(1,2))]:
 for index in indices:
  for name in ('sensor_metadata.pkl','perception.npz','guard_memory_report.json','optimization.json'):
   path=root/('step_%03d'%index)/name
   inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
d=dict(schema='mpc_runtime_acceptance_v2',passed=True,physical_math_unchanged=True,paired_frames=rows,held_out_expansion=dict(step=2,expanded=held['guard']['search_schedule']['expanded'],original_starts_preserved=True,best_controls_identical=True,original_iteration_budget_preserved=True,fresh_solve=True),runtime_sha256=hashes(critical),artifact_sha256=hashes(artifacts),source_input_sha256=inputs,global_repair_verified=False,new_gpu_executed=False,scope='Phase40 search held fixed; only polygon backend changes. Two saved slow frames paired on one CPU plus fresh old full-search fallback. No whole-loop speedup claim.')
write('performance_acceptance.json',d)
write('evidence/verified_runtime_receipt.json',dict(implementation_sha256=d['runtime_sha256'],verification_kind='Fresh paired phase39 slow frames and full-budget held-out regression; unchanged objective and physical constraints',source_runtime='phase40/mpc_runtime_v2',global_repair_verified=False))
from performance_acceptance import validate
validate(P);print(json.dumps(d['paired_frames'],indent=2))
