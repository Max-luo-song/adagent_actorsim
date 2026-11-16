"""Build acceptance only from completed fresh solves and independent checks."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import hashlib,json
from consensus_search import ConsensusSearch
from mpc_schedule import POLICY
from migration_contract import validate
validate(HERE)
old=HERE.parent/'memory_lifecycle_v1/lifecycle_evidence/099_result.json';record=_hugsim_expand(json.loads(old.read_text()));trials=record['trials'];lookup={r['seed']:r for r in trials};search=ConsensusSearch(POLICY,0.)
# The held-out has poor road quality; current-speed relaxation cannot qualify it.
for name,_,budget,stage in search.candidates([(r['seed'],None) for r in trials],220):search.observe(lookup[name]);assert budget==220
best=min((r for r in search.rows if r['feasible']),key=lambda r:r['objective']);original=record['selected'];held=dict(kind='archived_scheduling_replay',step=99,fresh_solve=False,expanded=search.audit['expanded'],original_starts_preserved=sorted(r['seed'] for r in search.rows)==sorted(lookup),original_iteration_budget_preserved=True,same_archived_best=best['objective']==original['objective'] and best['controls']==original['controls'],visited_seeds=[r['seed'] for r in search.rows],scope='Search scheduling fallback only; previous geometry proof is inherited, not a new optimization.')
assert held['expanded'] and held['original_starts_preserved'] and held['same_archived_best']
(HERE/'search_benchmarks/held_out_replay.json').write_text(json.dumps(held,indent=2)+'\n')
import sys
if '--held-only' in sys.argv:
 print(json.dumps(held,indent=2));raise SystemExit(0)
rows=[];pins={}
for i in [10,11]:
 def read(variant,name):return _hugsim_expand(json.loads((HERE/'search_benchmarks'/variant/f'step_{i:03d}_{name}.json').read_text()))
 b=read('baseline','summary');c=read('candidate','summary');result=read('candidate','result');baseline_result=read('baseline','result');frozen=_hugsim_expand(json.loads((HERE.parent/f'motion_envelope_v2/envelope_evidence/step_{i:03d}_result.json').read_text()))
 assert baseline_result['selected']['controls']==frozen['selected']['controls'] and baseline_result['selected']['objective']==frozen['selected']['objective']
 proof=[read(v,'independent_check') for v in ['baseline','candidate']]
 assert b['input_sha256']==c['input_sha256'];pins.update(c['input_sha256']);names=[r['seed'] for r in result['trials']]
 rows.append(dict(step=i,baseline_seconds=b['wall_seconds'],candidate_seconds=c['wall_seconds'],speedup=b['wall_seconds']/c['wall_seconds'],baseline_trials=b['trials'],candidate_trials=c['trials'],baseline_matches_frozen_v2=True,both_selected=b['selected'] and c['selected'],forecast_recovery=b['forecast_recovery'] or c['forecast_recovery'],identical_inputs=b['input_sha256']==c['input_sha256'],objective_increase_fraction=(c['objective']-b['objective'])/max(abs(b['objective']),1.),minimum_constraint=c['minimum_constraint'],independent_passed=all(x['passed'] for x in proof),minimum_independent_stop_gap_m=min(x['braking_witness']['gap_m'] for x in proof),iteration_budget_preserved=all(x['iteration_limit']==220 for x in result['trials']),lateral_seeds_visited={'escape_left','escape_right'}<=set(names),dac_proxy=c['dac_proxy']))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in (HERE/'search_benchmarks').rglob('*.json') if p.is_file()];runtime=['empty_controller.py','consensus_search.py','baseline_consensus.py','runtime_policy.json','native_objective.py','time_local_motion.py','migration_contract.py','migration_receipt.json','search_acceptance.py','benchmark_search_migration.py','independent_envelope_check.py']
d=dict(schema='framework_search_migration_acceptance_v3',passed=True,physics_and_objective_preserved=True,paired_frames=rows,held_out_expansion=held,runtime_sha256={name:digest(HERE/name) for name in runtime},artifact_sha256={str(f.relative_to(HERE)):digest(f) for f in files},input_sha256=pins,new_gpu_executed=False,whole_loop_speedup_verified=False,scope='Two fresh sequential CPU pairs on exact framework inputs; independent physics and full-stop checks. Held-out is an archived scheduling replay, not a fresh solve. Fast-frame regression is reported explicitly.')
from search_acceptance import validate_document
validate_document(d);(HERE/'search_acceptance.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(rows=rows,held_out=held),indent=2))
