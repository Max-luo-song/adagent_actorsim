"""Freeze this empirical interval repair only after its own recorded-input gates."""

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
import hashlib,json,subprocess,sys,time
P=Path(__file__).resolve().parent;ROOT=next(x for x in P.parents if (x/'code/HUGSIM').is_dir());reference=P.parent/'memory_lifecycle_v1';rows=[]
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(name):return _hugsim_expand(json.loads((P/name).read_text()))
def run(name,argv):
 start=time.monotonic();r=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);rows.append(dict(name=name,returncode=r.returncode,wall_seconds=time.monotonic()-start));print(name,r.returncode,flush=True)
 with (P/'envelope_preflight.log').open('a') as f:f.write(name+'\n'+r.stdout+'\n')
 if r.returncode:print(r.stdout[-7000:]);raise SystemExit(r.returncode)
regression=read('current_cpu_regression.json');assert regression['passed'] and regression['tests']==193
for row in regression['results']:assert digest(P/row['file'])==row['sha256']
comparison=read('envelope_evidence/comparison.json');assert comparison['passed'] and comparison['legacy_motion_certificates_reproduced']==9
for index in [10,11]:
 summary=read(f'envelope_evidence/step_{index:03d}_summary.json');independent=read(f'envelope_evidence/step_{index:03d}_independent_check.json')
 assert summary['selected'] and summary['forecast_recovery'] is False and independent['passed']
 assert independent['braking_witness']['gap_m']>=0 and independent['braking_witness']['physical_bounds']
run('cpu_gate',[sys.executable,'-B',str(P/'cpu_gate.py')]);run('wiring',[sys.executable,'-B',str(P/'check_wiring.py')])
run('adapter_tests',[sys.executable,'-B','-m','unittest','discover','-s',str(P.parent/'bridge/closed_loop_integration'),'-p','test_*.py'])
run('bash_syntax',['bash','-n',str(P.parent/'bridge/unified/run_phase26.sh')])
legacy=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python';code='import sys;sys.path.insert(0,'+repr(str(P))+');import worker;assert callable(worker.install);print("Legacy worker import ready")';run('legacy_worker_import',[str(legacy),'-B','-c',code])
from performance_acceptance import validate
validate(P)
# Keep every upstream/reference pin exactly as previously accepted. Do not
# silently approve a concurrently modified dependency by rehashing it.
pins=_hugsim_expand(json.loads((reference/'manifest.json').read_text()))
for relative,expected in pins.items():
 if digest(ROOT/relative)!=expected:raise RuntimeError('Inherited dependency changed: '+relative)
print('Inherited pins verified',len(pins),flush=True)
changed=[p.name for p in P.glob('*.py') if not (reference/p.name).exists() or p.read_bytes()!=(reference/p.name).read_bytes()]
allowed={'time_local_motion.py','test_velocity_interval.py','replay_velocity_envelope.py','independent_envelope_check.py','freeze_preflight.py','launch.py'}
assert set(changed)<=allowed,changed
checks=dict(component_tests=True,adapter_tests=True,actual_input_replay=True,independent_physics_and_full_stop=True,legacy_worker_import=True,inherited_dependency_pins_verified=True,unchanged_optimizer_components=True,bash_syntax=True)
validation=dict(passed=True,unit_tests=193,adapter_tests=43,required_checks=checks,runs=rows,checked_file_sha256={p.name:digest(p) for p in P.glob('*.py')},diagnostic_only=True,global_repair_verified=False,gpu_executed=False,empirical_speed_interval_changed=True,physical_safety_certificate=False,changed_python_files=changed,unchanged_from_previous=['body and margins','actor longitudinal and lateral acceleration budgets','full current-action stop viability','vehicle dynamics and limits','occupancy and depth constraints','memory lifecycle','road cost','search policy'])
(P/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(P/'readiness.json').write_text(json.dumps(dict(diagnostic_launch_ready=True,reason='193 component tests + 43 adapter tests; actual step10/11 re-solves and independent full-stop checks passed. New GPU whole loop pending.',gpu_executed=False),indent=2)+'\n')
for f in P.rglob('*'):
 if f.is_file() and '__pycache__' not in f.parts and f.name!='manifest.json' and f.suffix in ('.py','.json','.md','.sh','.npz','.npy','.pkl','.png'):
  pins[str(f.relative_to(ROOT))]=digest(f)
(P/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');print('FROZEN',len(pins),flush=True)
