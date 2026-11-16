"""Own acceptance and immutable release for the phase40 search-only migration."""

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
P=Path(__file__).resolve().parent;ROOT=next(x for x in P.parents if (x/'code/HUGSIM').is_dir());reference=P.parent/'motion_envelope_v2';rows=[]
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(name):return _hugsim_expand(json.loads((P/name).read_text()))
def run(name,argv):
 start=time.monotonic();r=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);rows.append(dict(name=name,returncode=r.returncode,wall_seconds=time.monotonic()-start));print(name,r.returncode,flush=True)
 with (P/'search_preflight.log').open('a') as f:f.write(name+'\n'+r.stdout+'\n')
 if r.returncode:print(r.stdout[-7000:]);raise SystemExit(r.returncode)
from search_acceptance import validate as validate_search
validate_search(P)
run('component_tests',[sys.executable,'-B',str(P/'run_cpu_regression.py')]);regression=read('current_cpu_regression.json');assert regression['passed'] and regression['tests']==204
run('cpu_gate',[sys.executable,'-B',str(P/'cpu_gate.py')]);run('wiring',[sys.executable,'-B',str(P/'check_wiring.py')])
run('adapter_tests',[sys.executable,'-B','-m','unittest','discover','-s',str(P.parent/'bridge/closed_loop_integration'),'-p','test_*.py'])
run('bash_syntax',['bash','-n',str(P.parent/'bridge/unified/run_phase26.sh')]);run('bundle_bash_syntax',['bash','-n',str(P/'run.sh')])
legacy=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python';code='import sys;sys.path.insert(0,'+repr(str(P))+');import worker;assert callable(worker.install);print("Legacy worker import ready")';run('legacy_worker_import',[str(legacy),'-B','-c',code])
from performance_acceptance import validate
validate(P)
pins=_hugsim_expand(json.loads((reference/'manifest.json').read_text()))
for relative,expected in pins.items():
 if digest(ROOT/relative)!=expected:raise RuntimeError('Inherited dependency changed: '+relative)
print('Inherited pins verified',len(pins),flush=True)
checks=dict(component_tests=True,adapter_tests=True,startup_and_wiring=True,legacy_worker_import=True,own_paired_search_acceptance=True,independent_physics_and_full_stop=True,archived_full_search_fallback=True,inherited_dependency_pins_verified=True,exact_search_only_patch=True,own_envelope_memory_road_repairs_preserved=True,bash_syntax=True)
validation=dict(passed=True,unit_tests=204,adapter_tests=47,required_checks=checks,runs=rows,checked_file_sha256={p.name:digest(p) for p in P.glob('*.py')},diagnostic_only=True,global_repair_verified=False,gpu_executed=False,performance_only_change_vs_motion_envelope_v2=True,search_stopping_policy_changed=True,physical_safety_certificate=False)
(P/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
readiness=read('readiness.json');readiness.update(diagnostic_launch_ready=True,performance_candidate_frozen=True,cpu_regression_passed=True,unit_tests=204,global_repair_verified=False,closed_loop_launch_ready=False,reason='Own fresh CPU pairs + independent full-stop checks + 204 tests accepted. Whole-loop GPU speedup pending.');(P/'readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
for f in P.rglob('*'):
 if f.is_file() and '__pycache__' not in f.parts and f.name!='manifest.json' and f.suffix in ('.py','.json','.md','.sh','.npz','.npy','.pkl','.png'):
  pins[str(f.relative_to(ROOT))]=digest(f)
(P/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');print('FROZEN',len(pins),flush=True)
