
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
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'code/HUGSIM').is_dir());ref=P.parent/'motion_search_v3';rows=[]
def digest(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def run(name,args):
 t=time.monotonic();r=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);rows.append(dict(name=name,returncode=r.returncode,wall_seconds=time.monotonic()-t));print(name,r.returncode,flush=True)
 with (P/'corridor_preflight.log').open('a') as log:log.write(name+'\n'+r.stdout+'\n')
 if r.returncode:print(r.stdout[-6000:]);raise SystemExit(r.returncode)
from corridor_acceptance import validate
validate(P)
d=_hugsim_expand(json.loads((P/'current_cpu_regression.json').read_text()));assert d['passed'] and d['tests']==210
for row in d['results']:assert digest(P/row['file'])==row['sha256']
run('cpu_gate',[sys.executable,'-B',str(P/'cpu_gate.py')]);run('wiring',[sys.executable,'-B',str(P/'check_wiring.py')])
legacy=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python';run('legacy_worker_import',[str(legacy),'-B','-c','import sys;sys.path.insert(0,'+repr(str(P))+');import worker;assert callable(worker.install)'])
readiness=_hugsim_expand(json.loads((P/'readiness.json').read_text()));readiness.update(diagnostic_launch_ready=True,performance_candidate_frozen=True,global_repair_verified=False,closed_loop_launch_ready=False,unit_tests=210,reason='Own corridor policy acceptance: 3 selected + 2 left branch checks; 210 CPU tests. No new GPU driving.',performance_only_change=False);(P/'readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
checks=dict(component_tests=True,own_policy_acceptance=True,independent_selected_geometry=True,independent_left_corridor_geometry=True,physics_and_evidence_preserved=True,startup_and_wiring=True,legacy_worker_import=True)
validation=dict(passed=True,unit_tests=210,required_checks=checks,runs=rows,checked_file_sha256={f.name:digest(f) for f in P.glob('*.py')},policy_changed=True,performance_only_change=False,gpu_executed=False,formal_evaluation_admitted=False,global_repair_verified=False)
(P/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
pins=_hugsim_expand(json.loads((ref/'manifest.json').read_text()))
for name,expected in pins.items():
 if digest(ROOT/name)!=expected:raise RuntimeError('Frozen parent changed: '+name)
for f in P.rglob('*'):
 if f.is_file() and '__pycache__' not in f.parts and f.name!='manifest.json' and f.suffix in ['.py','.json','.md','.sh','.npz','.npy','.pkl','.png']:pins[str(f.relative_to(ROOT))]=digest(f)
(P/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');print('Frozen corridor v4',len(pins),'files',flush=True)
