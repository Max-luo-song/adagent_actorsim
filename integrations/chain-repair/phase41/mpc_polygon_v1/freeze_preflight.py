"""Run actual CPU gates, then pin this independent diagnostic package."""

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

import hashlib,json,os,subprocess,sys,time
pass  # Path is provided by the portable boundary above
P=Path(__file__).resolve().parent;ROOT=next(x for x in P.parents if (x/'code/HUGSIM').is_dir());CR=P.parents[1];rows=[]
def run(name,argv):
 start=time.monotonic();r=subprocess.run(argv,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);rows.append(dict(script=name,returncode=r.returncode,wall_seconds=time.monotonic()-start));print(name,r.returncode,flush=True)
 with (P/'preflight.log').open('a') as f:f.write(name+'\n'+r.stdout+'\n')
 if r.returncode:print(r.stdout[-6000:],flush=True);raise SystemExit(r.returncode)
run('cpu_regression',[sys.executable,'-B',str(P/'run_cpu_regression.py')])
run('cpu_gate',[sys.executable,'-B',str(P/'cpu_gate.py')]);run('check_wiring',[sys.executable,'-B',str(P/'check_wiring.py')])
legacy=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python';code='import sys;sys.path.insert(0,'+repr(str(P))+');import worker;assert callable(worker.install);print("Legacy worker import ready")';run('legacy_worker_import',[str(legacy),'-B','-c',code])
run('bash_syntax',['bash','-n',str(P/'run.sh')]);run('actual_problem_equivalence',[sys.executable,'-B',str(P/'verify_polygon_problem.py')])
run('independent_saved_geometry',[sys.executable,'-B',str(P/'independent_depth_check.py'),'--steps','0','9','14'])
read=lambda name:_hugsim_expand(json.loads((P/name).read_text()));source=read('evidence/verified_runtime_receipt.json')
for name,digest in source['implementation_sha256'].items():
 if hashlib.sha256((P/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed runtime: '+name)
run('independent_performance_geometry',[sys.executable,'-B',str(P/'verify_runtime_benchmarks.py'),'--variant','candidate','--steps','8','9'])
run('independent_held_out_geometry',[sys.executable,'-B',str(P/'verify_runtime_benchmarks.py'),'--variant','candidate','--steps','2','--source','phase37','--output-folder','held_out_benchmarks'])
from performance_acceptance import validate as validate_performance
validate_performance(P)
readiness=read('readiness.json');readiness.update(diagnostic_launch_ready=True,performance_candidate_frozen=True,unit_tests=read('current_cpu_regression.json')['tests'],reason='Paired CPU performance acceptance and independent geometry passed; frozen for the next bounded diagnostic. Whole-loop speedup and global behavior remain unverified.');(P/'readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
checks=dict.fromkeys(('cpu_regression','startup_and_wiring','legacy_worker_import','independent_saved_geometry','runtime_unchanged_from_verified_source','bash_syntax','performance_acceptance'),True)
files=[f for f in P.iterdir() if f.is_file() and f.suffix=='.py']+[P/name for name in ('arm.json','request_template.json','smoke_scope.json','diagnostic_admission.json','run.sh')]
validation=dict(passed=True,unit_tests=read('current_cpu_regression.json')['tests'],required_checks=checks,runs=rows,checked_file_sha256={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},diagnostic_only=True,global_repair_verified=False,gpu_executed=False)
(P/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
pins={}
for relative in _hugsim_expand(json.loads((CR/'phase40/mpc_runtime_v2/manifest.json').read_text())):
 path=ROOT/relative
 if not path.is_file():raise FileNotFoundError('Previously pinned dependency missing: '+str(path))
 pins[relative]=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
for name,digest in read('performance_acceptance.json')['source_input_sha256'].items():
 pins[str(Path(name).relative_to(ROOT))]=digest
for f in P.rglob('*'):
 if f.is_file() and '__pycache__' not in f.parts and f.name!='manifest.json' and f.suffix in ('.py','.json','.md','.sh','.npz','.npy','.pkl','.cpp','.so'):
  pins[str(f.relative_to(ROOT))]=hashlib.file_digest(f.open('rb'),'sha256').hexdigest()
(P/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');run('frozen_launch_check',[sys.executable,'-B',str(P/'launch.py'),'--check']);print('FROZEN',len(pins),'files;',validation['unit_tests'],'tests',flush=True)
