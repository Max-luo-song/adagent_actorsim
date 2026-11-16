"""Own integration, edit, dependency and release verification before GPU load."""

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

import argparse,ast,hashlib,importlib.util,json,os,re,subprocess,sys,time
pass  # Path is provided by the portable boundary above
from paths import *
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def verify_frozen():
 ready=_hugsim_expand(json.loads((HERE/'readiness.json').read_text()))
 if ready.get('diagnostic_launch_ready') is not True:raise ValueError('New cut-in integration has not passed own CPU freeze')
 pins=_hugsim_expand(json.loads((HERE/'manifest.json').read_text()))
 for name,digest in pins.items():
  if sha(ROOT/name)!=digest:raise ValueError('Changed frozen input: '+name)
 print('Cut-in experiment SHA256 verified:',len(pins),'files',flush=True)
def build():
 initial_ready=dict(diagnostic_launch_ready=False,cpu_regression_passed=False,performance_candidate_frozen=False,global_repair_verified=False,closed_loop_launch_ready=False,formal_evaluation_admitted=False,native_evaluation_admitted=False,reason='Own integration preflight in progress; GPU pending')
 for folder in (HERE,DRIVER):(folder/'readiness.json').write_text(json.dumps(initial_ready,indent=2)+'\n')
 from migration import validate
 migration=validate();env=dict(os.environ,LD_LIBRARY_PATH=str(HERE/'native_runtime/lib'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');py=str(ROOT/'code/HUGSIM/.pixi/envs/default/bin/python');rows=[]
 log=HERE/'validation/preflight.log';log.parent.mkdir(exist_ok=True)
 def run(name,args):
  start=time.monotonic();r=subprocess.run(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);match=re.search(r'Ran (\d+) tests?',r.stdout);rows.append(dict(name=name,passed=r.returncode==0,tests=int(match.group(1)) if match else 0,wall_seconds=time.monotonic()-start))
  with log.open('a') as f:f.write(name+'\n'+r.stdout+'\n')
  print(name,r.returncode,flush=True)
  if r.returncode:print(r.stdout[-5000:],flush=True);raise RuntimeError('CPU preflight failed: '+name)
 # These two old release tests assert the OLD envelope->search-only migration.
 # The replacement migration check above protects all our actual new changes.
 replaced={'test_performance_admission.py','test_search_migration.py'}
 for test in sorted(DRIVER.glob('test_*.py')):
  if test.name not in replaced:run(test.name,[py,'-B',str(test)])
 run('native_car_and_score_input_tests',[py,'-B',str(HERE/'test_cutin.py')])
 run('actual_launcher_import_order',[py,'-B',str(HERE/'validate_imports.py')])
 run('native_score_dependencies',[py,'-B','-c',"import importlib.util; p="+repr(str(SCORER))+";s=importlib.util.spec_from_file_location('official_score',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert callable(m.hugsim_evaluate);print('Unmodified native scoring dependencies ready')"])
 run('legacy_worker_import',[str(ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),'-B','-c',"import sys;sys.path.insert(0,"+repr(str(DRIVER))+");import worker;assert callable(worker.install)"])
 run('shell_syntax',['bash','-n',str(HERE/'run.sh')])
 for mode in ("baseline","candidate"):
  checks=_hugsim_expand(json.loads((HERE/"validation/benchmarks"/mode/"independent_checks.json").read_text()))
  if not checks.get("passed") or not all(r.get("passed") for r in checks["rows"]):raise ValueError("Independent integrated geometry verification failed")
 benchmarks=[]
 for i in (10,11):
  folder=HERE/'validation/benchmarks';a=_hugsim_expand(json.loads((folder/('baseline_step_%03d_result.json'%i)).read_text()));b=_hugsim_expand(json.loads((folder/('candidate_step_%03d_result.json'%i)).read_text()));strip=lambda row:{k:v for k,v in row.items() if k!='solve_wall_seconds'}
  if strip(a['selected'])!=strip(b['selected']) or [strip(x) for x in a['trials']]!=[strip(x) for x in b['trials']]:raise ValueError('Integrated numerical result changed')
  if not a['selected'] or b.get('forecast_recovery') or b['selected']['minimum_constraint'] < -1e-5:raise ValueError('Invalid paired integrated plan')
  benchmarks.append(dict(step=i,baseline=_hugsim_expand(json.loads((folder/('baseline_step_%03d_summary.json'%i)).read_text())),candidate=_hugsim_expand(json.loads((folder/('candidate_step_%03d_summary.json'%i)).read_text())),exact_plan_equality=True))
 source=_hugsim_expand(json.loads((SCENE_BUNDLE/'source_manifest.json').read_text()));arm=_hugsim_expand(json.loads((DRIVER/'arm.json').read_text()));scene=_hugsim_expand(json.loads(Path(arm['scene_manifest']).read_text()));actual=_hugsim_expand(json.loads(Path(scene['source_report']).read_text()))
 if actual['checkpoint']!=source['source_checkpoint'] or actual['scene']!=551 or actual['frames']!=199 or actual['start_timestep']!=0:raise ValueError('Native score and scene source mismatch')
 # Camera-Z and alpha definition must remain the reviewed renderer contract.
 spec=importlib.util.find_spec('gsplat');renderer=Path(next(iter(spec.submodule_search_locations)))/'rendering.py'
 if 'render_colors[..., -1:] / render_alphas.clamp(min=1e-10)' not in renderer.read_text():raise ValueError('Expected-depth renderer changed')
 validation=dict(passed=True,tests=sum(r['tests'] for r in rows),runs=rows,replaced_old_release_tests=sorted(replaced),replacement='migration.validate whole-controller/per-module preservation plus this new frozen experiment contract',migration=migration,benchmarks=benchmarks,gpu_executed=False,native_scores_computed=False)
 (DRIVER/'current_cpu_regression.json').write_text(json.dumps(dict(passed=True,tests=validation['tests'],results=rows,replaced_old_release_tests=sorted(replaced),gpu_executed=False),indent=2)+'\n')
 (HERE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');(DRIVER/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 ready=dict(diagnostic_launch_ready=True,cpu_regression_passed=True,performance_candidate_frozen=True,unit_tests=validation['tests'],new_gpu_executed=False,global_repair_verified=False,closed_loop_launch_ready=False,formal_evaluation_admitted=False,native_evaluation_admitted=False,reason='Original-native-vehicle cut-in and unmodified native metric bounds prepared; GPU pending.')
 (HERE/'readiness.json').write_text(json.dumps(ready,indent=2)+'\n');(DRIVER/'readiness.json').write_text(json.dumps(ready,indent=2)+'\n')
 pins={}
 # Preserve actual inherited imports and avoid silently replacing source code.
 for name,digest in _hugsim_expand(json.loads((UPSTREAM/'manifest.json').read_text())).items():
  file=ROOT/name
  if not file.is_file():raise FileNotFoundError(file)
  if sha(file)!=digest:raise ValueError('Upstream frozen dependency changed: '+name)
  pins[name]=digest
 for folder in (HERE,DRIVER,PHASE26/'framework_repo/models',PHASE26/'framework_repo/datasets',PHASE26/'framework_repo/utils'):
  for f in folder.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and f.name!='manifest.json' and f.suffix in ('.py','.json','.md','.sh','.npz','.npy','.cpp','.so','.1'):pins[str(f.relative_to(ROOT))]=sha(f)
 for f in [SCORER,SCORER.parent/'sim_utils.py',SCENE_BUNDLE/'actors.npz',SCENE_BUNDLE/'native_dense_route.npy',SCENE_BUNDLE/'background_sr_world.npy',SCENE_BUNDLE/'road_sr_world.npy',SCENE_BUNDLE/'point_semantics/background_semantics.npz',SCENE_BUNDLE/'coverage_audit.json',renderer]:pins[str(f.relative_to(ROOT))]=sha(f)
 (HERE/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');verify_frozen();print('FROZEN',len(pins),'files; CPU tests',validation['tests'],flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();build() if args.freeze else verify_frozen()
