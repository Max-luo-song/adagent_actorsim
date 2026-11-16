"""Verify immutable phase44 parent and freeze the white-edge extension only."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,ast,hashlib,json,os,re,subprocess,sys,time
from paths import *
PARENT=CR/'phase44/cutin551_optimized_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_frozen():
 ready=_hugsim_expand(json.loads((HERE/'readiness.json').read_text()))
 if ready.get('diagnostic_launch_ready') is not True:raise RuntimeError('White edge CPU admission incomplete')
 pins=_hugsim_expand(json.loads((HERE/'manifest.json').read_text()))
 for name,digest in pins.items():
  if sha(ROOT/name)!=digest:raise RuntimeError('Frozen input changed: '+name)
 print('White-edge experiment SHA256 verified:',len(pins),'files',flush=True)
def build():
 pins=_hugsim_expand(json.loads((PARENT/'manifest.json').read_text()))
 for name,digest in pins.items():
  if sha(ROOT/name)!=digest:raise RuntimeError('Original phase44 input changed: '+name)
 # All inherited driver source files outside this explicit extension stay equal.
 allowed={'empty_controller.py','context_factory.py','motion_policy.py','run_driver.py','protocol_harness.py','test_empty.py','test_global_policy.py'}
 changes=[]
 for file in (CR/'phase44/framework_driver_v1').glob('*.py'):
  target=DRIVER/file.name
  if file.read_bytes()!=target.read_bytes():
   if file.name not in allowed:raise RuntimeError('Unreviewed source change: '+file.name)
   changes.append(file.name)
 for file in [*DRIVER.glob('*.py'),*HERE.glob('*.py')]:ast.parse(file.read_text(),filename=str(file))
 report=_hugsim_expand(json.loads((HERE/'validation/cpu.json').read_text()))
 if not report['passed']:raise RuntimeError('Inherited CPU tests have failures')
 py=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python';rows=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 # Relevant current boundary tests repeat after the final conservative raster crop.
 checks=[('white_edge',[str(py),'-B',str(DRIVER/'test_white_edge.py')]),('white_edge_runtime',[str(py),'-B',str(DRIVER/'test_white_edge_runtime.py')]),('native_case',[str(py),'-B',str(HERE/'test_cutin.py')]),('launcher_import_order',[str(py),'-B',str(HERE/'validate_imports.py')]),('shell',['bash','-n',str(HERE/'run.sh')])]
 with (HERE/'validation/extension.log').open('w') as stream:
  for name,cmd in checks:
   t=time.monotonic();r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600);stream.write(name+'\n'+r.stdout+'\n');stream.flush();m=re.search(r'Ran (\d+) tests?',r.stdout);rows.append(dict(name=name,passed=r.returncode==0,tests=int(m.group(1)) if m else 0,seconds=time.monotonic()-t));print(name,r.returncode,flush=True)
   if r.returncode:print(r.stdout[-4000:],flush=True);raise RuntimeError('Extension preflight failed: '+name)
 validation=dict(passed=True,inherited_tests=report['tests'],new_runtime_tests=next(r['tests'] for r in rows if r['name']=='white_edge_runtime'),extension_checks=rows,reviewed_source_changes=changes,ranked_search_and_compiled_geometry_preserved=True,gpu_executed=False,avoidance_certified=False)
 (HERE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 ready=dict(diagnostic_launch_ready=True,performance_candidate_frozen=True,cpu_regression_passed=True,global_repair_verified=False,closed_loop_launch_ready=False,new_gpu_executed=False,reason='White edge current-camera extension CPU checked; new closed loop pending. Visual ground-plane estimate is not a certified physical road map.')
 for folder in (HERE,DRIVER):(folder/'readiness.json').write_text(json.dumps(ready,indent=2)+'\n')
 # Pin the parent manifest itself as well as all inherited immutable dependencies.
 pins[str((PARENT/'manifest.json').relative_to(ROOT))]=sha(PARENT/'manifest.json')
 for folder in (HERE,DRIVER):
  for f in folder.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and 'validation' not in f.relative_to(folder).parts and f.name!='manifest.json' and (f.suffix in ('.py','.json','.md','.sh','.npz','.npy','.cpp','.so','.1','.pkl') or f.name in ('METADATA','top_level.txt')):pins[str(f.relative_to(ROOT))]=sha(f)
 (HERE/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');verify_frozen()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();build() if args.freeze else verify_frozen()
