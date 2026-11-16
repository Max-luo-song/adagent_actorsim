"""Rebuild and verify this diagnostic package after CPU validation."""

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

import sys,subprocess,json,time,hashlib,re
pass  # Path is provided by the portable boundary above
p=Path(__file__).resolve().parent;root=next(x for x in p.parents if (x/'code/HUGSIM').is_dir());rows=[];tests=0
scripts=['cpu_gate.py','check_wiring.py']+[f.name for f in sorted(p.glob('test_*.py'))]+['independent_current_check.py']
with (p/'preflight.log').open('w') as log:
 for name in scripts:
  start=time.monotonic();result=subprocess.run([sys.executable,'-B',str(p/name)],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write(name+'\n'+result.stdout+'\n');log.flush();rows.append({'script':name,'returncode':result.returncode,'wall_seconds':time.monotonic()-start});print(name,'PASS' if result.returncode==0 else 'FAIL',flush=True)
  if result.returncode:print(result.stdout[-5000:],flush=True);raise SystemExit(result.returncode)
  match=re.search(r'Ran (\d+) tests?',result.stdout)
  if match:tests+=int(match.group(1))
 legacy=root/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python';code='import sys;sys.path.insert(0,'+repr(str(p))+');import worker;assert callable(worker.install);print("Legacy worker import ready")';r=subprocess.run([str(legacy),'-B','-c',code],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write('legacy_worker_import\n'+r.stdout);log.flush();print('legacy_worker_import','PASS' if r.returncode==0 else 'FAIL',flush=True)
 if r.returncode:print(r.stdout[-4000:]);raise SystemExit(r.returncode)
 rows.append({'script':'legacy_worker_import','returncode':r.returncode})
checks=['regression_tests','typed_transport_tests','resident_worker_export_tests','continuous_loss_and_recovery_tests','driver_runtime_gate','legacy_worker_import','solver_and_geometry_regression','strategy_speed_policy','independent_speed_geometry','current_frame_reference','road_soft_cost','independent_current_geometry','native_alignment_tests','smoke_scope']
validation={'passed':True,'unit_tests':tests,'required_checks':dict.fromkeys(checks,True),'runs':rows,'diagnostic_controls':30,'simulation_seconds':3.,'gpu_executed':False,'native_evaluation_admitted':False,'formal_evaluation_admitted':False,'controller_byte_identical_to_native_alignment':(p/'empty_controller.py').read_bytes()==(p.parent/'native_alignment_v1/empty_controller.py').read_bytes(),'independent_checks':_hugsim_expand(json.loads((p/'independent_current_checks.json').read_text()))}
if not validation['controller_byte_identical_to_native_alignment']:raise ValueError('Planner unexpectedly changed')
(p/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
pins=_hugsim_expand(json.loads((p.parents[1]/'phase32/current_frame_v1/manifest.json').read_text()))
for f in p.rglob('*'):
 if f.is_file() and f.suffix in ['.py','.json','.sh','.md'] and f.name!='manifest.json':pins[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
(p/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');print('CPU tests:',tests,'frozen pins:',len(pins),flush=True)
r=subprocess.run([sys.executable,'-B',str(p/'launch.py'),'--check'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(r.stdout,flush=True);raise SystemExit(r.returncode)
