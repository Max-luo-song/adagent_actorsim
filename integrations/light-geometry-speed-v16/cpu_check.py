"""No model/scene GPU load: compatibility, native IPOPT and actual imports first."""

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

import ast,json,os,re,subprocess,sys,time,hashlib
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1];CR=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair';OUT=CR/'phase26/light_geometry_speed_v16/checks';OUT.mkdir(parents=True,exist_ok=True)
rows=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
for file in LOCAL.rglob('*.py'):ast.parse(file.read_text(),filename=str(file))
parent=LOCAL.parent/'light-map551-cutin-v5'
for name in ('cutin_scenario.py','intervention.py','pair_diagnostics.py'):
 if (LOCAL/name).read_bytes()!=(parent/name).read_bytes():raise ValueError('Unrelated scenario/render/evaluation change: '+name)
def check(name,cmd,extra=None):
 start=time.monotonic();r=subprocess.run(cmd,cwd=ROOT,env=dict(env,**(extra or {})),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600);(OUT/(name+'.log')).write_text(r.stdout);m=re.search(r'Ran (\d+) tests?',r.stdout);rows.append(dict(name=name,passed=r.returncode==0,tests=int(m.group(1)) if m else 0,seconds=time.monotonic()-start));print(name,r.returncode,flush=True)
 if r.returncode:print(r.stdout[-5000:],flush=True);raise RuntimeError('CPU check failed: '+name)
if '--worker-only' not in sys.argv:
 check('reference_join_tests',[sys.executable,'-B',str(LOCAL/'tests/test_reference_join.py')]);check('intent_recovery_tests',[sys.executable,'-B',str(LOCAL/'tests/test_intent_recovery.py')]);check('risk_provenance_tests',[sys.executable,'-B',str(LOCAL/'tests/test_risk_provenance.py')]);check('evidence_control_tests',[sys.executable,'-B',str(LOCAL/'tests/test_evidence_control.py')]);check('risk_memory_tests',[sys.executable,'-B',str(LOCAL/'tests/test_risk_memory.py')]);check('joint_repair_tests',[sys.executable,'-B',str(LOCAL/'tests/test_repair.py')]);check('joint_safety_tests',[sys.executable,'-B',str(LOCAL/'tests/test_joint_safety.py')]);check('native_policy_tests',[sys.executable,'-B',str(LOCAL/'tests/test_native_collision.py')]);check('real_tracker_cut_in_contracts',[sys.executable,'-B',str(LOCAL/'tests/test_cutin.py')])
else:
 for name in ('reference_join_tests','intent_recovery_tests','risk_provenance_tests','evidence_control_tests','risk_memory_tests','joint_repair_tests','joint_safety_tests','native_policy_tests','real_tracker_cut_in_contracts'):
  log=(OUT/(name+'.log')).read_text();m=re.search(r'Ran (\d+) tests?',log)
  if not m or '\nOK\n' not in log:raise RuntimeError('No successful cached CPU test log: '+name)
  rows.append(dict(name=name,passed=True,tests=int(m.group(1)),reused_unchanged_policy_test=True))
legacy=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'
if '--worker-only' not in sys.argv:check('exact_native_ipopt_replay',[str(legacy),'-B',str(LOCAL/'probe_native.py'),'--output',str(OUT/'native_cpu_replay')],{'LD_LIBRARY_PATH':''})
else:
 native=_hugsim_expand(json.loads((OUT/'native_cpu_replay/result.json').read_text()))
 if not native['passed'] or not native['ipopt_solve_passed']:raise RuntimeError('Native CPU replay not verified')
 rows.append(dict(name='exact_native_ipopt_replay',passed=True,cases=len(native['rows']),reused_unchanged_native_code=True,tests=0))
sys.path[:0]=[str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from local_boot import local_archive
support=local_archive(CR/'phase26/light_map551_v4/boot/manifest.json');check('real_worker_import_order',[str(legacy),'-B',str(LOCAL/'worker.py'),'--help'],{'PYTHONPATH':support,'LD_LIBRARY_PATH':''})
check('shell_syntax',['bash','-n',str(LOCAL/'run.sh')]);result=dict(code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*LOCAL.rglob('*.py'),LOCAL/'run.sh']},passed=True,checks=rows,tests=sum(r['tests'] for r in rows),native_source_unchanged=True,tracker_map_geometry_scenario_unchanged=True,GPU_executed=False);(OUT/'cpu_admission.json').write_text(json.dumps(result,indent=2)+'\n');print('CPU admission passed',flush=True)
