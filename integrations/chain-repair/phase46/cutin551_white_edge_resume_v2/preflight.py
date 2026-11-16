"""Freeze the restart harness while retaining the entire white-edge release."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,ast,json,os,re,subprocess,time
from paths import *
from resume_store import digest
PARENT=HERE.parent/'cutin551_white_edge_resume_v1'
def verify_frozen():
 ready=_hugsim_expand(json.loads((HERE/'readiness.json').read_text()))
 if ready.get('diagnostic_launch_ready') is not True:raise RuntimeError('Resumable release admission incomplete')
 pins=_hugsim_expand(json.loads((HERE/'manifest.json').read_text()))
 for name,expected in pins.items():
  if digest(ROOT/name)!=expected:raise RuntimeError('Frozen resume input changed: '+name)
 print('Resumable white-edge SHA256 verified:',len(pins),'files',flush=True)
def build():
 pins=_hugsim_expand(json.loads((PARENT/'manifest.json').read_text()))
 for name,expected in pins.items():
  if digest(ROOT/name)!=expected:raise RuntimeError('Frozen white-edge parent changed: '+name)
 compat=_hugsim_expand(json.loads((HERE/'compatibility.json').read_text()))
 if compat['source_manifest_sha256']!=digest(PARENT/'manifest.json') or compat['accepted_source_manifests']!=[compat['source_manifest_sha256']]:raise RuntimeError('Legacy resume compatibility identity changed')
 allowed={'preflight.py','run_cutin.py','run_resume.py','resume_store.py','test_resume.py'}
 for file in PARENT.glob('*.py'):
  if file.name not in allowed and file.read_bytes()!=(HERE/file.name).read_bytes():raise RuntimeError('Unexpected experiment behavior change: '+file.name)
 expected=(PARENT/'run_cutin.py').read_text().replace(" from native_scores import score\n try:report['native_scores']=score(out,setup['vehicle'])"," from local_modules import load_release_module\n try:report['native_scores']=load_release_module('native_scores').score(out,setup['vehicle'])")
 if (HERE/'run_cutin.py').read_text()!=expected:raise RuntimeError('Scene/controller changed beyond explicit local score import')
 for file in HERE.rglob('*.py'):
  if '__pycache__' not in file.parts:ast.parse(file.read_text(),filename=str(file))
 py=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');rows=[]
 checks=[(test,[str(py),'-B',str(HERE/test)]) for test in ('test_resume.py','test_process_owner.py','test_real_inputs.py','test_recovery_fix.py','test_cutin.py','validate_imports.py','check_actual_import_order.py')]+[('shell_'+name,['bash','-n',str(HERE/name)]) for name in ('run.sh','start_background.sh')]
 with (HERE/'validation/resume.log').open('w') as log:
  for name,cmd in checks:
   t=time.monotonic();r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600);log.write(name+'\n'+r.stdout+'\n');log.flush();m=re.search(r'Ran (\d+) tests?',r.stdout);rows.append(dict(name=name,passed=r.returncode==0,tests=int(m.group(1)) if m else 0,seconds=time.monotonic()-t));print(name,r.returncode,flush=True)
   if r.returncode:print(r.stdout[-5000:],flush=True);raise RuntimeError('Resume CPU preflight failed: '+name)
 (HERE/'validation.json').write_text(json.dumps(dict(passed=True,checks=rows,tests=sum(r['tests'] for r in rows),frozen_white_edge_controller_unchanged=True,gpu_executed=False,resume_method='checked archived sensor history plus cached MPC; live observations after prefix'),indent=2)+'\n')
 (HERE/'readiness.json').write_text(json.dumps(dict(diagnostic_launch_ready=True,cpu_regression_passed=True,new_gpu_executed=False,global_repair_verified=False,closed_loop_launch_ready=False,resume_method='archived_sensor_prefix_exact_mpc_replay',commits='each completed control step; atomic fsync+rename',input_fidelity='exact except declared scene generation/run directory identities'),indent=2)+'\n')
 pins[str((PARENT/'manifest.json').relative_to(ROOT))]=digest(PARENT/'manifest.json')
 for file in HERE.rglob('*'):
  if file.is_file() and '__pycache__' not in file.parts and 'validation' not in file.relative_to(HERE).parts and file.name!='manifest.json' and (file.suffix in ('.py','.json','.md','.sh','.npz','.npy','.cpp','.so','.1','.pkl') or file.name in ('METADATA','top_level.txt')):pins[str(file.relative_to(ROOT))]=digest(file)
 (HERE/'manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n');verify_frozen()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();build() if a.freeze else verify_frozen()
