
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

import argparse,ast,json,os,re,subprocess,sys,time
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent
from pipeline import OUT,pins,write
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);a=ap.parse_args()
for p in LOCAL.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
t=time.monotonic();r=subprocess.run([sys.executable,'-u','-B',str(LOCAL/'tests/test_validation.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
log=a.output/'checks/cpu_tests.log';log.parent.mkdir(parents=True,exist_ok=True);log.write_text(r.stdout);print(r.stdout[-6000:])
shell=subprocess.run(['bash','-n',str(LOCAL/'run.sh')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
m=re.search(r'Ran (\d+) tests',r.stdout);passed=r.returncode==0 and shell.returncode==0
write(a.output/'checks/cpu_tests.json',dict(passed=passed,tests=int(m.group(1)) if m else 0,seconds=time.monotonic()-t,code_sha256=pins(),GPU_executed=False,mocked_scene_and_neural_inference=True,actual_ilqr_and_motion=True))
raise SystemExit(0 if passed else 1)
