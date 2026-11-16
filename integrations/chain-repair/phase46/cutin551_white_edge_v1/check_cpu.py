
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import json,os,re,subprocess,time,sys
from paths import *
rows=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');py=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python'
log=HERE/'validation/cpu.log'
only_failed='--failed' in sys.argv
previous=_hugsim_expand(json.loads((HERE/'validation/cpu.json').read_text()))['rows'] if only_failed else []
failed={r['name'] for r in previous if not r['passed']}
if only_failed:rows=[r for r in previous if r['passed']]
with log.open('a' if only_failed else 'w') as stream:
 for p in sorted(DRIVER.glob('test_*.py')):
  if p.name in ('test_performance_admission.py','test_search_migration.py') or (only_failed and p.name not in failed):continue
  t=time.monotonic();r=subprocess.run([str(py),'-B',str(p)],env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600);m=re.search(r'Ran (\d+) tests?',r.stdout);rows.append(dict(name=p.name,passed=r.returncode==0,tests=int(m.group(1)) if m else 0,seconds=time.monotonic()-t));stream.write(p.name+'\n'+r.stdout+'\n');stream.flush();print(p.name,r.returncode,flush=True)
  if r.returncode:print(r.stdout[-3000:],flush=True)
report=dict(passed=all(r['passed'] for r in rows),tests=sum(r['tests'] for r in rows),rows=rows,gpu_executed=False)
(HERE/'validation/cpu.json').write_text(json.dumps(report,indent=2)+'\n');print('CPU TOTAL',report['tests'],report['passed'],flush=True)
