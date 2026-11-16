
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

import sys,subprocess,time,json,re
pass  # Path is provided by the portable boundary above
p=Path(__file__).resolve().parent;root=next(x for x in p.parents if (x/'code/HUGSIM').is_dir());rows=[];tests=0
with (p/'preflight.log').open('w') as log:
 for name in ['cpu_gate.py','check_wiring.py']+[f.name for f in sorted(p.glob('test_*.py'))]:
  began=time.monotonic();r=subprocess.run([sys.executable,'-B',str(p/name)],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write(name+'\n'+r.stdout+'\n');log.flush();print(name,'PASS' if r.returncode==0 else 'FAIL',flush=True);rows.append({'script':name,'returncode':r.returncode,'wall_seconds':time.monotonic()-began})
  if r.returncode:print(r.stdout[-6000:],flush=True);(p/'preflight_status.json').write_text(json.dumps({'passed':False,'rows':rows},indent=2));raise SystemExit(r.returncode)
  found=re.search(r'Ran (\d+) tests?',r.stdout)
  if found:tests+=int(found.group(1))
(p/'preflight_status.json').write_text(json.dumps({'passed':True,'unit_tests':tests,'rows':rows,'gpu_executed':False},indent=2)+'\n');print('All tests',tests,flush=True)
