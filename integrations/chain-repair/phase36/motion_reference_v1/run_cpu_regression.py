
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
import hashlib,json,os,re,subprocess,sys,time
p=Path(__file__).resolve().parent;names=re.findall(r'\$SCENE_DIR/(test_[^" ]+\.py)',(p/'run.sh').read_text());rows=[];started=time.time()
for name in names:
 result=subprocess.run([sys.executable,'-B',str(p/name)],cwd=p,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 match=re.search(r'Ran (\d+) tests?',result.stdout);count=int(match.group(1)) if match else 0
 rows.append(dict(file=name,sha256=hashlib.sha256((p/name).read_bytes()).hexdigest(),passed=result.returncode==0,tests=count,output=result.stdout))
 print(name,result.returncode,count,flush=True)
summary=dict(passed=all(r['passed'] for r in rows),tests=sum(r['tests'] for r in rows),elapsed_seconds=time.time()-started,results=rows,gpu_executed=False)
(p/'current_cpu_regression.json').write_text(json.dumps(summary,indent=2)+'\n');print('TOTAL',summary['tests'],'PASS',summary['passed']);raise SystemExit(0 if summary['passed'] else 1)
