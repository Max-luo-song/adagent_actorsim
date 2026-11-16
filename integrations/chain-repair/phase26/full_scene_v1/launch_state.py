
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
import datetime,json,os,sys
P=Path(__file__).resolve().parent
state=sys.argv[1];run=Path(sys.argv[2]).resolve();data=dict(status=state,output=str(run),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),launcher_pid=os.getppid(),gpu_executed=False)
if state=='finished':
 data['exit_code']=int(sys.argv[3]);data['status']='completed' if data['exit_code']==0 else 'failed_or_not_avoided'
 f=run/'analysis/summary.json'
 if f.exists():
  summary=_hugsim_expand(json.loads(f.read_text()));data.update(gpu_executed=bool(summary.get('observations')),controls=summary.get('controls'),observations=summary.get('observations'),driving_acceptance_passed=summary.get('driving_acceptance_passed'),avoidance_validation_passed=summary.get('avoidance_validation_passed'),bbox_video=(summary.get('bbox_diagnostic') or {}).get('video'))
tmp=P/'launch_status.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(P/'launch_status.json')
