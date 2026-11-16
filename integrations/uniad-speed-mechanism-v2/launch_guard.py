"""Serialize probes and preserve demonstrably stopped partial attempts."""

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

import contextlib,datetime,fcntl,json,os,uuid
pass  # Path is provided by the portable boundary above

@contextlib.contextmanager
def experiment_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.probe.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Mechanism probe already running; existing attempt left untouched')
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)

def archive_stopped_partial(root):
    root=Path(root);worker=root/'worker';log=root/'worker.log';state=root/'worker_process.json'
    if not log.exists() and not worker.exists() and not state.exists():return None
    if (root/'result.json').exists():raise RuntimeError('Existing result retained; inspect its validation instead of overwriting it')
    meta=_hugsim_expand(json.loads(state.read_text())) if state.exists() else {}
    text=log.read_text(errors='replace') if log.exists() else ''
    failed_import=(not (worker/'ready.json').exists() and not list(worker.glob('done_*.json')) and 'Traceback (most recent call last)' in text and ('ImportError:' in text or 'ModuleNotFoundError:' in text))
    closed=meta.get('status')=='closed' and (meta.get('returncode') is not None or meta.get('pid') is None)
    if not (failed_import or closed):raise RuntimeError('Partial probe liveness unknown; refusing to move its files; inspect worker.log and worker_process.json')
    folder=root/'failed_attempts'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    for file in (worker,log,state):
        if file.exists():file.rename(folder/file.name)
    if (root/'status.json').exists():(folder/'previous_status.json').write_bytes((root/'status.json').read_bytes())
    print('Stopped partial probe preserved:',folder,'; restarting all cases with fresh model history',flush=True)
    return folder
