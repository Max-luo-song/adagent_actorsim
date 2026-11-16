"""Clean only an exact stale worker owned by this experiment under its run lock."""

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

import json,os,signal,time
pass  # Path is provided by the portable boundary above
from resume_store import atomic_json

def identity(pid):
 try:
  p=Path('/proc')/str(pid);stat=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
  return dict(pid=int(pid),start_ticks=stat[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=[x.decode() for x in argv])
 except (FileNotFoundError,ProcessLookupError):return None

def record(worker,pid):
 value=identity(pid)
 if value is None:raise RuntimeError('Worker disappeared before ownership recording')
 atomic_json(Path(worker)/'owned_process.json',value)

def cleanup(root,worker,python):
 root=Path(root).resolve();count=0
 for file in root.glob('attempt_*/uniad_worker/owned_process.json'):
  saved=_hugsim_expand(json.loads(file.read_text()));current=identity(saved['pid'])
  if current!=saved:continue  # PID exited/reused or boot changed: never signal it.
  argv=saved['argv'];job=str(file.parent.resolve())
  if len(argv)<4 or argv[0]!=str(python) or argv[1]!=str(worker) or '--job' not in argv or argv[argv.index('--job')+1]!=job:raise RuntimeError('Owned worker command mismatch; refusing cleanup')
  if not file.resolve().is_relative_to(root):raise RuntimeError('Worker outside resume root')
  os.kill(saved['pid'],signal.SIGTERM);deadline=time.monotonic()+5.
  while identity(saved['pid'])==saved and time.monotonic()<deadline:time.sleep(.05)
  if identity(saved['pid'])==saved:os.kill(saved['pid'],signal.SIGKILL)
  count+=1
 print('Cleaned exact stale workers owned by this experiment:',count,flush=True);return count
