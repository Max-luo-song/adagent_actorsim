"""Small durable primitives; no simulation or API imports."""

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

import contextlib,datetime,fcntl,hashlib,json,math,os,re,tempfile,time
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent
ROOT=LOCAL.parents[1]
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
class AgentError(Exception):pass
class Deferred(AgentError):pass
class Rejected(AgentError):pass
class BudgetExceeded(AgentError):pass

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def decode(text):
 def bad(x):raise AgentError('Nonfinite JSON is forbidden')
 def pairs(items):
  result={}
  for k,v in items:
   if k in result:raise AgentError('Duplicate JSON key: '+k)
   result[k]=v
  return result
 return _hugsim_expand(json.loads(text,parse_constant=bad,object_pairs_hook=pairs))
def read(path):return decode(Path(path).read_text())
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 payload=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
  os.replace(name,path)
 finally:
  if os.path.exists(name):os.unlink(name)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def asset(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def verify(a):
 if sha(a['path'])!=a['sha256']:raise AgentError('Artifact changed: '+a['path'])
def pins(folder):
 folder=Path(folder)
 files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json','.txt') and '__pycache__' not in p.parts and 'tests' not in p.parts)
 return {str(p.resolve()):sha(p) for p in files}
def code_pins(folder):return {str(p.resolve()):sha(p) for p in sorted([*Path(folder).rglob('*.py'),Path(folder)/'run.sh'])}
def verify_pins(data):
 for p,h in data.items():
  if sha(p)!=h:raise AgentError('Frozen source changed: '+p)
def identity(value):
 if not isinstance(value,str) or not re.fullmatch('[a-z][a-z0-9_]{0,63}',value):raise AgentError('Invalid identifier')
 return value
def confined(root,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts or not p.parts:raise AgentError('Path escapes candidate')
 target=Path(root)/p
 if any(q.is_symlink() for q in [target,*target.parents] if q!=Path(root).parent):raise AgentError('Symlink path forbidden')
 if not target.resolve().is_relative_to(Path(root).resolve()):raise AgentError('Path escapes candidate')
 return target
@contextlib.contextmanager
def lock(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise Deferred('Another coordinator owns this campaign')
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def event(root,kind,**fields):
 p=Path(root)/'events.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(dict(time=utc(),kind=kind,**fields),ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def process_identity(pid):
 try:
  fields=Path('/proc')/str(pid)/'stat';text=fields.read_text();tail=text[text.rfind(')')+2:].split()
  if tail[0]=='Z':return None
  return tail[19]
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def remaining(state,config):
 if state['started_epoch'] is None:return config['budgets']['wall_seconds']
 value=config['budgets']['wall_seconds']-(time.time()-state['started_epoch'])
 if value<=0:raise BudgetExceeded('Campaign wall budget exhausted')
 return value
