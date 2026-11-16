"""Keep model-authored tests away from private validation data and executors."""

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

import ast,os,sys
pass  # Path is provided by the portable boundary above
from core import AgentError
BAD_ROOTS={'ctypes','subprocess','socket','multiprocessing','importlib','inspect','builtins'}
BAD_NAMES={'eval','exec','compile','globals','locals','vars','__import__','getattr','setattr','delattr'}

def validate_test(source):
 tree=ast.parse(source)
 for node in ast.walk(tree):
  if isinstance(node,(ast.Import,ast.ImportFrom)):
   names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
   if any(x.split('.')[0] in BAD_ROOTS for x in names):raise AgentError('Private validation isolation forbids this causal-test import')
  if isinstance(node,ast.Attribute) and (node.attr.startswith('__') or 'ctypes' in node.attr):raise AgentError('Causal-test introspection is outside the isolated interface')
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in BAD_NAMES:raise AgentError('Dynamic causal-test code is outside the isolated interface')
 return True

def install(paths,released_paths=()):
 roots=[Path(p).resolve() for p in paths];released=[Path(p).resolve() for p in released_paths]
 def protected(value):
  if not isinstance(value,(str,bytes,os.PathLike)):return False
  try:path=Path(os.fsdecode(value)).resolve()
  except (OSError,ValueError):return False
  return any(path==root or path.is_relative_to(root) for root in roots) and not any(path==root or path.is_relative_to(root) for root in released)
 def audit(event,args):
  if event in ('open','os.listdir','os.scandir','os.chdir') and args and protected(args[0]):raise PermissionError('Private validation access blocked')
  if event in ('subprocess.Popen','os.system','os.posix_spawn','os.exec','ctypes.dlopen') or event.startswith('socket.'):raise PermissionError('Isolated causal tests cannot use external executors or network')
 sys.addaudithook(audit)
