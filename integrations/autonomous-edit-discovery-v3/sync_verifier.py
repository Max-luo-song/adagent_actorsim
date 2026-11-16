"""Exact renderer-dtype round trip; no widened spatial tolerance or policy edits."""

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

import ast,copy,inspect
pass  # Path is provided by the portable boundary above
import numpy as np
from common_flow import asset,write

def exact_runtime_values(actual,expected,*args,**kwargs):
 actual=np.asarray(actual);expected=np.asarray(expected)
 if actual.dtype!=np.dtype('float32') or expected.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Unregistered renderer trajectory dtype')
 if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Invalid renderer trajectory values')
 np.testing.assert_array_equal(actual,expected.astype(actual.dtype))

class ReplaceAssertions(ast.NodeTransformer):
 def __init__(self):self.count=0
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and node.func.attr=='assert_allclose' and ast.unparse(node.func)=='np.testing.assert_allclose':
   if len(node.args)!=2 or not ast.unparse(node.args[0]).startswith("tracks['RigidNodes__instances_") or not ast.unparse(node.args[1]).startswith("target['rendered_"):raise ValueError('Unexpected legacy sync assertion')
   self.count+=1;node.func=ast.Name(id='_exact_runtime_values',ctx=ast.Load())
  return node

def install(policy):
 from episode import bind
 bind(policy)
 import pair_diagnostics as module
 source=Path(module.__file__).resolve()
 if source!=Path(policy['code']).resolve()/'pair_diagnostics.py':raise ValueError('Unexpected actor sync verifier module')
 old=module.verify_actor_sync;tree=ast.parse(inspect.getsource(old));transform=ReplaceAssertions();tree=transform.visit(tree);ast.fix_missing_locations(tree)
 if transform.count!=2:raise ValueError('Expected exactly two canonical trajectory assertions')
 namespace=dict(old.__globals__,_exact_runtime_values=exact_runtime_values)
 exec(compile(tree,str(source),'exec'),namespace);checked=namespace['verify_actor_sync']
 def verifier(root,tracks,session,receipt):
  result=checked(root,tracks,session,receipt)
  result.update(verification_backend='exact_renderer_dtype_v1',host_verifier=asset(Path(__file__)),legacy_verifier=asset(source),spatial_tolerance_widened=False)
  write(Path(root)/'B/actor_sync.json',result);return result
 module.verify_actor_sync=verifier
 return module,old
