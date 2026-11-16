"""Narrow attachment to the frozen live driver; missing evidence fails closed."""

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

import ast,inspect,json
pass  # Path is provided by the portable boundary above
from evidence_packet import build

def attach(perception,stepdir,info,vehicle):
 stepdir=Path(stepdir)
 def read(name):return _hugsim_expand(json.loads((stepdir/name).read_text()))
 packet=build(perception,read('neural_query_origin.json'),read('global_evidence.json'),read('geometry_refinement.json'),info,vehicle)
 (stepdir/'box_observation_contract.json').write_text(json.dumps(packet['box_observation_contract'],indent=2)+'\n')
 return packet

def rewrite(source):
 tree=ast.parse(source);hits={'constructor':0,'attachment':0}
 class Hook(ast.NodeTransformer):
  def visit_Assign(self,node):
   self.generic_visit(node)
   if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='road_reference' and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='JointReference':
    hits['constructor']+=1
    return [node,*ast.parse("assert getattr(road_reference,'supports_box_observation_contract',False), 'Wrong controller: observation contract unsupported'\nroad_reference.require_box_observation_contract=True").body]
   if ast.unparse(node.value)=="np.asarray(perception['trajectory'])":
    hits['attachment']+=1
    return [*ast.parse('perception=attach_box_observation(perception,stepdir,info,spec.vehicle)').body,node]
   return node
 tree=ast.fix_missing_locations(Hook().visit(tree))
 if hits!={'constructor':1,'attachment':1}:raise ValueError('Frozen runtime attachment anchors changed: '+str(hits))
 return tree

def install(runtime):
 if getattr(runtime,'box_contract_installed',False):raise ValueError('Contract already installed')
 tree=rewrite(inspect.getsource(runtime.run_arm));runtime.attach_box_observation=attach
 exec(compile(tree,runtime.__file__,'exec'),runtime.__dict__)
 runtime.box_contract_installed=True
 return dict(raw_model_unchanged=True,only_driver_attachments=['require_contract','attach_aligned_box_evidence'],GPU_executed=False)
