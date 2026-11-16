"""Retarget only scene/frame constants of verified data-export helpers."""
import ast,sys,types
from common_eval import SOURCE,context,sha

def load(name,file):
 scene,raw,checkpoint,frames=context();path=SOURCE/file;tree=ast.parse(path.read_text());changes=[]
 class Retarget(ast.NodeTransformer):
  def visit_Constant(self,node):
   if type(node.value) is int and node.value in (551,199,995):
    value={551:scene,199:frames,995:frames*5}[node.value];changes.append({'line':node.lineno,'before':node.value,'after':value});return ast.copy_location(ast.Constant(value=value),node)
   return node
 tree=ast.fix_missing_locations(Retarget().visit(tree));m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m;exec(compile(tree,str(path),'exec'),m.__dict__)
 if file=='scene_data.py':m.RAW=raw;m.CHECKPOINT=checkpoint
 m.RETARGET_AUDIT={'source_sha256':sha(path),'scene':scene,'frames':frames,'constant_changes':changes,'algorithms_changed':False};return m
