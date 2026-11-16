"""Reject obvious task-specific policy branches; held-out validation is still required."""
import ast
from common_flow import read

def validate(parent,proposal,task):
 source=(__import__('pathlib').Path(parent['code'])/'joint_safety.py').read_text();after=source
 for op in proposal['patch_instruction']['operations']:after=after.replace(op['before'],op['after'])
 a,b=ast.parse(source),ast.parse(after)
 old_strings={n.value for n in ast.walk(a) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
 strings={n.value for n in ast.walk(b) if isinstance(n,ast.Constant) and isinstance(n.value,str)}-old_strings
 tokens={task['id'],'scene551','waymo_551','native:RigidNodes:3','o24FGKGBO7aPlRVYkDyU7A','neural_track_id','track_id','annotation_id','scene_id','scene_token','left','right'}
 if any(any(t in value for t in tokens) for value in strings):raise ValueError('New policy contains a task/actor/direction-specific literal; repair the general mechanism')
 old_ints={n.value for n in ast.walk(a) if isinstance(n,ast.Constant) and type(n.value) is int}
 new_ints={n.value for n in ast.walk(b) if isinstance(n,ast.Constant) and type(n.value) is int}-old_ints
 if new_ints & {551,552,172,621}:raise ValueError('New scene-ID constant in policy')
 old_calls={ast.dump(n) for n in ast.walk(a) if isinstance(n,(ast.Call,ast.Import,ast.ImportFrom))}
 for n in ast.walk(b):
  if isinstance(n,(ast.Import,ast.ImportFrom)) and ast.dump(n) not in old_calls:raise ValueError('New runtime import inside policy is outside the registered repair scope')
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'open','eval','exec','__import__','compile'} and ast.dump(n) not in old_calls:raise ValueError('Policy may not access experiment files or execute dynamic code')
 return dict(passed=True,scope='static task-specific-branch screening only; does not prove generalization')
