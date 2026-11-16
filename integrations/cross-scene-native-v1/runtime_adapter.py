"""Whitelisted infrastructure substitutions in frozen runner AST, never policy AST."""
import ast,copy,importlib.util,sys,types
import numpy as np
from scene_contracts import *

def rewrite(source):
 tree=ast.parse(source);edits=[]
 class Adapt(ast.NodeTransformer):
  def visit_keyword(self,node):
   node=self.generic_visit(node)
   if node.arg in ('scene','nav_command','command') and isinstance(node.value,ast.Constant) and node.value.value in (551,2):
    key='scene' if node.arg=='scene' else 'navigation_command';node.value=ast.parse("plan["+repr(key)+"]",mode='eval').body;edits.append((node.lineno,node.arg))
   return node
  def visit_If(self,node):
   text=ast.unparse(node.test)
   if text=="name == 'B'":
    edits.append((node.lineno,'source_actor_verification'));return ast.parse("report['actor_sync']=verify_source_actors(root,tracks,session)").body[0]
   if text=="report['rigid_actor_count'] != 15":
    node.test=ast.parse("report['rigid_actor_count'] != read(root/'source_metadata.json')['actors']['RigidNodes']['instances']",mode='eval').body;edits.append((node.lineno,'actor_count'));return node
   if "name not in ('A', 'B')" in text:
    node.test=ast.parse("name != 'A' or common is not None or candidate is not None",mode='eval').body;edits.append((node.lineno,'fresh_original_only'))
   return self.generic_visit(node)
  def visit_Assign(self,node):
   if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='make_types':
    node.value=ast.parse('canonical_types(make_types)',mode='eval').body;edits.append((node.lineno,'canonical_smpl'))
   return self.generic_visit(node)
 tree=ast.fix_missing_locations(Adapt().visit(tree))
 names=[n for _,n in edits]
 for name in ('scene','nav_command','command','source_actor_verification','actor_count','fresh_original_only','canonical_smpl'):
  if names.count(name)!=1:raise ValueError('Frozen runtime infrastructure changed: '+name)
 return tree,edits

def canonical_types(make):
 Backend,Session=make()
 path=ROOT/'integrations/light-execution-horizon-v17-r2/smpl_runtime.py';spec=importlib.util.spec_from_file_location('_canonical_scene_smpl',path);support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
 return support.verified_backend(Backend),Session

def verify_source_actors(root,tracks,session):
 from native_car_edit import effective_source
 with np.load(Path(root)/'source_actor_tracks.npz') as packed:expected={k:packed[k].copy() for k in packed.files if k.split('__')[0] in ('RigidNodes','DeformableNodes','SMPLNodes')}
 for group in ('RigidNodes','DeformableNodes','SMPLNodes'):
  key=group+'__instances_quats'
  if key not in expected:continue
  q=expected[key]
  if q.ndim==4:q=q[:,:,0,:]
  t=expected[group+'__instances_trans'];fv=expected[group+'__instances_fv']
  for i in range(t.shape[1]):
   original_t=t[:,i].copy();original_q=q[:,i].copy()
   for frame in session.vehicles.tests:t[frame,i],q[frame,i]=effective_source(original_t,original_q,fv[:,i],frame,session.vehicles.tests)
  expected[key]=q
 if set(tracks)!=set(expected):raise ValueError('Scene actor export schema differs')
 if any(not np.array_equal(tracks[k],expected[k]) for k in tracks):raise ValueError('Scene actor values changed')
 return {'passed':True,'all_original_actor_arrays_identical':True,'source':asset(Path(root)/'source_actor_tracks.npz'),'heldout_interpolation_verified':True}

def load(row,bundle):
 code=Path(row['code']);verify_pins(row['code_pins']);sys.path[:0]=[str(code),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
 import common;common.configure()
 # configure adds snapshot paths: restore tested policy import precedence.
 sys.path[:0]=[str(code),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
 common.BUNDLE=Path(bundle)
 import environment,map_reference
 environment.BUNDLE=Path(bundle)
 from map_adapter import install
 install(map_reference)
 import joint_safety
 if Path(joint_safety.__file__).resolve()!=code/'joint_safety.py':raise ValueError('Wrong controller imported')
 # Preserve __file__ so the exact frozen worker path and imports are selected.
 source=code/'runtime.py';tree,edits=rewrite(source.read_text());module=types.ModuleType('runtime');module.__file__=str(source);module.verify_source_actors=verify_source_actors;module.canonical_types=canonical_types;sys.modules['runtime']=module;exec(compile(tree,str(source),'exec'),module.__dict__);module.BUNDLE=Path(bundle)
 module.TERMINATION=copy.deepcopy(module.TERMINATION) # No limit/metric changes.
 import scoring
 scoring.BUNDLE=Path(bundle)
 # Native helper is lazily imported by score_arm after configure prepends SOURCE.
 common.configure()
 import native_scores
 if Path(native_scores.__file__).resolve()!=common.SOURCE/'native_scores.py':raise ValueError('Wrong native scorer')
 native_scores.SCENE_BUNDLE=Path(bundle)
 return module,scoring,{'runner_source':asset(source),'infrastructure_substitutions':edits,'controller':asset(code/'joint_safety.py'),'native_scorer':asset(common.SCORER),'policy_source_changed':False,'native_score_formula_changed':False}
