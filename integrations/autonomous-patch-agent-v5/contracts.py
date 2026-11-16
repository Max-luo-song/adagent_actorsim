"""Validate model decisions and preserve all registered safety code."""
import ast,copy,re
from core import AgentError,digest,finite
ALLOWED_METHODS={'JointReference.retime_selected','JointReference.retiming_cost','JointReference.execution_benefit'}
DIAGNOSTICS={'trace_window','source_method','matched_replay_summary'}

def require_keys(value,required,optional=()):
 if not isinstance(value,dict):raise AgentError('Expected one JSON object; got '+type(value).__name__+'; arrays and wrappers are not accepted')
 missing=set(required)-set(value);extra=set(value)-set(required)-set(optional)
 if missing or extra:raise AgentError('Missing fields: '+str(sorted(missing))+'; unexpected fields: '+str(sorted(extra)))
def text(value,limit=12000):
 if not isinstance(value,str) or not value.strip() or len(value)>limit:raise AgentError('Invalid/bounded text required')
 return value

def decision(value,evidence):
 require_keys(value,('schema','evidence_sha256','decision','failure_reason','edit_instruction','patch_instruction','diagnostics'),('iteration_plan',))
 if value['schema']!='autonomous_patch_decision_v1' or value['evidence_sha256']!=digest(evidence):raise AgentError('Decision is not bound to supplied evidence')
 if value['decision'] not in ('patch','diagnose','stop'):raise AgentError('Unknown decision')
 f=value['failure_reason'];require_keys(f,('observations','mechanism','hypotheses','unresolved'))
 if not isinstance(f['observations'],list) or not 1<=len(f['observations'])<=20:raise AgentError('Evidence-backed observations required')
 for obs in f['observations']:
  require_keys(obs,('claim','evidence_ids'));text(obs['claim'])
  if not isinstance(obs['evidence_ids'],list) or not obs['evidence_ids'] or any(i not in evidence['items'] for i in obs['evidence_ids']):raise AgentError('Unknown evidence IDs: '+str([i for i in obs['evidence_ids'] if i not in evidence['items']])+'; allowed IDs: '+str(sorted(evidence['items'])))
 text(f['mechanism'])
 for key in ('hypotheses','unresolved'):
  if not isinstance(f[key],list) or len(f[key])>20:raise AgentError('Invalid hypothesis list')
  for line in f[key]:text(line)
 edit=value['edit_instruction'];require_keys(edit,('scenario_contract','cases'))
 if edit!={'scenario_contract':'preserve','cases':['original551','cutin551']}:raise AgentError('Model cannot edit scene or evaluation cases')
 p=value['patch_instruction'];require_keys(p,('summary','rationale','operations','causal_test'))
 text(p['summary']);text(p['rationale'])
 if not isinstance(p['operations'],list) or len(p['operations'])>8:raise AgentError('At most eight exact replacements')
 for op in p['operations']:
  require_keys(op,('path','op','before','after'))
  if op['path']!='joint_safety.py' or op['op']!='replace_exact':raise AgentError('Unregistered code edit')
  text(op['before'],60000);text(op['after'],60000)
 if not isinstance(p['causal_test'],str) or len(p['causal_test'])>60000:raise AgentError('Invalid test source')
 if value['decision']=='patch' and (not p['operations'] or not p['causal_test'].strip()):raise AgentError('Patch needs code edits and a differential causal test')
 if value['decision']!='patch' and (p['operations'] or p['causal_test']):raise AgentError('Diagnostics/stop must not modify code')
 if not isinstance(value['diagnostics'],list) or len(value['diagnostics'])>4:raise AgentError('At most four registered diagnostics')
 for d in value['diagnostics']:
  require_keys(d,('kind','item_id','start','end'))
  if d['kind'] not in DIAGNOSTICS or d['item_id'] not in evidence['items']:raise AgentError('Unregistered diagnostic')
  if any(not isinstance(d[k],int) or isinstance(d[k],bool) for k in ('start','end')) or not 0<=d['start']<=d['end']<=10000 or d['end']-d['start']>60:raise AgentError('Invalid diagnostic range: require integer 0<=start<=end<=10000 and end-start<=60; split requests or use the supplied diagnostic limits')
 if value['decision']=='diagnose' and not value['diagnostics']:raise AgentError('Diagnostics decision needs registered requests')
 if value['decision']=='patch' and 'iteration_context' in evidence:
  from governance import validate_plan
  validate_plan(value.get('iteration_plan'),evidence)
 return copy.deepcopy(value)

def guarded_change(before,after,allowed_methods=None):
 allowed_methods=ALLOWED_METHODS if allowed_methods is None else set(allowed_methods)
 a=ast.parse(before);b=ast.parse(after)
 class Strip(ast.NodeTransformer):
  def visit_ClassDef(self,node):
   for n in node.body:
    if isinstance(n,ast.FunctionDef) and node.name+'.'+n.name in allowed_methods:n.body=[ast.Pass()]
   return node
 if ast.dump(Strip().visit(ast.parse(before)))!=ast.dump(Strip().visit(ast.parse(after))):raise AgentError('Protected imports, safety gates, policy, memory or class structure changed')
 old={c.name+'.'+m.name:ast.dump(m) for c in a.body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef)}
 changed=[]
 for c in b.body:
  if not isinstance(c,ast.ClassDef):continue
  for m in c.body:
   if not isinstance(m,ast.FunctionDef):continue
   name=c.name+'.'+m.name
   if name not in allowed_methods or old.get(name)==ast.dump(m):continue
   changed.append(name)
   tainted={arg.arg for arg in m.args.args}
   def root_name(node):
    while isinstance(node,(ast.Attribute,ast.Subscript)):node=node.value
    return node.id if isinstance(node,ast.Name) else None
   for node in ast.walk(m):
    if isinstance(node,ast.Assign):
     value=node.value;alias=root_name(value) in tainted
     if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute):alias=value.func.attr!='copy' and root_name(value.func.value) in tainted
     if alias:
      for target in node.targets:
       if isinstance(target,ast.Name):tainted.add(target.id)
   for n in ast.walk(m):
    if isinstance(n,(ast.Global,ast.Nonlocal,ast.Delete)):raise AgentError('Global mutation forbidden')
    if isinstance(n,(ast.Import,ast.ImportFrom)):
     names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module or '']
     if any(x not in ('numpy','math','copy') for x in names):raise AgentError('Unregistered method import')
    if isinstance(n,ast.Attribute) and n.attr.startswith('__'):raise AgentError('Dunder access forbidden')
    if isinstance(n,ast.Call):
     if isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','compile','open','getattr','setattr','delattr','globals','locals','vars','__import__','input','breakpoint'}:raise AgentError('Dynamic code or external IO forbidden')
     if isinstance(n.func,ast.Attribute) and n.func.attr in {'load','save','savez','savez_compressed','loadtxt','savetxt','fromfile','tofile','memmap','ctypes','system','popen'}:raise AgentError('Policy external IO forbidden')
    if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
     targets=n.targets if isinstance(n,ast.Assign) else [n.target]
     for target in targets:
      if any(isinstance(t,(ast.Attribute,ast.Subscript)) and root_name(t) in tainted for t in ast.walk(target)):raise AgentError('Proposal helper cannot mutate planner state or input aliases')
 if not changed:raise AgentError('No registered method changed')
 return changed
