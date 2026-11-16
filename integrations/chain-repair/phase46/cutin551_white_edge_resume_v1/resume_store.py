"""Crash-safe commits and checked deterministic-prefix replay (no tensor snapshot)."""

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

import copy,hashlib,json,os,pickle,uuid
pass  # Path is provided by the portable boundary above
import numpy as np
SCHEMA='checked_mpc_prefix_resume_v1'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def atomic_bytes(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(str(path.parent),os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic_json(path,data):atomic_bytes(path,(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def save_pickle(path,value):atomic_bytes(path,pickle.dumps(value,protocol=4))
def canonical(value,run,scene):
 """Only execution-directory and generation identities may change.

 Numerical arrays, RGB/depth, references, safety policy, clock, geometry, dtype,
 warm start, schedule and controls are exact. No rounded numerical hash.
 """
 if isinstance(value,np.ndarray):
  if value.dtype.hasobject:return ('object_array',value.shape,canonical(value.tolist(),run,scene))
  a=np.ascontiguousarray(value);return ('ndarray',a.dtype.str,a.shape,hashlib.sha256(a.tobytes()).hexdigest())
 if isinstance(value,np.generic):return canonical(value.item(),run,scene)
 if isinstance(value,dict):return ('dict',tuple((canonical(k,run,scene),canonical(v,run,scene)) for k,v in sorted(value.items(),key=lambda kv:repr(kv[0]))))
 if isinstance(value,(list,tuple)):return (type(value).__name__,tuple(canonical(x,run,scene) for x in value))
 if isinstance(value,(set,frozenset)):return (type(value).__name__,tuple(sorted((canonical(x,run,scene) for x in value),key=repr)))
 if isinstance(value,Path):return ('path',canonical(str(value),run,scene))
 if isinstance(value,str):
  value=value.replace(str(Path(run).resolve()),'<ATTEMPT>')
  if scene:
   value=value.replace(str(scene),'<SCENE_TOKEN>');generation=str(scene).rsplit(':',1)[0]
   if generation and generation!=scene:value=value.replace(generation,'<SCENE_GENERATION>')
  return ('str',value)
 if value is None or isinstance(value,(bool,int)):return (type(value).__name__,value)
 if isinstance(value,float):return ('float',value.hex())
 if hasattr(value,'__dict__'):return ('object',type(value).__module__,type(value).__qualname__,canonical(vars(value),run,scene))
 raise TypeError('Unsupported replay input type: '+str(type(value)))
def require_same(actual,expected,*,run,source,scene,source_scene,label):
 if canonical(actual,run,scene)!=canonical(expected,source,source_scene):raise RuntimeError('Resume fidelity gate failed: '+label+'; refusing archived controls')

def load_commit(attempt,manifest):
 path=Path(attempt)/'resume_commit.json'
 if not path.exists():return None
 doc=_hugsim_expand(json.loads(path.read_text()))
 if doc.get('schema')!=SCHEMA or doc.get('manifest_sha256')!=manifest:raise RuntimeError('Resume commit version/config mismatch: '+str(path))
 n=doc.get('completed_controls')
 if isinstance(n,bool) or not isinstance(n,int) or n<0 or n>100 or len(doc.get('trace',[]))!=n:raise RuntimeError('Malformed resume commit')
 if [r['step'] for r in doc['trace']]!=list(range(n)):raise RuntimeError('Noncontiguous committed controls')
 if hashlib.sha256(json.dumps(doc['trace'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=doc.get('trace_sha256'):raise RuntimeError('Committed trace checksum mismatch')
 for rel,sha in doc['files'].items():
  target=Path(rel)
  if target.is_absolute() or '..' in target.parts or digest(Path(attempt)/target)!=sha:raise RuntimeError('Committed file changed: '+rel)
 return doc

def select_source(root,manifest):
 rows=[]
 for p in Path(root).glob('attempt_*'):
  if p.is_dir():
   doc=load_commit(p,manifest)
   if doc is not None:rows.append((doc['completed_controls'],p.name,p,doc))
 return max(rows,key=lambda r:(r[0],r[1]))[2:] if rows else (None,None)

class ReplayStore:
 def __init__(self,output,manifest,source=None,checkpoint=None):
  self.output=Path(output).resolve();self.manifest=manifest;self.source=Path(source).resolve() if source is not None else None;self.checkpoint=checkpoint;self.prefix=checkpoint['completed_controls'] if checkpoint else 0;self.index=None;self.scene=None;self.source_scene=None;self.calls=[];self.cursor=0;self.files={};self.committed=0
 def folder(self,index=None):return self.output/'uniad_worker'/f'step_{self.index if index is None else index:03d}'
 def archived(self,name):return self.source/'uniad_worker'/f'step_{self.index:03d}'/name
 def load_archived(self,name):
  p=self.archived(name);rel=str(p.relative_to(self.source));expected=self.checkpoint['files'].get(rel)
  if expected is None or digest(p)!=expected:raise RuntimeError('Uncommitted or changed replay input: '+name)
  with p.open('rb') as f:return pickle.load(f)
 def observe(self,index,obs,info):
  if index>=self.prefix:return
  self.index=index;old_obs,old_info=self.load_archived('observation.pkl');self.source_scene=old_info['scene_token']
  require_same((obs,info),(old_obs,old_info),run=self.output,source=self.source,scene=info['scene_token'],source_scene=self.source_scene,label=f'observation/pose at step {index}')
 def enter(self,index,reference,occupancy,info,args,kwargs):
  self.index=index;self.scene=info.get('scene_token');self.calls=[];self.cursor=0
  payload=dict(reference=reference,occupancy=occupancy,info=info,args=args,kwargs=kwargs)
  if index<self.prefix:
   old=self.load_archived('resume_outer_input.pkl');self.source_scene=old['info'].get('scene_token');require_same(payload,old,run=self.output,source=self.source,scene=self.scene,source_scene=self.source_scene,label=f'optimizer outer input {index}')
   self.old_calls=self.load_archived('resume_inner_calls.pkl')
  save_pickle(self.folder()/'resume_outer_input.pkl',payload);save_pickle(self.folder()/'resume_inner_calls.pkl',self.calls)
 def inner(self,original,reference,occupancy,info,*args,**kwargs):
  payload=dict(reference=reference,occupancy=occupancy,info=info,args=args,kwargs=kwargs)
  if self.index<self.prefix:
   if self.cursor>=len(self.old_calls):raise RuntimeError('Resume inner optimizer call count changed')
   old=self.old_calls[self.cursor];require_same(payload,old['input'],run=self.output,source=self.source,scene=self.scene,source_scene=self.source_scene,label=f'MPC input {self.index}/{self.cursor}')
   result=copy.deepcopy(old['result'])
   # Regenerate only opaque generation/directory identities; numeric values stay exact.
   def rebind(v):
    if isinstance(v,str):
     v=v.replace(str(self.source),str(self.output))
     if self.source_scene:
      v=v.replace(str(self.source_scene),str(self.scene));oldgen=str(self.source_scene).rsplit(':',1)[0];newgen=str(self.scene).rsplit(':',1)[0]
      if oldgen!=self.source_scene:v=v.replace(oldgen,newgen)
     return v
    if isinstance(v,dict):return {k:rebind(x) for k,x in v.items()}
    if isinstance(v,list):return [rebind(x) for x in v]
    if isinstance(v,tuple):return tuple(rebind(x) for x in v)
    return v
   result=rebind(result)
  else:result=original(reference,occupancy,info,*args,**kwargs)
  self.calls.append(dict(input=copy.deepcopy(payload),result=copy.deepcopy(result)));self.cursor+=1;save_pickle(self.folder()/'resume_inner_calls.pkl',self.calls);return result
 def exit(self,result):
  if self.index<self.prefix:
   if self.cursor!=len(self.old_calls):raise RuntimeError('Resume skipped an archived optimizer call')
   require_same(result,self.load_archived('resume_outer_result.pkl'),run=self.output,source=self.source,scene=self.scene,source_scene=self.source_scene,label=f'optimizer output {self.index}')
  save_pickle(self.folder()/'resume_outer_result.pkl',result)
 def commit(self,trace):
  n=len(trace)
  if n<=self.committed:return
  if n!=self.committed+1 or self.index!=n-1:raise RuntimeError('Nonsequential resume commit')
  if n<=self.prefix:
   # Paths in the trace are not optimizer inputs; compare actual control/pose evidence.
   keys=('step','timestamp','action','state_before','state_after','control_decision','neural_plan_raw','physics_integration_verified','predicted_road_swept_clearance_m')
   a={k:trace[-1][k] for k in keys};b={k:self.checkpoint['trace'][n-1][k] for k in keys};require_same(a,b,run=self.output,source=self.source,scene=self.scene,source_scene=self.source_scene,label=f'executed action/pose {n-1}')
  names=['observation.pkl','perception.npz','guard_memory_report.json','optimization.json','control_decision.json','resume_outer_input.pkl','resume_inner_calls.pkl','resume_outer_result.pkl']
  for name in names:
   p=self.folder()/name;self.files[str(p.relative_to(self.output))]=digest(p)
  frozen_trace=copy.deepcopy(trace);trace_sha=hashlib.sha256(json.dumps(frozen_trace,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  atomic_json(self.output/'resume_commit.json',dict(schema=SCHEMA,manifest_sha256=self.manifest,completed_controls=n,trace=frozen_trace,trace_sha256=trace_sha,files=self.files,source_attempt=str(self.source) if self.source else None,replayed_prefix_controls=self.prefix))
  self.committed=n
