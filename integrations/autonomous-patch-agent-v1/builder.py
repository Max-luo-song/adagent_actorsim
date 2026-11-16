"""Host-owned release generator. Model output never selects paths or commands."""

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

import ast,copy,json,os,shutil,tempfile
pass  # Path is provided by the portable boundary above
from core import AgentError,LOCAL,ROOT,P26,asset,code_pins,confined,digest,identity,read,sha,verify_pins,write
from contracts import guarded_change

PLATFORM='light-execution-horizon-v17-r2'
RELEASE='light_execution_horizon_v17_r2'
HARNESS='patch-v17-validation-r2-v1'
MATRIX='patch_v17_validation_r2_v1'
ROUND='v17_runtime_r2'

def clean_copy(source,target):
 for p in Path(source).rglob('*'):
  if p.is_symlink():raise AgentError('Release tree contains a symlink: '+str(p))
 shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','agent_build.json'))
def replace_tree(folder,mapping):
 for p in Path(folder).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.sh','.md'):
   s=p.read_text()
   for a,b in mapping.items():s=s.replace(a,b)
   p.write_text(s)

def build(round_dir,round_id,parent,proposal,config,smoke=False):
 round_dir=Path(round_dir);rid=identity(round_id);name='light-agent-'+rid.replace('_','-');namespace='light_agent_'+rid
 candidate=ROOT/'integrations'/name;release=P26/namespace
 request=digest(dict(round_id=rid,parent=parent['code_pins'],proposal=proposal,smoke=smoke))
 manifest=round_dir/'bundle.json'
 if manifest.exists():
  b=read(manifest)
  if b['request']!=request:raise AgentError('Round application request changed')
  verify_pins(b['candidate_pins']);return b
 source=Path(parent['code']);verify_pins(parent['code_pins'])
 if candidate.exists():
  receipt=read(candidate/'agent_build.json')
  if receipt['request']!=request:raise AgentError('Existing candidate belongs to another request')
  verify_pins(receipt['code_pins']);changed=receipt['changed_methods']
 else:
  staging=Path(tempfile.mkdtemp(prefix='.'+name+'-',dir=ROOT/'integrations'));staging.rmdir()
  clean_copy(source,staging)
  # Carry the previously verified renderer/SMPL integration independently of
  # the policy parent; never ask the model to regenerate launch infrastructure.
  for file in ('pipeline.py','cpu_check.py','run.sh','runtime.py','smpl_runtime.py','compare_native.py'):
   shutil.copyfile(ROOT/'integrations'/PLATFORM/file,staging/file)
  mapping={source.name:name,parent['release_name']:namespace,PLATFORM:name,RELEASE:namespace}
  replace_tree(staging,mapping)
  before=(staging/'joint_safety.py').read_text();after=before
  for op in proposal['patch_instruction']['operations']:
   if after.count(op['before'])!=1:raise AgentError('Patch context is not unique')
   after=after.replace(op['before'],op['after'])
  changed=[] if smoke else guarded_change(before,after)
  (staging/'joint_safety.py').write_text(after)
  if not smoke:
   target=confined(staging,'tests/test_agent_'+rid+'.py')
   if target.exists():raise AgentError('Causal test would overwrite an existing regression')
   ast.parse(proposal['patch_instruction']['causal_test']);target.write_text(proposal['patch_instruction']['causal_test'])
  for p in staging.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
  compare=staging/'compare_native.py';s=compare.read_text().replace('v17 execution-benefit gate',rid).replace('v13 reference join','previous working champion').replace('execution_horizon_v13_v17_3CAM.mp4','agent_comparison_3CAM.mp4');compare.write_text(s)
  staging.rename(candidate)
  write(candidate/'agent_build.json',dict(request=request,code_pins=code_pins(candidate),changed_methods=changed))
 parent_fixture=round_dir/'parent_fixture'
 if not parent_fixture.exists():
  staging=round_dir/'parent_fixture_building'
  if staging.exists():raise AgentError('Interrupted parent fixture build retained; create a fresh round')
  clean_copy(source,staging);staging.rename(parent_fixture)
 extras=round_dir/'registered_tests';extras.mkdir(exist_ok=True)
 previous=P26/'automated_patch_workflow_v2/rounds'/ROUND
 for old,new in [('test_execution_horizon.py','test_execution_horizon.py'),('test_smpl_runtime.py','test_smpl_runtime.py')]:
  s=(previous/old).read_text().replace(PLATFORM,name);(extras/new).write_text(s)
 replay=round_dir/'matched_replay.py';s=(previous/'replay.py').read_text().replace(PLATFORM,name).replace(RELEASE,namespace);replay.write_text(s)
 b=dict(request=request,round_id=rid,code=str(candidate),release=str(release),release_name=namespace,candidate_pins=code_pins(candidate),parent_fixture=str(parent_fixture),parent_fixture_pins=code_pins(parent_fixture),new_test=str(candidate/'tests'/('test_agent_'+rid+'.py')),extra_tests=[str(extras/'test_execution_horizon.py'),str(extras/'test_smpl_runtime.py')],replay=str(replay),changed_methods=changed,smoke=smoke)
 write(manifest,b);return b

def attempt(round_dir,bundle,number):
 round_dir=Path(round_dir);rid=bundle['round_id'];suffix=rid+'_a'+str(number).zfill(2)
 root=round_dir/'attempts'/('a'+str(number).zfill(2));root.mkdir(parents=True,exist_ok=True);manifest=root/'bundle.json'
 if manifest.exists():
  b=read(manifest);verify_pins(b['harness_pins']);return b
 name='patch-agent-'+suffix.replace('_','-');outname='patch_agent_'+suffix;harness=ROOT/'integrations'/name;output=P26/outname
 if harness.exists():
  receipt=read(harness/'agent_build.json')
  if receipt['round_id']!=rid:raise AgentError('Existing harness belongs to another round')
  verify_pins(receipt['harness_pins'])
 else:
  staging=Path(tempfile.mkdtemp(prefix='.'+name+'-',dir=ROOT/'integrations'));staging.rmdir();clean_copy(ROOT/'integrations'/HARNESS,staging)
  replace_tree(staging,{PLATFORM:Path(bundle['code']).name,RELEASE:bundle['release_name'],HARNESS:name,MATRIX:outname})
  for p in staging.rglob('*.py'):p.write_text(p.read_text().replace("'v17'",repr(rid)))
  run=staging/'run.sh';s=run.read_text().replace("echo 'GPU lock busy; no matrix episodes started' >&2;exit 1;","echo 'GPU lock busy; no matrix episodes started' >&2;exit 75;");run.write_text(s)
  staging.rename(harness);write(harness/'agent_build.json',dict(round_id=rid,harness_pins=code_pins(harness)))
 result=dict(number=number,root=str(root),harness=str(harness),output=str(output),harness_pins=code_pins(harness));write(manifest,result);return result
