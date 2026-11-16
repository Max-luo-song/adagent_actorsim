
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

import copy,json,os,ssl,tempfile,unittest,urllib.error
pass  # Path is provided by the portable boundary above
from unittest.mock import patch,MagicMock
from helpers import config,proposal,LOCAL,metrics
from core import AgentError,asset,code_pins,digest,read
from contracts import decision,normalize_response
from governance import validate_plan,version_aliases
from provider import HTTPProvider,TransientAPI
import builder
class VersionTransportTests(unittest.TestCase):
 def evidence(self):
  return dict(items={'x':{}},iteration_context={'frontier_ids':['v15','v17'],'required_base':'v15','protected_target_hdscore':.265,'version_aliases':{'seed_v17_r2':'v17','v15':'v15','v17':'v17','v16':'v16','v13':'v13'},'discriminating_evidence_ids':[]},base_pool={k:{} for k in ['v13','v15','v16','v17']})
 def plan(self):return dict(base_id='v15',strategy='minimal_repair',why_this_base='preserve measured gain',preserve='highest score',fix_only='lateral behavior',compared_versions=['v13','v15','v16','v17','seed_v17_r2'],protected_score=.265,rollback_evidence_ids=[])
 def test_alias_and_historical_comparison_normalize_without_changing_patch(self):
  e=self.evidence();p=proposal(e);p['iteration_plan']=self.plan();before=copy.deepcopy(p['patch_instruction']);n,a=normalize_response([p],e);self.assertEqual(n['iteration_plan']['compared_versions'],['v13','v15','v16','v17']);self.assertEqual(n['patch_instruction'],before);self.assertTrue(a);decision(n,e)
 def test_singleton_does_not_bypass_missing_fields_or_merge_multiple_decisions(self):
  e=self.evidence()
  for value in ([{}],[{},{}]):
   with self.assertRaises(AgentError):decision(value,e)
 def test_exact_unknown_and_missing_feedback(self):
  e=self.evidence();p=self.plan();p['compared_versions']=['v17','invented']
  with self.assertRaises(AgentError) as error:validate_plan(p,e)
  self.assertIn("missing_required=['v15']",str(error.exception));self.assertIn("unknown_ids=['invented']",str(error.exception))
 def test_readonly_reference_cannot_become_policy_base(self):
  e=self.evidence();p=self.plan();p['base_id']='v16'
  with self.assertRaises(AgentError):validate_plan(p,e)
 def test_real_controller_causal_runner_bootstraps_planning_guard_without_fixture_side_effect(self):
  import subprocess,sys
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);test=folder/'bootstrap.py';test.write_text('import unittest\nfrom joint_safety import JointReference\nfrom planning_guard import sweep_allowance\nclass C(unittest.TestCase):\n def test_import(self):self.assertTrue(callable(sweep_allowance));self.assertTrue(hasattr(JointReference,"rollout"))\n')
   target=LOCAL.parent/'light-tracking-curvature-v15-r2';output=folder/'result.json';r=subprocess.run([sys.executable,'-B',str(LOCAL/'test_runner.py'),'--code',str(target),'--test',str(test),'--output',str(output)],capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertTrue(read(output)['passed']);self.assertEqual(read(output)['controller'],asset(target/'joint_safety.py'))
 def test_seed_alias_requires_same_code_identity(self):
  pool={'v17':dict(version='v17',code='/known/c',code_pins={'a':'1'})};champ=dict(version='seed_v17_r2',code='/known/c',code_pins={'a':'1'});self.assertEqual(version_aliases(pool,champ)['seed_v17_r2'],'v17');champ['code_pins']={'a':'other'};self.assertNotIn('seed_v17_r2',version_aliases(pool,champ))
 def test_tls_preflight_failure_sends_no_prompt_key_or_token_reservation(self):
  opener=MagicMock();opener.open.side_effect=urllib.error.URLError(ssl.SSLEOFError('EOF'));charged=[]
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PATCH_API_BASE_URL':'https://example.invalid/v1','PATCH_MODEL':'kimi-k3','PATCH_API_KEY':'SECRET'}),patch('urllib.request.build_opener',return_value=opener):
   path=Path(d)/'call.json'
   with self.assertRaises(TransientAPI):HTTPProvider(config()['api']).complete([{'role':'user','content':'PRIVATE PROMPT'}],{},charged.append,path)
   req=opener.open.call_args.args[0];self.assertEqual(req.get_method(),'HEAD');self.assertIsNone(req.data);self.assertNotIn('Authorization',req.headers);self.assertEqual(charged,[]);meta=read(path)['metadata'];self.assertEqual(meta['reserved_tokens'],0);self.assertFalse(meta['external_API_called'])
 def test_head_405_still_allows_post_with_normal_accounting(self):
  response=MagicMock();response.headers={'Content-Type':'application/json'};response.read.return_value=b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}],"usage":{"total_tokens":8}}';wrapper=MagicMock();wrapper.__enter__.return_value=response;opener=MagicMock();opener.open.side_effect=[urllib.error.HTTPError('https://example.invalid',405,'method',{},None),wrapper];charged=[]
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'PATCH_API_BASE_URL':'https://example.invalid/v1','PATCH_MODEL':'kimi-k3','PATCH_API_KEY':'SECRET'}),patch('urllib.request.build_opener',return_value=opener):
   value,meta=HTTPProvider(config()['api']).complete([],{},charged.append,Path(d)/'call.json');self.assertEqual(value,{});self.assertEqual(meta['usage_tokens'],8);self.assertEqual(len(charged),1);self.assertEqual(opener.open.call_args.args[0].get_method(),'POST')
 def test_real_v5_proposals_pass_version_gate_but_scope_guard_remains(self):
  base=LOCAL.parents[1]/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/autonomous_patch_agent_v5/campaigns/patch_auto_v5/rounds';count=0;scope_rejections=0
  for folder in sorted(base.iterdir()):
   for path in sorted((folder/'model_calls').glob('request_*.json')):
    if '.request.' in path.name or '.progress.' in path.name:continue
    data=read(path);content=data.get('content')
    if not isinstance(content,str):continue
    try:p=_hugsim_expand(json.loads(content))
    except ValueError:continue
    raw=p[0] if isinstance(p,list) and len(p)==1 else p
    if not isinstance(raw,dict) or raw.get('decision')!='patch':continue
    record=read(path.with_name(path.stem+'.request.json'));e=_hugsim_expand(json.loads(record['messages'][-1]['content']))['evidence'];self.assertEqual(raw['evidence_sha256'],digest(e))
    e['iteration_context'].update(self.evidence()['iteration_context']);e['iteration_context']['protected_target_hdscore']=raw['iteration_plan']['protected_score'];e['base_pool']['v16']={'profile':'reference_only','eligible':False};n,a=normalize_response(p,e);n['evidence_sha256']=digest(e);decision(n,e);count+=1
    source=Path(e['items']['source']['asset']['path']).parent;parent=dict(code=str(source),code_pins=code_pins(source),profile='v15_joint')
    try:builder.preview(parent,n,config())
    except AgentError as error:
     self.assertTrue('Protected code changed' in str(error) or 'Patch syntax invalid' in str(error));scope_rejections+=1
  self.assertGreaterEqual(count,9);self.assertGreaterEqual(scope_rejections,1)
if __name__=='__main__':unittest.main()
