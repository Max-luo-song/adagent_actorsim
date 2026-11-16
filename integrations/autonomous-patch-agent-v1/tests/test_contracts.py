
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

import copy,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from helpers import LOCAL,config,proposal
from core import AgentError,decode,digest,confined,identity,read
from contracts import decision,guarded_change
class ContractsTests(unittest.TestCase):
 def setUp(self):self.e={'items':{'trace':{'facts':{}}}}
 def test_valid_decision_binds_evidence(self):self.assertEqual(decision(proposal(self.e),self.e)['decision'],'patch')
 def test_unknown_evidence_and_scene_edits_are_rejected(self):
  for change in ('evidence','scene','command','path','hash'):
   p=proposal(self.e)
   if change=='evidence':p['failure_reason']['observations'][0]['evidence_ids']=['invented']
   elif change=='scene':p['edit_instruction']['cases']=['other']
   elif change=='command':p['shell']='rm anything'
   elif change=='path':p['patch_instruction']['operations'][0]['path']='../runtime.py'
   else:p['evidence_sha256']='0'*64
   with self.assertRaises(AgentError,msg=change):decision(p,self.e)
 def test_stop_and_diagnostics_cannot_apply_code(self):
  p=proposal(self.e,'stop');decision(p,self.e);p['patch_instruction']['operations']=proposal(self.e)['patch_instruction']['operations']
  with self.assertRaises(AgentError):decision(p,self.e)
  p=proposal(self.e,'diagnose');p['diagnostics']=[dict(kind='shell',item_id='trace',start=0,end=1)]
  with self.assertRaises(AgentError):decision(p,self.e)
 def test_duplicate_or_nonfinite_json_is_rejected(self):
  for s in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
   with self.assertRaises((AgentError,ValueError)):decode(s)
 def test_paths_and_identifiers_cannot_escape(self):
  for p in ('../x','/tmp/x'):
   with self.assertRaises(AgentError):confined(Path('/tmp/root'),p)
  for name in ('../x','a;whoami','$(id)',''):
   with self.assertRaises(AgentError):identity(name)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'dir').symlink_to('/tmp',target_is_directory=True)
   with self.assertRaises(AgentError):confined(p,'dir/x')
 def test_only_registered_method_body_can_change(self):
  path=LOCAL.parent/'light-execution-horizon-v17-r2/joint_safety.py';old=path.read_text();new=old.replace('.2*np.mean(controls[:,0]**2)+.1*np.mean(controls[:,1]**2)-6*','.21*np.mean(controls[:,0]**2)+.1*np.mean(controls[:,1]**2)-6*')
  # Replace only in the retiming helper, leaving GeometryReference.process intact.
  before,tail=old.split('    def retiming_cost(',1);new=before+'    def retiming_cost('+tail.replace('.2*np.mean(controls[:,0]**2)','.21*np.mean(controls[:,0]**2)',1)
  self.assertEqual(guarded_change(old,new),['JointReference.retiming_cost'])
  for bad in (old.replace('minimum>=.05','minimum>=-.05'),old.replace("'tracking_reserve_m']","'tracking_reserve_m']+1",1),old.replace("tracking_reserve_m=.25","tracking_reserve_m=0.")):
   with self.assertRaises(AgentError):guarded_change(old,bad)
 def test_io_and_planner_mutation_are_rejected(self):
  s='class JointReference:\n def retiming_cost(self,x):\n  return x\n'
  for line in ('open("/tmp/x")','self.selected=None','exec("pass")','import os','x.__class__'):
   new=s.replace('  return x','  '+line+'\n  return x')
   with self.assertRaises(AgentError,msg=line):guarded_change(s,new)
if __name__=='__main__':unittest.main()
