
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

import copy,json,shutil,tempfile,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
from helpers import LOCAL,config,proposal
from core import AgentError,ROOT,P26,asset,code_pins,read,sha
import builder
class BuilderTests(unittest.TestCase):
 def fixture(self,root):
  fake=root/'project';(fake/'integrations').mkdir(parents=True)
  for name in (builder.PLATFORM,builder.HARNESS):shutil.copytree(ROOT/'integrations'/name,fake/'integrations'/name,ignore=shutil.ignore_patterns('__pycache__'))
  p26=fake/'outputs';d=p26/'automated_patch_workflow_v2/rounds'/builder.ROUND;d.mkdir(parents=True)
  original=P26/'automated_patch_workflow_v2/rounds'/builder.ROUND
  for name in ('test_execution_horizon.py','test_smpl_runtime.py','replay.py'):shutil.copyfile(original/name,d/name)
  code=fake/'integrations'/builder.PLATFORM
  parent=dict(code=str(code),release_name=builder.RELEASE,code_pins=code_pins(code))
  evidence={'items':{'source':{'text':'fixture'}}};p=proposal(evidence)
  source=(code/'joint_safety.py').read_text();_,tail=source.split('    def retiming_cost(',1);before='    def retiming_cost('+tail.split('    def execution_benefit(',1)[0]
  p['patch_instruction']['operations'][0].update(before=before,after=before.replace('.2*np.mean(controls[:,0]**2)','.21*np.mean(controls[:,0]**2)'))
  rd=root/'round';rd.mkdir();return fake,p26,parent,p,rd
 def test_real_template_generation_and_reuse_preserve_parent_and_guards(self):
  with tempfile.TemporaryDirectory() as d:
   fake,p26,parent,p,rd=self.fixture(Path(d));before=sha(Path(parent['code'])/'joint_safety.py')
   with patch.object(builder,'ROOT',fake),patch.object(builder,'P26',p26):
    b=builder.build(rd,'unit_r001',parent,p,config());again=builder.build(rd,'unit_r001',parent,p,config());a=builder.attempt(rd,b,1)
   self.assertEqual(b,again);self.assertEqual(sha(Path(parent['code'])/'joint_safety.py'),before);self.assertEqual(b['changed_methods'],['JointReference.retiming_cost'])
   self.assertIn(Path(b['code']).name,(Path(a['harness'])/'pipeline.py').read_text());self.assertIn("BASE_MATRIX=P26/'patch_v13_validation_r2_v1'",(Path(a['harness'])/'pipeline.py').read_text());self.assertIn('exit 75',(Path(a['harness'])/'run.sh').read_text())
   self.assertTrue((Path(b['code'])/'smpl_runtime.py').exists());self.assertTrue(Path(b['new_test']).exists())
   target=Path(b['code'])/'joint_safety.py';target.write_text(target.read_text()+'\n#tamper\n')
   with patch.object(builder,'ROOT',fake),patch.object(builder,'P26',p26),self.assertRaises(AgentError):builder.build(rd,'unit_r001',parent,p,config())
 def test_nonunique_patch_context_never_publishes_candidate(self):
  with tempfile.TemporaryDirectory() as d:
   fake,p26,parent,p,rd=self.fixture(Path(d));p['patch_instruction']['operations'][0].update(before='return',after='return')
   with patch.object(builder,'ROOT',fake),patch.object(builder,'P26',p26),self.assertRaises(AgentError):builder.build(rd,'unit_r001',parent,p,config())
   self.assertFalse((fake/'integrations/light-agent-unit-r001').exists())
 def test_retry_attempts_have_separate_outputs_and_same_candidate(self):
  with tempfile.TemporaryDirectory() as d:
   fake,p26,parent,p,rd=self.fixture(Path(d))
   with patch.object(builder,'ROOT',fake),patch.object(builder,'P26',p26):
    b=builder.build(rd,'unit_r001',parent,p,config());a=builder.attempt(rd,b,1);z=builder.attempt(rd,b,2)
   self.assertNotEqual(a['output'],z['output']);self.assertNotEqual(a['harness'],z['harness']);self.assertEqual(a['number'],1);self.assertEqual(z['number'],2)
if __name__=='__main__':unittest.main()
