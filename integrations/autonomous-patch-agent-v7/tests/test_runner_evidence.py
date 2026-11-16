
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

import json,subprocess,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from helpers import LOCAL,config
from core import AgentError,asset,read,write
from evidence import diagnostics,report_summary,bounded
class RunnerEvidenceTests(unittest.TestCase):
 def test_differential_runner_records_assertion_failure_then_success(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);test=root/'causal.py';test.write_text('import unittest\nfrom joint_safety import response\nclass C(unittest.TestCase):\n def test_response(self):self.assertEqual(response(),2)\n')
   for name,value,failures in [('parent',1,1),('candidate',2,0)]:
    code=root/name;code.mkdir();(code/'joint_safety.py').write_text('def response():return '+str(value)+'\n');output=root/(name+'.json')
    p=subprocess.run([sys.executable,'-B',str(LOCAL/'test_runner.py'),'--code',str(code),'--test',str(test),'--output',str(output)],capture_output=True,text=True)
    self.assertEqual(p.returncode,0,p.stdout+p.stderr);r=read(output);self.assertEqual(r['failures'],failures);self.assertEqual(r['errors'],0);self.assertEqual(r['controller'],asset(code/'joint_safety.py'))
 def test_reports_keep_failure_facts_and_bound_requested_windows(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'run_report.json';trace=[dict(index=i,motion_state=[0,i,0,3,0],action={'acc_mps2':-.6},road_reference={'mode':'verified_wait','admitted_candidates':0},road_gate={'selected':'brake'}) for i in range(5)]
   write(p,dict(status='failed',error='fixture',trace=trace,observations=[]));s=report_summary(p);self.assertEqual(s['controls'],5);self.assertEqual(s['error'],'fixture')
   e={'items':{'r':{'asset':asset(p),'facts':s}}};n,ids=diagnostics(e,[dict(kind='trace_window',item_id='r',start=1,end=2)],100000);self.assertEqual(len(n['items'][ids[0]]['facts']),2)
   p.write_text('{}')
   with self.assertRaises(AgentError):diagnostics(e,[dict(kind='trace_window',item_id='r',start=0,end=1)],100000)
 def test_context_budget_does_not_silently_drop_required_source(self):
  e={'items':{'source':{'text':'x'*1000},'report':{'facts':{'sampled_frames':[{'x':'a'*100} for _ in range(40)]}}}}
  r=bounded(e,1600);self.assertEqual(r['items']['source']['text'],'x'*1000);self.assertLess(len(json.dumps(r)),1600)
if __name__=='__main__':unittest.main()
