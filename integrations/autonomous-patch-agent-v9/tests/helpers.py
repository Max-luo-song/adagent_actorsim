
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

import copy,json,sys,tempfile,time
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL))
from core import asset,code_pins,digest,read,write

def config():
 c=read(LOCAL/'config.example.json');c['generalization_fixture_bypass']=True;c['evidence_roots']=[];c['context']['history_rounds']=4
 c['budgets'].update(max_rounds=2,poll_seconds=.02,retry_delay_seconds=.02,cpu_stage_seconds=10,gpu_stage_seconds=10,wall_seconds=60,max_api_tokens=2000000)
 return c

def metric(hd=.26):return dict(score=dict(hdscore=hd,pdms=1.,rc=hd),status='completed',controls=188,observations=189,collision=False,map_departure=False,unresolved_stop=False,progress_m=100.,stopped_s=0.,final_abs_lateral_m=1.,peak_abs_lateral_m=2.)
def metrics(hd=.26):return dict(original551=metric(.3),cutin551=metric(hd))
def proposal(evidence,kind='patch'):
 return dict(schema='autonomous_patch_decision_v1',evidence_sha256=digest(evidence),decision=kind,failure_reason=dict(observations=[dict(claim='Measured fixture observation',evidence_ids=[next(iter(evidence['items']))])],mechanism='Fixture mechanism, not a real driving diagnosis',hypotheses=['Unconfirmed until validation'],unresolved=[]),edit_instruction=dict(scenario_contract='preserve',cases=['original551','cutin551']),patch_instruction=dict(summary='Fixture proposal',rationale='Exercise orchestration only',operations=[dict(path='joint_safety.py',op='replace_exact',before='return 1',after='return 2')] if kind=='patch' else [],causal_test='import unittest\nclass C(unittest.TestCase):\n def test_behavior(self):self.assertEqual(1,1)\n' if kind=='patch' else ''),diagnostics=[])

def reply(messages,n):return proposal(_hugsim_expand(json.loads(messages[-1]['content']))['evidence'])
class FakeAdapter:
 simulation_only=True
 def __init__(self,root,outcomes=None,fail_cpu=False,slow=False,busy_once=False):self.root=Path(root);self.outcomes=outcomes or [.262,.261];self.fail_cpu=fail_cpu;self.slow=slow;self.busy_once=busy_once
 def bootstrap(self,c):
  code=self.root/'seed';code.mkdir();(code/'joint_safety.py').write_text('class JointReference:\n def retiming_cost(self):return 1\n');(code/'run.sh').write_text('#!/bin/sh\nexit 0\n')
  matrix=self.root/'seed_matrix';matrix.mkdir()
  return dict(version='fixture_seed',code=str(code),release=str(self.root/'release'),release_name='fixture_release',matrix=str(matrix),code_pins=code_pins(code),metrics=metrics(),reference_metrics=metrics())
 def build(self,rd,rid,parent,p,c):
  code=Path(rd)/'candidate';code.mkdir();(code/'joint_safety.py').write_text('class JointReference:\n def retiming_cost(self):return 2\n');(code/'run.sh').write_text('#!/bin/sh\nexit 0\n')
  return dict(code=str(code),release=str(code/'release'),release_name=rid,candidate_pins=code_pins(code),round_id=rid)
 def attempt(self,rd,b,number):
  root=Path(rd)/('attempt_'+str(number));root.mkdir(exist_ok=True);return dict(root=str(root),output=str(root),number=number,harness_pins={})
 def stages(self,rd,b,a,c):
  proof=Path(rd)/'proof.json';ev=Path(a['output'])/'evaluation.json';counter=Path(rd)/'dispatch_count';busy=Path(rd)/'busy_once'
  cmd="from pathlib import Path;import json,time;p=Path("+repr(str(counter))+");p.write_text(str(int(p.read_text())+1) if p.exists() else '1');time.sleep("+('1' if self.slow else '0')+");Path("+repr(str(proof))+").write_text('{\"passed\": true}')"
  if self.fail_cpu:cmd='raise SystemExit(1)'
  gpu="from pathlib import Path;p=Path("+repr(str(ev))+");p.write_text('{\"status\":\"fixture\",\"missing_cases\":[\"original172\"]}')"
  if self.busy_once:gpu="from pathlib import Path;import sys;p=Path("+repr(str(busy))+");first=not p.exists();p.write_text('1');sys.exit(75) if first else None;"+gpu
  def stage(name,command,outputs,gpu=False):return dict(name=name,command=[sys.executable,'-B','-c',command],outputs=list(map(str,outputs)),inputs=b['candidate_pins'],deps=[],timeout=8,gpu=gpu)
  return [stage('preflight',cmd,[proof]),stage('harness_cpu', [sys.executable] if False else "pass",[]),stage('gpu',gpu,[ev],True)]
 def validate_stage(self,*args):pass
 def results(self,b,a):
  n=int(b['round_id'].rsplit('r',1)[-1])-1;x=metrics(self.outcomes[min(n,len(self.outcomes)-1)])
  return x,read(Path(a['output'])/'evaluation.json')
 def classify_failure(self,stage,r):
  if r.get('returncode')==75:return 'gpu_busy'
  if r['status'] in ('timeout','interrupted','cancelled'):return 'interrupted'
  return 'infrastructure' if stage['gpu'] else 'offline_rejection'
