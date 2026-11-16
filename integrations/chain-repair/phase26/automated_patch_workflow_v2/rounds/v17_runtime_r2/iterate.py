"""Current-assistant-authored v17 round; gates, fresh paired run, judgment, video."""

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

import sys,json,hashlib,subprocess,os
pass  # Path is provided by the portable boundary above
D=Path(__file__).resolve().parent
R=Path(_hugsim_expand('${HUGSIM_ROOT}'));P=Path(_hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'));C=R/'integrations/light-execution-horizon-v17-r2';H=R/'integrations/patch-v17-validation-r2-v1';OUT=P/'patch_v17_validation_r2_v1';RELEASE=P/'light_execution_horizon_v17_r2'
def read(p):return _hugsim_expand(json.loads(p.read_text()))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p):return dict(path=str(p),sha256=sha(p))
def gate():
 sys.path[:0]=[str(C),str(R/'integrations/light-map551-v4'),str(R/'integrations/light-cutin-diagnostic')]
 from common import configure
 configure();sys.path.insert(0,str(C))
 from smpl_runtime import preflight
 write(D/'smpl_preflight.json',preflight())
 freeze=read(D/'workflow_freeze.json')
 for f,h in freeze['sha256'].items():
  if sha(Path(f))!=h:raise ValueError('Frozen workflow changed: '+f)
 a0=read(D/'apply_receipt.json')['candidate_main']
 if sha(Path(a0['path']))!=a0['sha256']:raise ValueError('Reviewable candidate differs from applied patch')
 e=read(D/'evidence.json')
 for f,h in {**e['source_code'],**e['source_assets'],**e['previous_candidate_code']}.items():
  if sha(Path(f))!=h:raise ValueError('Frozen v13 source changed: '+f)
 a=read(RELEASE/'checks/cpu_admission.json')
 if not a['passed'] or any(sha(Path(f))!=h for f,h in a['code_sha256'].items()):raise ValueError('Candidate CPU admission changed')
 replay=read(RELEASE/'checks/v13_matched_replay.json')
 if not replay['passed'] or len(replay['rows'])!=40:raise ValueError('Matched replay failed/incomplete')
 t=subprocess.run([sys.executable,'-B',str(D/'test_execution_horizon.py')],capture_output=True,text=True,check=False);(D/'execution_horizon_tests.log').write_text(t.stdout+t.stderr)
 if t.returncode:raise ValueError('Execution-horizon regression failed')
 w=subprocess.run([sys.executable,'-B',str(D/'test_workflow_policy.py')],capture_output=True,text=True,check=False);(D/'workflow_policy_tests.log').write_text(w.stdout+w.stderr)
 if w.returncode:raise ValueError('Side-effect acceptance regression failed')
 k=subprocess.run([sys.executable,'-B',str(D/'test_smpl_runtime.py')],capture_output=True,text=True,check=False);(D/'smpl_runtime_tests.log').write_text(k.stdout+k.stderr)
 if k.returncode:raise ValueError('Actual SMPL constructor regression failed')
 write(D/'diagnostic_gate.json',dict(passed=True,existing_regression_tests=a['tests'],additional_tests=19,replay_cases=40,candidate_admission=asset(RELEASE/'checks/cpu_admission.json'),replay=asset(RELEASE/'checks/v13_matched_replay.json'),new_tests=asset(D/'test_execution_horizon.py'),new_tests_log=asset(D/'execution_horizon_tests.log'),GPU_validation_allowed=True))
 return a
def stage(name):return subprocess.run(['bash',str(H/'run.sh'),name],check=True)
def prepare():
 gate();subprocess.run(['bash',str(C/'run.sh'),'prepare'],check=True);subprocess.run(['bash',str(C/'run.sh'),'check'],check=True);stage('prepare');stage('check')
 write(D/'state.json',dict(stage='CPU_passed_GPU_pending',baseline='v13',candidate='v17-r2',GPU_executed=False,promoted=False,global_patch_validated=False))
def path_comparison():
 # A diagnostic nonregression gate for this paired test, not an online lane
 # anchor, score formula change, or a claim that every maneuver must return.
 comparisons={};limit=.25
 for case,arm in [('original551','A'),('cutin551','B')]:
  base=read(P/f'patch_v13_validation_r2_v1/cases/{case}/on/{arm}/run_report.json')
  new=read(OUT/f'cases/{case}/on/{arm}/run_report.json')
  def lateral(report):
   x=[abs(float(t['motion_state'][0])) for t in report['trace']]
   return dict(final_abs_lateral_m=x[-1],peak_abs_lateral_m=max(x))
  b=lateral(base);n=lateral(new)
  passed=n['final_abs_lateral_m']<=b['final_abs_lateral_m']+limit and n['peak_abs_lateral_m']<=b['peak_abs_lateral_m']+limit
  comparisons[case]=dict(baseline_v13=b,candidate_v17=n,passed=passed)
  previous=P/f'patch_v15_validation_r2_v1/cases/{case}/on/{arm}/run_report.json'
  if previous.exists():comparisons[case]['previous_v15']=lateral(read(previous))
 return dict(passed=all(x['passed'] for x in comparisons.values()),cases=comparisons,tolerance_m=limit,tolerance_source='existing tracking reserve; paired validation only',global_metric=False)

def evaluate():
 stage('summarize');e=read(OUT/'evaluation.json');results=e['results'];gain=results['cutin551'].get('hd_gain');orig=results['original551'];local=orig.get('passed',False) and gain is not None and gain>0 and not [x for x in results['cutin551'].get('reasons',[]) if x!='no_target_improvement']
 paths=path_comparison();local=local and paths['passed']
 out=dict(path_comparison=paths,working_baseline_improved=local,working_gain_policy='Any strictly positive target HD gain, original nonregression, paired lateral nonregression and no new safety/stop failure; user authorized before v14 results',target_hd_gain=gain,original551=orig,strict_matrix=e['status'],global_patch_validated=e['global_patch_validated'],formal_certification=False,evaluation=asset(OUT/'evaluation.json'))
 write(D/'acceptance.json',out);write(D/'state.json',dict(stage='evaluated',**out));print(json.dumps(out,indent=2))
def render():
 sys.path[:0]=[str(C),str(R/'integrations/light-map551-v4'),str(R/'integrations/light-cutin-diagnostic')];from compare_native import render_pair
 old=P/'patch_v13_validation_r2_v1/cases/cutin551/on/B';new=OUT/'cases/cutin551/on/B';print(json.dumps(render_pair(OUT,old,new,{'off':read(old/'run_report.json'),'on':read(new/'run_report.json')}),indent=2))
if __name__=='__main__':
 s=sys.argv[1] if len(sys.argv)>1 else 'status'
 if s=='prepare':prepare()
 elif s=='full':
  gate();stage('check');stage('full');evaluate();render()
 elif s=='evaluate':evaluate()
 elif s=='render':render()
 elif s=='status':print(json.dumps(read(D/'state.json'),indent=2))
 else:raise ValueError(s)
