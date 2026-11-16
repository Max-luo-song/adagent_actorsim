"""Independent paired episodes; frozen v11 untouched; cross-scene gaps explicit."""

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

import argparse,ast,copy,importlib.util,datetime,hashlib,importlib.util,json,os,subprocess,sys
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
CODE=ROOT/'integrations/light-geometry-speed-v16';RELEASE=P26/'light_geometry_speed_v16';OUT=P26/'patch_v16_validation_v1'
from protocol import CASES,LIMITS,evaluate_pair,overall
COPY=['scene_manifest.json','baseline_freeze.json','baseline_verification.json','initial_context.json','baseline_input_audit.json','map/map.json','boot/manifest.json','scenario/scenario.json','scenario/target_track.npz','checks/cpu_admission.json','checks/native_cpu_replay/result.json']
def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def pins():return {str(p):sha(p) for p in sorted([*LOCAL.rglob('*.py'),LOCAL/'run.sh'])}
def provider():
 sys.path[:0]=[str(CODE),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
 spec=importlib.util.spec_from_file_location('_validation_v11',CODE/'pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.bindings();m.check(RELEASE);sys.path.insert(0,str(LOCAL));return m

def prepare(root):
 root=Path(root)
 if (root/'matrix.json').exists():return check(root)
 admission=read(root/'checks/cpu_tests.json')
 if not admission['passed'] or admission['code_sha256']!=pins():raise ValueError('Current harness CPU tests required')
 p=provider();base=read(RELEASE/'plan.json');cases={};receipts={}
 for name,case in CASES.items():
  if not case['ready']:
   path=P26.parent/'phase15/runs/20261025T093932_full/fast_gate'/f"{case['scene']}_unedited/baseline/scene_manifest.json"
   cases[name]=dict(**case,status='unavailable',existing_manifest=asset(path) if path.exists() else None,
     reason='No scene-matched authorized static map + semantic/native-score bundle registered for this controller; 551 adapters cannot be reused on this scene')
   continue
  cases[name]=dict(**case,status='prepared',arms={})
  for arm in ('off','on'):
   work=root/'cases'/name/arm
   for file in COPY:
    dst=work/file;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((RELEASE/file).read_bytes())
   plan=copy.deepcopy(base);plan.update(validation_case=name,validation_patch=arm,new_A_required=True,user_requested_direct_B=False)
   for key,file in [('scene','scene_manifest.json'),('scenario','scenario/scenario.json'),('baseline','baseline_freeze.json'),('target_track','scenario/target_track.npz')]:plan['assets'][key]=asset(work/file)
   write(work/'plan.json',plan)
   contract=dict(scene=551,edited=case['edited'],seed=base['seed'],controls=base['controls'],initial_context=base['initial_context'],
    scene_manifest=sha(work/'scene_manifest.json'),map=sha(work/'map/map.json'),scenario=sha(work/'scenario/scenario.json'),target_track=sha(work/'scenario/target_track.npz'),
    vehicle=asset(read(work/'scene_manifest.json')['vehicle_config']),
    worker=sha(CODE/'worker.py'),native_collision=base['native_collision_policy'],tracker=sha(ROOT/'integrations/light-cutin-diagnostic/tracker.py'),
    evaluator=sha(ROOT/'integrations/light-map551-v4/scoring.py'),termination=base['termination'],source_assets=base['assets'],
    patch=arm,fresh_history=True,score_kind='executed_path')
   # Per-arm staged paths differ; compare immutable asset CONTENTS instead.
   contract['source_assets']={k:v['sha256'] for k,v in base['assets'].items()}
   write(work/'contract.json',contract)
   cases[name]['arms'][arm]=str(work)
   for f in [*COPY,'plan.json','contract.json']:receipts[str(work/f)]=sha(work/f)
 from protocol import validate_contracts
 for name,c in cases.items():
  if c['ready']:validate_contracts(read(Path(c['arms']['off'])/'contract.json'),read(Path(c['arms']['on'])/'contract.json'))
 matrix=dict(schema='full_external_patch_nonregression_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  cases=cases,criteria=LIMITS,code_sha256=pins(),frozen_files=receipts,source_release=asset(RELEASE/'plan.json'),
  source_patch_cpu=asset(RELEASE/'checks/cpu_admission.json'),harness_cpu=asset(root/'checks/cpu_tests.json'),
  baseline='Frozen current-frame UniAD WITH original collision postprocessor, shared dt-matched HUGSIM iLQR and common environment termination; full external patch disabled',
  patch='Frozen v11 full external planning/evidence/execution guard enabled; same NN/iLQR/environment',
  archived_run_reuse=False,incremental_v10_v11_is_separate_evidence=True,
  held_out_scope='Cross-scene cases remain unavailable until assets and adapters are independently validated',
  existing_target_v11_known=True,criteria_frozen_before_new_pairs=True,formal_score_available=False)
 write(root/'matrix.json',matrix);return dict(prepared=True,ready_cases=[k for k,v in cases.items() if v['ready']],missing_cases=[k for k,v in cases.items() if not v['ready']],new_episodes=4,GPU_executed=False)

def check(root):
 root=Path(root);m=read(root/'matrix.json')
 if m['code_sha256']!=pins() or m['criteria']!=LIMITS:raise ValueError('Frozen harness/criteria changed')
 for key in ['source_release','source_patch_cpu','harness_cpu']:
  a=m[key]
  if sha(a['path'])!=a['sha256']:raise ValueError('Changed admission: '+a['path'])
 for f,h in m['frozen_files'].items():
  if sha(f)!=h:raise ValueError('Changed work input: '+f)
 provider()
 for c in m['cases'].values():
  if c['ready']:
   from protocol import validate_contracts
   validate_contracts(*[read(Path(c['arms'][a])/'contract.json') for a in ['off','on']])
 return dict(verified=True,ready_cases=[k for k,v in m['cases'].items() if v['ready']],missing_cases=[k for k,v in m['cases'].items() if not v['ready']],global_patch_validated=False)

def annotate_score(scores,arm):
 out=copy.deepcopy(scores)
 out.update(ground_truth_used_by_optimizer=arm=='on',validation_patch=arm,
   truth_scope='authorized static map used by external patch only when on; shared environment boundary check in both arms')
 return out

def load_runtime(arm):
 import types,controller_switch as switch
 switch.MODE=arm
 tree=switch.rewrite_runtime((CODE/'runtime.py').read_text())
 m=types.ModuleType('_paired_frozen_runtime');m.__file__=str(CODE/'runtime.py')
 exec(compile(tree,str(CODE/'runtime.py'),'exec'),m.__dict__)
 return m

def execute(root,case,arm):
 root=Path(root);check(root);m=read(root/'matrix.json');c=m['cases'][case]
 if not c['ready']:raise ValueError('Scene adapter not admitted')
 os.fstat(9) # Held by run.sh throughout the matrix, explicitly inherited below.
 work=Path(c['arms'][arm]);name='B' if c['edited'] else 'A'
 if (work/name).exists():raise FileExistsError('Existing/partial episode retained; no implicit replay')
 p=provider();p.check(work)
 runtime=load_runtime(arm)
 report=runtime.run_arm(work,name,read(work/'contract.json')['controls'],common=None,candidate=None)
 p.check(work)
 report.update(validation_contract=asset(work/'contract.json'),validation_patch=arm,validation_case=case,
  controls_selected_by='frozen_v16_external_patch_then_shared_ilqr' if arm=='on' else 'native_UniAD_collision_postprocessing_then_shared_ilqr',
  model_weights_unchanged=True,experiment_plan=asset(work/'plan.json'))
 write(work/name/'run_report.json',report)
 if not report['trace']:raise RuntimeError('No executed controls; inspect run_report')
 from scoring import score_arm
 from diagnostics import arm_summary
 scores=annotate_score(score_arm(work,name),arm);write(work/name/'native_eval/native_scores.json',scores)
 summary=arm_summary(work/name);check(root)
 write(work/'completion.json',dict(report=asset(work/name/'run_report.json'),summary=asset(work/name/'summary.json'),contract=asset(work/'contract.json'),source_pins=m['code_sha256']))
 return summary

def result(work,edited):
 work=Path(work);done=work/'completion.json'
 if not done.exists():return None
 receipt=read(done)
 for key in ['report','summary','contract']:
  a=receipt[key]
  if sha(a['path'])!=a['sha256']:raise ValueError('Result receipt mismatch')
 r=read(receipt['report']['path']);s=read(receipt['summary']['path']);contract=read(receipt['contract']['path'])
 applied=r['actor_intervention'].get('object_id')=='native:RigidNodes:3'
 if applied!=edited:raise ValueError('Actual edit does not match pair case')
 if r.get('validation_patch')!=contract['patch'] or r.get('validation_contract')!=asset(work/'contract.json'):raise ValueError('Execution mode/contract mismatch')
 if r.get('real_closed_loop') is not True or r.get('model_weights_unchanged') is not True:raise ValueError('Unverified execution')
 score=s.get('native_scores',{});values=score.get('scores',{}).get('executed_path')
 return dict(contract=contract,status=r['status'],score={k:v['value'] for k,v in values.items()} if values else None,
  collision=any(t.get('swept_collision',False) for t in r['trace']) or s['anomalies']['environment_collision'],
  map_departure=any(x['environment'].get('authorized_static_map_departure',False) for x in r['observations']),
  unresolved_stop=s['stop_recovery_unresolved'],progress_m=s['progress_along_initial_forward_m'],stopped_s=s['stopped_seconds'],formal_score=score.get('formal_native_score_available',False))

def summarize(root):
 root=Path(root);m=read(root/'matrix.json');results={};details={}
 for name,c in m['cases'].items():
  if not c['ready']:results[name]=dict(status='unavailable',passed=False,reason=c['reason']);continue
  arms={arm:result(c['arms'][arm],c['edited']) for arm in ['off','on']};details[name]=arms
  if all(arms.values()):
   # Exogenous actor programs must be identical despite independent ego paths.
   paths=[Path(c['arms'][arm])/('B' if c['edited'] else 'A')/'actor_tracks.npz' for arm in ['off','on']]
   import numpy as np
   with np.load(paths[0]) as a,np.load(paths[1]) as b:
    if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Paired actor worlds differ')
  results[name]=evaluate_pair(name,arms['off'],arms['on'])
 out=dict(results=results,details=details,**overall(results));write(root/'evaluation.json',out);return out

def full(root):
 import torch
 if not torch.cuda.is_available():return dict(status='deferred',reason='CUDA unavailable',GPU_executed=False)
 check(root);m=read(Path(root)/'matrix.json')
 for case,c in m['cases'].items():
  if not c['ready']:continue
  for arm in ['off','on']:
   if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('STOP present; next episode not started')
   work=Path(c['arms'][arm]);name='B' if c['edited'] else 'A'
   if (work/'completion.json').exists():result(work,c['edited']);continue
   if (work/name).exists():raise FileExistsError('Partial run retained: '+str(work))
   subprocess.run([sys.executable,'-u','-B',str(LOCAL/'pipeline.py'),'_execute','--output',str(root),'--case',case,'--arm',arm],cwd=ROOT,check=True,pass_fds=(9,))
 return summarize(root)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','check','full','summarize','_execute']);ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--case',choices=list(CASES));ap.add_argument('--arm',choices=['off','on']);a=ap.parse_args()
 if a.stage=='prepare':out=prepare(a.output)
 elif a.stage=='check':out=check(a.output)
 elif a.stage=='summarize':check(a.output);out=summarize(a.output)
 elif a.stage=='full':out=full(a.output)
 else:out=execute(a.output,a.case,a.arm)
 print(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False))
# This registered comparison uses frozen COMPLETED v13 runs, never native-off.
BASE_RELEASE=P26/'light_reference_join_v13_r2'
BASE_CODE=ROOT/'integrations/light-reference-join-v13-r2'
BASE_MATRIX=P26/'patch_v13_validation_r2_v1'

def baseline_snapshot(case):
 work=BASE_MATRIX/'cases'/case/'on';receipt=read(work/'completion.json')
 old=read(BASE_MATRIX/'matrix.json')
 if receipt['source_pins']!=old['code_sha256']:raise ValueError('Baseline validation source identity mismatch')
 for f,h in old['code_sha256'].items():
  if sha(f)!=h:raise ValueError('Baseline comparison code changed: '+f)
 for key in ('report','summary','contract'):
  a=receipt[key]
  if sha(a['path'])!=a['sha256']:raise ValueError('Baseline completed artifact changed')
 r=read(receipt['report']['path'])
 if r.get('real_closed_loop') is not True or r.get('model_weights_unchanged') is not True or r.get('validation_patch')!='on' or r['status']!='completed' or len(r['trace'])!=188:raise ValueError('Not a healthy completed full-v13 baseline')
 if r['experiment_plan']!=asset(work/'plan.json'):raise ValueError('Baseline plan changed')
 return dict(work=str(work),completion=asset(work/'completion.json'),report=asset(receipt['report']['path']),summary=asset(receipt['summary']['path']),contract=asset(work/'contract.json'),actor_tracks=asset(work/('B' if CASES[case]['edited'] else 'A')/'actor_tracks.npz'))

def base_check():
 spec=importlib.util.spec_from_file_location('_paired_baseline_v13_provider',BASE_CODE/'pipeline.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.check(BASE_RELEASE)
 sys.path[:0]=[str(LOCAL),str(CODE),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]

def prepare(root):
 root=Path(root)
 if (root/'matrix.json').exists():return check(root)
 admission=read(root/'checks/cpu_tests.json')
 if not admission['passed'] or admission['code_sha256']!=pins():raise ValueError('Current harness CPU tests required')
 base_check();provider();base=read(RELEASE/'plan.json');cases={};receipts={}
 for name,case in CASES.items():
  if not case['ready']:
   cases[name]=dict(**case,status='unavailable',reason='Scene-matched map/evaluator not admitted; never use scene551 assets here');continue
  baseline=baseline_snapshot(name);work=root/'cases'/name/'on'
  for file in COPY:
   dst=work/file;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((RELEASE/file).read_bytes())
  plan=copy.deepcopy(base);plan.update(validation_case=name,validation_patch='on',comparison_kind='v13_full_patch_vs_v16_full_patch',new_A_required=True,user_requested_direct_B=False)
  for key,file in [('scene','scene_manifest.json'),('scenario','scenario/scenario.json'),('baseline','baseline_freeze.json'),('target_track','scenario/target_track.npz')]:plan['assets'][key]=asset(work/file)
  write(work/'plan.json',plan)
  contract=dict(scene=551,edited=case['edited'],seed=base['seed'],controls=base['controls'],initial_context=base['initial_context'],scene_manifest=sha(work/'scene_manifest.json'),map=sha(work/'map/map.json'),scenario=sha(work/'scenario/scenario.json'),target_track=sha(work/'scenario/target_track.npz'),vehicle=asset(read(work/'scene_manifest.json')['vehicle_config']),worker=sha(CODE/'worker.py'),native_collision=base['native_collision_policy'],tracker=sha(ROOT/'integrations/light-cutin-diagnostic/tracker.py'),evaluator=sha(ROOT/'integrations/light-map551-v4/scoring.py'),termination=base['termination'],source_assets={k:v['sha256'] for k,v in base['assets'].items()},patch='on',fresh_history=True,score_kind='executed_path',comparison_version='v16')
  from protocol import validate_contracts
  left=read(baseline['contract']['path']);left['comparison_version']='v13';validate_contracts(left,contract)
  write(work/'contract.json',contract)
  cases[name]=dict(**case,status='prepared',arms=dict(on=str(work)),baseline_v13=baseline)
  for f in [*COPY,'plan.json','contract.json']:receipts[str(work/f)]=sha(work/f)
 matrix=dict(schema='v13_to_v16_paired_validation_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,criteria=LIMITS,code_sha256=pins(),frozen_files=receipts,source_release=asset(RELEASE/'plan.json'),baseline_release=asset(BASE_RELEASE/'plan.json'),source_patch_cpu=asset(RELEASE/'checks/cpu_admission.json'),harness_cpu=asset(root/'checks/cpu_tests.json'),baseline='existing completed full v13, frozen outcomes and fresh full history in those recorded episodes',patch='new v16 full patch; same exogenous conditions; fresh closed loop',archived_run_reuse=True,archived_controls_replayed=False,formal_score_available=False)
 write(root/'matrix.json',matrix);return dict(prepared=True,new_episodes=2,completed_v13_baselines_reused=2,missing_cases=[k for k,v in cases.items() if not v['ready']],GPU_executed=False)

def check(root):
 root=Path(root);m=read(root/'matrix.json')
 if m['code_sha256']!=pins() or m['criteria']!=LIMITS:raise ValueError('Frozen comparison harness/criteria changed')
 for key in ('source_release','baseline_release','source_patch_cpu','harness_cpu'):
  a=m[key]
  if sha(a['path'])!=a['sha256']:raise ValueError('Comparison admission changed')
 for f,h in m['frozen_files'].items():
  if sha(f)!=h:raise ValueError('Candidate exogenous input changed: '+f)
 base_check();provider()
 from protocol import validate_contracts
 for name,c in m['cases'].items():
  if c['ready']:
   if baseline_snapshot(name)!=c['baseline_v13']:raise ValueError('Frozen baseline receipt changed')
   left=read(c['baseline_v13']['contract']['path']);left['comparison_version']='v13';validate_contracts(left,read(Path(c['arms']['on'])/'contract.json'))
 return dict(verified=True,new_episodes=2,completed_v13_baselines_reused=2,global_patch_validated=False)

def summarize(root):
 root=Path(root);m=read(root/'matrix.json');results={};details={}
 for name,c in m['cases'].items():
  if not c['ready']:results[name]=dict(status='unavailable',passed=False,reason=c['reason']);continue
  base=baseline_snapshot(name);left=result(base['work'],c['edited']);right=result(c['arms']['on'],c['edited'])
  if left is None:raise ValueError('Completed baseline disappeared')
  left['contract']['comparison_version']='v13'
  if right:
   right['contract']['comparison_version']='v16'
   import numpy as np
   folder='B' if c['edited'] else 'A'
   with np.load(Path(base['work'])/folder/'actor_tracks.npz') as a,np.load(Path(c['arms']['on'])/folder/'actor_tracks.npz') as b:
    if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Exogenous actor worlds differ')
  details[name]=dict(baseline_v13=left,candidate_v16=right);results[name]=evaluate_pair(name,left,right)
 out=dict(results=results,details=details,baseline_versions=dict(baseline='v13',candidate='v16'),**overall(results));write(root/'evaluation.json',out);return out

def full(root):
 import torch
 if not torch.cuda.is_available():return dict(status='deferred',GPU_executed=False,reason='CUDA unavailable')
 check(root);m=read(Path(root)/'matrix.json')
 for case,c in m['cases'].items():
  if not c['ready']:continue
  if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('STOP present')
  work=Path(c['arms']['on']);name='B' if c['edited'] else 'A'
  if (work/'completion.json').exists():result(work,c['edited']);continue
  if (work/name).exists():raise FileExistsError('Partial candidate episode retained: '+str(work))
  subprocess.run([sys.executable,'-u','-B',str(LOCAL/'pipeline.py'),'_execute','--output',str(root),'--case',case,'--arm','on'],cwd=ROOT,check=True,pass_fds=(9,))
 return summarize(root)

if __name__=='__main__':main()
