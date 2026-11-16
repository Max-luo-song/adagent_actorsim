"""Current MPC teacher: cached-current verified or offline local refinement."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,copy,hashlib,json,os,pickle,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
from paths import *
DRIVER_MODULE=None

def initialize():
 global DRIVER_MODULE
 os.environ['CUDA_VISIBLE_DEVICES']=''
 from run_cutin import configure
 DRIVER_MODULE=configure()

def work(job):
 from direct_problem import make_problem,PlanningProblem
 from policy_features import encode,identity,mode_of,vehicle_identity
 from scipy.optimize import minimize
 began=time.perf_counter()
 with Path(job['input']).open('rb') as f:data=pickle.load(f)
 info=copy.deepcopy(data['info']);info['ego_velo']+=job['dv'];info['ego_steer']+=job['ds'];kw=copy.deepcopy(data['kwargs']);opt=DRIVER_MODULE.optimized_controller;problem=make_problem(opt,data['reference'],data['occupancy'],info,kw)
 meta=dict(job,source_step=data['source_step'],source_episode=data['source_episode'],teacher_seconds=0.)
 if not isinstance(problem,PlanningProblem):return dict(meta,accepted=False,rejection='original_problem_rejected')
 features=encode(problem);original=data['source_result'].get('selected')
 if not original:return dict(meta,accepted=False,rejection='no_original_teacher_seed')
 seed=opt.bounded_seed(np.asarray(original['controls']),info['ego_velo'],info['ego_steer'],problem.interval,problem.vehicle);row=problem.evaluate(seed);cached=job['dv']==0. and job['ds']==0. and row is not None and (original.get('solver_success') is True or original.get('continuation')) and not original.get('used_verified_seed') and abs(row['objective']-original['objective'])<=1e-7*max(1.,abs(original['objective']))
 if not cached:
  result=minimize(problem.objective,seed.ravel(),method='SLSQP',bounds=problem.bounds,constraints=[dict(type='ineq',fun=problem.constraints)],options=dict(maxiter=100,ftol=1e-6,disp=False));row=problem.evaluate(np.asarray(result.x).reshape(6,2)) if result.success else None
  meta.update(solver_iterations=int(result.nit),solver_success=bool(result.success),solver_message=str(result.message))
 full_fallback=False
 if row is None and job['split'] in ('test','calibration'):
  # Hard validation cases may not be silently excluded because a short local
  # refinement failed. Ask the full frozen-policy teacher and recheck its rows.
  full_args=dict(kw,_continuation=None,_proposal_provider=None,_problem_callback=None)
  full=opt.optimize(data['reference'],data['occupancy'],info,**full_args);rows=list(full.get('trials',[]))
  if full.get('selected'):rows.append(full['selected'])
  for candidate in sorted(rows,key=lambda r:r.get('objective',float('inf'))):
   if candidate.get('solver_success') is not True or candidate.get('used_verified_seed'):continue
   row=problem.evaluate(np.asarray(candidate['controls']))
   if row is not None:full_fallback=True;break
 if row is None:return dict(meta,accepted=False,rejection='no_converged_current_quality_feasible_label',teacher_seconds=time.perf_counter()-began)
 return dict(meta,accepted=True,features=features.tolist(),feature_hash=identity(features),controls=row['controls'],states=row['states'],lidar=row['lidar'],teacher_objective=row['objective'],minimum_constraint=row['minimum_constraint'],native_proxy=row['native_alignment_proxy'],mode=mode_of(row,problem.reference,problem.vehicle),vehicle_identity=vehicle_identity(problem.vehicle),initial=problem.initial.tolist(),vehicle=vars(problem.vehicle),interval=problem.interval,interpolation=problem.interpolation,teacher_seconds=time.perf_counter()-began,teacher_kind='current_problem_verified_cached_expert' if cached else 'current_phase44_full_teacher_fallback' if full_fallback else 'current_phase44_objective_local_mpc_refinement',optimality_certificate=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();folder=HERE/'teacher_data';folder.mkdir(exist_ok=True);jobs=[]
 # Whole cut-in episode is untouched by training and calibration.
 for episode in ('train_search','train_envelope','test_cutin'):
  for path in sorted((HERE/'data_inputs'/episode).glob('*.pkl')):
   index=int(path.stem.split('_')[-1]);split='test' if episode=='test_cutin' else 'calibration' if 8<=index<=12 else 'unused_guard_band' if index in (6,7,13,14) else 'train'
   if split=='unused_guard_band':continue
   changes=[(0.,0.)] if split!='train' else [(0.,0.),(-.5,0.),(.5,0.),(0.,-.015),(0.,.015)]
   for dv,ds in changes:
    spec=dict(input=str(path),source_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),split=split,dv=dv,ds=ds,synthetic_planning_counterfactual=bool(dv or ds));spec['job_id']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20];jobs.append(spec)
 code=[DRIVER/'empty_controller.py',DRIVER/'direct_problem.py',DRIVER/'policy_features.py',Path(__file__)];signature=hashlib.sha256(b''.join(f.read_bytes() for f in code)).hexdigest();cache=folder/signature[:12];cache.mkdir(exist_ok=True)
 plan=dict(teacher_code_sha256=signature,jobs=jobs,workers=args.workers,split='Whole cut-in episode held out; disjoint source-frame families for calibration; adjacent guard bands; no random perturbation leakage',scene_coverage=[551],cross_scene_generalization_claimed=False);(cache/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');pending=[j for j in jobs if not (cache/(j['job_id']+'.json')).exists()];print('Teacher jobs',len(jobs),'pending',len(pending),'cache',cache,flush=True)
 with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),initializer=initialize) as pool:
  futures={pool.submit(work,j):j for j in pending}
  for count,future in enumerate(as_completed(futures),1):
   job=futures[future]
   try:row=future.result()
   except Exception as exc:row=dict(job,accepted=False,rejection=repr(exc))
   (cache/(job['job_id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
   if count%10==0 or count==len(pending):print('Collected',count,'/',len(pending),flush=True)
 rows=[_hugsim_expand(json.loads((cache/(j['job_id']+'.json')).read_text())) for j in jobs];good=[r for r in rows if r.get('accepted')];test_hashes={r['feature_hash'] for r in good if r['split']=='test'};cal_hashes={r['feature_hash'] for r in good if r['split']=='calibration'};kept=[];dropped=[]
 for row in good:
  if row['split']=='train' and row['feature_hash'] in test_hashes|cal_hashes or row['split']=='calibration' and row['feature_hash'] in test_hashes:dropped.append(row['job_id']);continue
  kept.append(row)
 dataset=dict(schema='mpc_distill_dataset_v1',plan=plan,rows=kept,all_job_count=len(rows),rejected_count=len(rows)-len(good),cross_split_exact_duplicates_removed=dropped,source_frame_family_isolation=True,teacher_is_global_optimum=False)
 (HERE/'dataset.json').write_text(json.dumps(dataset)+'\n');summary=dict(jobs=len(rows),accepted=len(good),kept=len(kept),splits={key:sum(r['split']==key for r in kept) for key in ('train','calibration','test')},rejections={key:sum(r.get('rejection')==key for r in rows) for key in {r.get('rejection') for r in rows if not r.get('accepted')}},mode_counts={str(k):sum(r['mode']==k for r in kept if r['split']=='train') for k in range(3)},teacher_core_seconds=sum(r.get('teacher_seconds',0.) for r in rows),cache=str(cache));(HERE/'dataset_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
