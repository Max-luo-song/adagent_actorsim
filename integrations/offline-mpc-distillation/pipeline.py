"""Resume phase45 offline teachers, train a CPU student and validate current MPC proposals."""

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

import argparse
import copy
import datetime
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import pickle
import subprocess
import sys
import time
import numpy as np
import torch
from mpc_policy import Student,Provider,SCHEMA,control_bounds,family_distances

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase45/policy_distill_v1'
DRIVER=SOURCE.parent/'framework_driver_v1'
SEED=20261109


def read(path):return _hugsim_expand(json.loads(Path(path).read_text()))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def asset(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def verify(a):
    if sha(a['path'])!=a['sha256']:raise ValueError('Changed input: '+a['path'])


def pins():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+list(SOURCE.glob('*.py'))+list(DRIVER.glob('*.py'))
    files+=list(DRIVER.glob('*policy.json'))
    files+=[ROOT/'waymo23-native-import/configs/vehicle.json',ROOT/'waymo23-native-import/configs/planning.json']
    return {str(f.resolve()):sha(f) for f in files}


def prepare(output):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    inputs={str(f.resolve()):sha(f) for f in sorted((SOURCE/'data_inputs').glob('*/*.pkl'))}
    if len(inputs)!=88:raise ValueError('Expected the three captured episodes / 88 exact inputs')
    out.mkdir(parents=True)
    write(out/'plan.json',dict(schema='resumed_mpc_distillation_v1',code=pins(),inputs=inputs,
        paused_status=read(SOURCE/'paused_status.json'),seed=SEED,
        split='whole test_cutin episode held out; calibration timestamps 8..12; guard bands 6,7,13,14 excluded',
        collection_stage='resume phase45 collect_teachers.py; four CPU workers by default',
        training_device='cpu',online_solver_device_unchanged=True,automatic_promotion=False))
    write(out/'status.json',dict(status='resumed_by_user',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_pause_retained=True,student_trained=False,closed_loop_executed=False))
    return dict(prepared=True,output=str(out),captured_inputs=len(inputs))


def load(output):
    out=Path(output).resolve();plan=read(out/'plan.json')
    if plan['code']!=pins():raise ValueError('Pinned implementation changed')
    for path,digest in plan['inputs'].items():
        if sha(path)!=digest:raise ValueError('Captured optimizer input changed')
    return out,plan


def collect(output,workers):
    out,_=load(output)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
        PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(SOURCE/'native_runtime/lib'))
    with (out/'collect.log').open('a',buffering=1) as log:
        process=subprocess.Popen([sys.executable,'-u','-B',str(SOURCE/'collect_teachers.py'),'--workers',str(workers)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:print(line,end='',flush=True);log.write(line)
        if process.wait():raise RuntimeError('Teacher collection failed; see collect.log')
    return adopt(output)


def adopt(output):
    out,plan=load(output);dataset=read(SOURCE/'dataset.json');summary=read(SOURCE/'dataset_summary.json')
    teacher_files=[DRIVER/'empty_controller.py',DRIVER/'direct_problem.py',DRIVER/'policy_features.py',SOURCE/'collect_teachers.py']
    expected=hashlib.sha256(b''.join(f.read_bytes() for f in teacher_files)).hexdigest()
    if dataset['plan']['teacher_code_sha256']!=expected:raise ValueError('Teacher implementation mismatch')
    for job in dataset['plan']['jobs']:
        if plan['inputs'].get(job['input'])!=job['source_input_sha256']:raise ValueError('Teacher input provenance mismatch')
    if len(dataset['rows'])!=summary['kept'] or not dataset['source_frame_family_isolation']:raise ValueError('Incomplete teacher dataset')
    write(out/'dataset_receipt.json',dict(completed=True,dataset=asset(SOURCE/'dataset.json'),summary=asset(SOURCE/'dataset_summary.json'),
        plan=asset(out/'plan.json'),teacher_code_sha256=dataset['plan']['teacher_code_sha256']))
    return summary


def dataset_at(out):
    receipt=read(out/'dataset_receipt.json')
    for k in ['dataset','summary','plan']:verify(receipt[k])
    dataset=read(receipt['dataset']['path']);groups={k:[r for r in dataset['rows'] if r['split']==k] for k in ['train','calibration','test']}
    if any(not rows for rows in groups.values()):raise ValueError('Empty required dataset partition')
    families={k:{(r['source_episode'],r['source_step']) for r in rows} for k,rows in groups.items()}
    hashes={k:{r['feature_hash'] for r in rows} for k,rows in groups.items()}
    for a,b in [('train','calibration'),('train','test'),('calibration','test')]:
        if families[a]&families[b] or hashes[a]&hashes[b]:raise ValueError('Source-family/descriptor train-evaluation leakage')
    if any(r['source_episode']=='test_cutin' for r in groups['train']+groups['calibration']):raise ValueError('Held-out episode leaked')
    return receipt,dataset,groups


def train(output,epochs=500):
    out,_=load(output);receipt,dataset,groups=dataset_at(out)
    stage=out/'student';stage.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    rows=groups['train'];x=np.asarray([r['features'] for r in rows],np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Nonfinite/inconsistent training descriptors')
    identities={r['vehicle_identity'] for r in dataset['rows']}
    if len(identities)!=1:raise ValueError('Mixed vehicle identities')
    mean=x.mean(0);std=np.maximum(x.std(0),.05);z=(x-mean)/std
    distance=family_distances(z,[(r['source_episode'],r['source_step']) for r in rows])
    # This support gate is fit on training data alone. Test labels cannot tune it.
    threshold=float(max(np.quantile(distance,.95)*1.25,1e-4))
    lo,hi=control_bounds(rows[0]['vehicle']);labels=np.asarray([r['controls'] for r in rows]);y=(2*(labels-lo)/(hi-lo)-1).astype(np.float32)
    if not np.isfinite(y).all() or np.max(np.abs(y))>1.0001:raise ValueError('Teacher controls outside physical bounds')
    modes=np.asarray([r['mode'] for r in rows],np.int64)
    model=Student(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    tensors=torch.from_numpy(z),torch.from_numpy(y),torch.from_numpy(modes)
    def loss_on(part):
        xx=torch.tensor((np.asarray([r['features'] for r in part],np.float32)-mean)/std)
        yy=torch.tensor((2*(np.asarray([r['controls'] for r in part])-lo)/(hi-lo)-1),dtype=torch.float32)
        mm=torch.tensor([r['mode'] for r in part]);logits,controls=model(xx)
        loss=torch.nn.functional.smooth_l1_loss(controls[torch.arange(len(part)),mm],yy)+.1*torch.nn.functional.cross_entropy(logits,mm)
        return float(loss)
    best=float('inf');best_state=None;best_epoch=None
    with (stage/'training.jsonl').open('w',buffering=1) as stream:
        for epoch in range(epochs):
            model.train();values=[]
            for ids in np.array_split(rng.permutation(len(rows)),max(1,int(np.ceil(len(rows)/32)))):
                xx,yy,mm=(v[ids] for v in tensors);logits,controls=model(xx)
                loss=torch.nn.functional.smooth_l1_loss(controls[torch.arange(len(ids)),mm],yy)+.1*torch.nn.functional.cross_entropy(logits,mm)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();values.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():calibration=loss_on(groups['calibration'])
            if calibration<best:best=calibration;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            record=dict(epoch=epoch+1,train_loss=float(np.mean(values)),calibration_loss=calibration)
            stream.write(json.dumps(record)+'\n')
            if epoch==0 or (epoch+1)%50==0:print('student',record,flush=True)
    # Actual feature-schema version is pinned alongside its implementation.
    sys.path.insert(0,str(DRIVER));from policy_features import FEATURE_SCHEMA
    state=dict(schema=SCHEMA,model=best_state,dimension=x.shape[1],mean=mean.tolist(),std=std.tolist(),
        training_descriptors=z.tolist(),distance_threshold=threshold,trained_modes=sorted(set(modes.tolist())),
        vehicle_identity=next(iter(identities)),feature_schema=FEATURE_SCHEMA,selected_epoch=best_epoch,
        selection='calibration loss only; held-out episode never used',seed=SEED)
    torch.save(state,stage/'policy.pt')
    Provider(stage/'policy.pt')  # Reload the artifact before declaring training complete.
    result=dict(completed=True,checkpoint=asset(stage/'policy.pt'),dataset_receipt=asset(out/'dataset_receipt.json'),
        epochs=epochs,selected_epoch=best_epoch,calibration_loss=best,trained_modes=state['trained_modes'],
        samples={k:len(v) for k,v in groups.items()},feature_dimension=x.shape[1],distance_threshold=threshold,
        cpu_training=True,online_mpc_migrated_to_gpu=False,automatic_promotion=False)
    write(stage/'receipt.json',result);return result


def configure_source():
    sys.path.insert(0,str(SOURCE))
    from run_cutin import configure
    return configure()


def evaluate(output,execute_fallback=False):
    out,_=load(output);_,dataset,_=dataset_at(out);trained=read(out/'student/receipt.json');verify(trained['checkpoint'])
    stage=out/('evaluation_full_fallback' if execute_fallback else 'evaluation');stage.mkdir(exist_ok=False)
    torch.set_num_threads(1);driver=configure_source()
    from direct_problem import make_problem,PlanningProblem
    from policy_features import encode,identity
    provider=Provider(trained['checkpoint']['path'],trained['checkpoint']['sha256'])
    labels={r['input']:r for r in dataset['rows'] if r['split']=='test'};scores=[]
    # Evaluate ALL captured test problems, including rejected/missing teacher labels.
    for file in sorted((SOURCE/'data_inputs/test_cutin').glob('*.pkl')):
        began=time.perf_counter()
        with file.open('rb') as handle:data=pickle.load(handle)
        problem=make_problem(driver.optimized_controller,data['reference'],data['occupancy'],data['info'],copy.deepcopy(data['kwargs']))
        record=dict(input=asset(file),source_step=data['source_step'],source_episode=data['source_episode'])
        if not isinstance(problem,PlanningProblem):
            record.update(status='original_problem_unavailable',student_accepted=False,original_response_preserved=True)
        else:
            teacher=labels.get(str(file.resolve()))
            if teacher and teacher['feature_hash']!=identity(encode(problem)):raise ValueError('Reconstructed test features differ from teacher')
            decision=provider(problem);selected=decision['selected']
            record.update(status='student_checked' if selected else 'requires_original_mpc',student_accepted=selected is not None,audit=decision['audit'])
            if selected:
                # Re-evaluate controls rather than trusting the provider's claimed result.
                checked=problem.evaluate(np.asarray(selected['controls']))
                if checked is None:raise ValueError('Student receipt failed current full constraints')
                record.update(controls=selected['controls'],minimum_constraint=checked['minimum_constraint'],objective=checked['objective'])
                if teacher:
                    record['teacher_objective']=teacher['teacher_objective']
                    record['objective_increase_fraction']=(checked['objective']-teacher['teacher_objective'])/max(abs(teacher['teacher_objective']),1.)
            if not selected and execute_fallback:
                # A rejected proposal invokes the original full solver, not a substitute trajectory.
                kwargs=copy.deepcopy(data['kwargs']);kwargs.update(_proposal_provider=provider,_continuation=None)
                fallback=driver.optimized_controller.optimize(data['reference'],data['occupancy'],data['info'],**kwargs)
                path=stage/(file.stem+'_fallback.json');write(path,fallback)
                record.update(fallback_executed=True,fallback_result=asset(path),fallback_selected=fallback.get('selected') is not None)
        record['wall_seconds']=time.perf_counter()-began;scores.append(record)
        write(stage/(file.stem+'.json'),record)
        print('student heldout',file.stem,record['status'],flush=True)
    accepted=[r for r in scores if r['student_accepted']];ratios=[r['objective_increase_fraction'] for r in accepted if 'objective_increase_fraction' in r]
    result=dict(completed=True,test_problems=len(scores),student_accepted=len(accepted),
        original_mpc_required=sum(r['status']=='requires_original_mpc' for r in scores),
        original_problem_unavailable=sum(r['status']=='original_problem_unavailable' for r in scores),
        acceptance_fraction=len(accepted)/max(len(scores),1),accepted_objective_increase_mean=float(np.mean(ratios)) if ratios else None,
        accepted_objective_increase_max=max(ratios) if ratios else None,
        accepted_within_original_1pct_quality=sum(v<=.01 for v in ratios),quality_compared_accepted=len(ratios),
        all_accepted_full_current_constraints_passed=True,full_fallback_executed=execute_fallback,
        mean_pipeline_seconds=float(np.mean([r['wall_seconds'] for r in scores])),
        mean_provider_seconds=float(np.mean([r['audit']['total_seconds'] for r in scores if 'audit' in r])),
        checkpoint=trained['checkpoint'],metrics=scores,
        new_closed_loop_executed=False,whole_episode_speedup_verified=False,automatic_promotion=False,
        generalization_scope='fixed-input replay of one held-out episode, same scene551; not a fresh closed loop')
    write(stage/'report.json',result)
    # Explicit artifact for later opt-in integration; never silently install into the live driver.
    checks=dict(nonzero_heldout_acceptance=bool(accepted),all_accepted_have_teacher_comparison=len(ratios)==len(accepted),
        accepted_teacher_objective_within_1pct=bool(ratios) and max(ratios)<=.01,
        all_test_inputs_accounted_for=len(scores)==28)
    write(out/'candidate.json',dict(schema='checked_mpc_student_candidate_v1',checkpoint=trained['checkpoint'],
        training_receipt=asset(out/'student/receipt.json'),evaluation=asset(stage/'report.json'),
        checks=checks,offline_review_gate_passed=all(checks.values()),requires_phase45_current_problem_hooks=True,
        requires_full_current_constraints=True,reject_to_original_mpc=True,automatic_promotion=False,
        closed_loop_validation_required=True))
    write(out/'status.json',dict(status='offline_chain_completed',student_trained=True,evaluation=asset(stage/'report.json'),
        closed_loop_executed=False,automatic_promotion=False))
    return {k:v for k,v in result.items() if k!='metrics'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','collect','adopt','train','evaluate','evaluate-fallback','check'])
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--epochs',type=int,default=500);a=p.parse_args()
    actions={'prepare':lambda:prepare(a.output),'collect':lambda:collect(a.output,a.workers),'adopt':lambda:adopt(a.output),
        'train':lambda:train(a.output,a.epochs),'evaluate':lambda:evaluate(a.output),'evaluate-fallback':lambda:evaluate(a.output,True),'check':lambda:dict(verified=True,inputs=len(load(a.output)[1]['inputs']))}
    print(json.dumps(actions[a.stage](),indent=2,allow_nan=False))
