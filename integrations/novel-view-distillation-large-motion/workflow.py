"""Reuse frozen conditioned-v5 stages for one more visible trajectory experiment."""

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

import argparse,importlib.util,json,sys,subprocess
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=ROOT/'integrations/novel-view-distillation'
sys.path[:0]=[str(HERE),str(BASE)]
from contracts import P26,read,write,sha
from trajectory import build
SOURCE=P26/'novel_distillation_v5/runs/20261109T054653Z'
OUTPUT=P26/'novel_distillation_large_motion_v1/scene551_lanechange4m'
PRESENT=ROOT/'integrations/novel-view-distillation-presentation/compare_trajectory.py'

def pins():
    files=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))+[PRESENT]
    return {str(p):sha(p) for p in files}

def provider():
    spec=importlib.util.spec_from_file_location('_conditioned_v5_large_motion',BASE/'regeneration_v5/workflow.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def verify_file(a):
    if sha(a['path'])!=a['sha256']:raise ValueError('Changed registered input: '+a['path'])

def asset(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

def prepare(output):
    out=Path(output).resolve()
    if out.exists():
        verify(out);return dict(prepared=True,reused_prepared_job=True,output=str(out),GPU_executed=False)
    original=read(SOURCE/'job.json')
    if sha(SOURCE/'compiled_trajectory.json')!=original['trajectory_sha256']:raise ValueError('Source pose pack changed')
    trajectory,audit=build(read(SOURCE/'compiled_trajectory.json'))
    inputs=HERE/'examples/scene551_lanechange4m';inputs.mkdir(parents=True,exist_ok=True)
    trajectory_path=inputs/'trajectory.json';task_path=inputs/'task.json'
    task=read(BASE/'regeneration_v5/task.json');task['trajectory']=str(trajectory_path)
    for path,value in [(trajectory_path,trajectory),(task_path,task)]:
        if path.exists() and read(path)!=value:raise ValueError('Existing experiment input differs: '+str(path))
        if not path.exists():write(path,value)
    result=provider().prepare(task_path,out)
    write(out/'large_motion_contract.json',dict(schema='large_motion_distillation_v1',code=pins(),
        trajectory=asset(trajectory_path),task=asset(task_path),source_pose_pack=asset(SOURCE/'compiled_trajectory.json'),
        policy=audit['policy'],original_experiment_unmodified=True,training_policy_reused='conditioned_v5 structure',
        no_online_training=True,no_driving_controller_change=True,automatic_promotion=False))
    write(out/'trajectory_diagnostics.json',audit)
    result.update(maximum_relative_heading_deg=audit['maximum_relative_heading_deg'],final_left_m=audit['final_left_m'])
    return result

def verify(output):
    out=Path(output).resolve();contract=read(out/'large_motion_contract.json')
    if contract['code']!=pins():raise ValueError('Experiment code changed; new job required')
    for k in ['trajectory','task','source_pose_pack']:verify_file(contract[k])
    module=provider();module.configure(out)
    return out,module

def completed(stage,out):
    relative={'render':'render/manifest.json','pseudo-gt':'pseudo_gt/manifest.json',
        'train':'structure/train/receipt.json','evaluate':'structure/evaluation/report.json','present':'trajectory_presentation/report.json'}[stage]
    path=out/relative
    if not path.exists():return False
    record=read(path)
    if stage=='present':return record.get('full_decode_passed') is True
    if record.get('completed') is not True:raise ValueError('Incomplete stage retained: '+stage)
    if stage in ('train','evaluate'):verify_file(record['checkpoint'])
    if stage=='render':
        from pipeline import validate_rows
        validate_rows(record['rows']);verify_file(record['appearance_conditioning_sources'])
    if stage=='pseudo-gt':
        for row in record['rows']:
            for name in ['target','valid_mask']:
                if name in row:verify_file(row[name])
    return True

def dispatch(stage,output):
    out,module=verify(output)
    if completed(stage,out):return dict(stage=stage,reused_completed_stage=True,output=str(out))
    if stage=='present':
        subprocess.run([sys.executable,'-B',str(PRESENT),str(out),str(out/'trajectory_presentation')],check=True)
        return dict(presentation=str(out/'trajectory_presentation/report.json'))
    return module.dispatch(stage,out)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','status','render','pseudo-gt','train','evaluate','present']);p.add_argument('--output',default=str(OUTPUT));a=p.parse_args()
    if a.stage=='prepare':result=prepare(a.output)
    elif a.stage=='status':
        out,_=verify(a.output);result=dict(output=str(out),stages={s:completed(s,out) for s in ['render','pseudo-gt','train','evaluate','present']})
    else:result=dispatch(a.stage,a.output)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
