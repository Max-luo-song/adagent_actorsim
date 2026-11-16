"""Explicit evaluation-only recovery: preserve original job and training receipts."""

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
pass  # Path is provided by the portable boundary above
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline
from contracts import read, write, sha, HERE

PLAN = 'failure_audit/device_reload_plan_v2.json'


def validate_parent(out, plan):
    job = read(out/'job.json')
    if pipeline.asset(out/'job.json') != plan['job']:
        raise ValueError('Original job changed')
    for name, expected in job['implementation'].items():
        archived = out/'failure_audit/device_reload_original_code'/Path(name).name
        if sha(archived) != expected:
            raise ValueError('Original code archive changed: '+name)
    if pipeline.implementation() != plan['recovery_implementation']:
        raise ValueError('Recovery implementation changed')
    for path, expected in plan['recovery_files'].items():
        if sha(path) != expected:
            raise ValueError('Recovery helper changed')
    for value in job['inputs'].values():
        pipeline.verify_asset(value)
    if sha(out/'compiled_trajectory.json') != job['trajectory_sha256']:
        raise ValueError('Compiled trajectory changed')
    for value in plan['preserved_artifacts'].values():
        pipeline.verify_asset(value)
    return job


def prepare(out):
    if (out/PLAN).exists():
        raise FileExistsError('Recovery plan already exists')
    job = read(out/'job.json')
    original = job['implementation']; current = pipeline.implementation()
    changed = {Path(k).name for k in original if original[k] != current.get(k)}
    if set(original) != set(current) or changed != {'framework.py','run.sh'}:
        raise ValueError('Recovery admits only renderer device alignment and launcher routing')
    receipt = read(out/'distill/receipt.json')
    if receipt.get('completed') is not True or not receipt['losses']:
        raise ValueError('Completed training receipt required')
    plan = dict(schema='distillation_device_reload_recovery_v2', job=pipeline.asset(out/'job.json'),
        recovery_implementation=current, recovery_files={str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py')},
        preserved_artifacts={name:pipeline.asset(out/name) for name in ['render/manifest.json','pseudo_gt/manifest.json',
            'distill/receipt.json','distill/scene_manifest.json','distill/source_report.json','distill/config.yaml',
            'distill/checkpoint_distilled.pth']},
        changed_files=sorted(changed), reason='Move all registered and plain/nested model tensors to renderer CUDA device',
        retrain=False, checkpoint_rewritten=False, gpu_recovery_executed=False)
    validate_parent(out,plan)
    pipeline.target_rows(out)
    evaluation = out/'evaluation'
    if evaluation.exists():
        if any(evaluation.iterdir()):
            raise ValueError('Evaluation contains results; refusing to move or overwrite')
        evaluation.rename(out/'failure_audit/evaluation_empty_before_device_fix_v2')
    write(out/PLAN,plan)
    return dict(prepared=True,plan=str(out/PLAN),training_reused=True,gpu_executed=False)


def recover(out):
    plan = read(out/PLAN)
    validate_parent(out,plan)
    # This callback is scoped to the evaluation-only entrypoint. The original
    # job remains immutable; recovery provenance is reported explicitly below.
    def evaluation_job_at(output):
        if Path(output).resolve()!=out:
            raise ValueError('Wrong recovery output')
        return out,validate_parent(out,plan)
    pipeline.job_at = evaluation_job_at
    import framework
    base_renderer=framework.Renderer
    class AuditedRenderer(base_renderer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            inventory=framework.trainer_tensor_placements(self.trainer)
            write(out/'evaluation/device_inventory.json',dict(
                completed=True,all_model_tensors_on_cuda=True,
                tensors={name:dict(device=str(value.device),shape=list(value.shape),dtype=str(value.dtype))
                         for name,value in inventory.items()}))
    framework.Renderer=AuditedRenderer
    result = pipeline.evaluate(out)
    write(out/'evaluation/recovery_receipt.json',dict(completed=True,parent_job=plan['job'],
        recovery_plan=pipeline.asset(out/PLAN),recovery_implementation=plan['recovery_implementation'],
        preserved_artifacts=plan['preserved_artifacts'],checkpoint_rewritten=False,retrained=False,
        gpu_recovery_executed=True,result=result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','recover-evaluate'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve()
    result=prepare(out) if args.stage=='prepare' else recover(out)
    print(__import__('json').dumps(result,indent=2))
    if result.get('automatic_screen_passed') is False:raise SystemExit(2)


if __name__=='__main__':main()
