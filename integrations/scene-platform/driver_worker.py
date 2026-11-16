"""Isolated process for frozen provider imports and real GPU execution."""

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

import argparse,time,traceback
pass  # Path is provided by the portable boundary above
from platform_io import read,write,asset
from platform_catalog import Catalog
from platform_contracts import compile_task

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','run','sensor-validate']);p.add_argument('job');a=p.parse_args()
    job=Path(a.job);events=[];began=time.monotonic()
    def event(module,state,details):
        events.append(dict(module=module,state=state,elapsed_wall_s=time.monotonic()-began,details=details))
        write(job/'module_events.json',events)
        print(module+': '+state,flush=True)
    try:
        task=read(job/'task.json');catalog=Catalog(job/'catalog.json');compiled=compile_task(task,catalog)
        profile=catalog.verify(compiled['profile_id'])
        for entry in read(job/'manifest.json')['platform_code']:__import__('platform_io').verify(entry)
        if compiled['profile_id']=='legacy_scene_task_v1':
            from legacy_adapter import execute_legacy
            execute_legacy(a.stage,task,profile,job,event);return
        from driver_adapter import stage_work,execute
        stage_work(task,profile,job/'work')
        import torch
        check=dict(verified=True,cuda_available=torch.cuda.is_available(),compiled_task=asset(job/'compiled_task.json'),
            driver_sources_verified=True,controller_changed=False,control_executed=False)
        from sensor_contracts import configuration
        check['optional_sensor_configuration']=configuration(task.get('sensors',{}).get('lidar'))
        write(job/'preflight.json',check)
        if a.stage=='check':return
        if not check['cuda_available']:raise RuntimeError('CUDA unavailable; no control executed')
        if a.stage=='sensor-validate':
            from sensor_validation import validate
            result=validate(task,profile,job/'work',event);write(job/'task_result.json',result);return
        arm=execute(task,profile,job/'work',event)
        from platform_results import collect,present
        report=read(job/'work'/arm/'run_report.json')
        if report['observations']:
            event('presentation','started',{});present(job/'work',arm);event('presentation','finished',{})
        result=collect(task,profile,job/'work')
        result['platform_wall_seconds']=time.monotonic()-began
        result['artifacts'].append(dict(kind='actual_module_dispatch_events',**asset(job/'module_events.json')))
        write(job/'task_result.json',result)
    except BaseException as exc:
        write(job/'worker_error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        if a.stage in ('run','sensor-validate'):
            result=None
            try:
                from platform_results import collect
                result=collect(task,profile,job/'work')
            except Exception:pass
            if result is None:
                source=job/'work'/('B' if read(job/'task.json').get('edits') else 'A')/'run_report.json'
                partial=read(source) if source.exists() else None
                cleanup_path=job/('work/sensor_validation/cleanup.json' if a.stage=='sensor-validate' else 'work/platform_lifecycle/cleanup.json')
                cleanup=read(cleanup_path) if cleanup_path.exists() else None
                started=(job/'work/A').exists() or (job/'work/B').exists() or (job/'legacy').exists()
                result=dict(schema='scene_task_result_v2',task_id=read(job/'task.json')['task_id'],status='failed',
                    execution=dict(control_executed=bool(partial['trace']) if partial else None if started else False,
                        executed_controls=len(partial['trace']) if partial else None,execution_verified=False),
                    recovery_required=cleanup.get('recovery_required',True) if cleanup else started,
                    artifacts=[dict(kind='partial_actual_run_report',**asset(source))] if partial else [])
            result['status']='failed';result.setdefault('failure_reasons',[]).append(dict(domain='platform_worker',reason=repr(exc)))
            result['artifacts'].append(dict(kind='worker_error',**asset(job/'worker_error.json')))
            write(job/'task_result.json',result)
        raise

if __name__=='__main__':main()
