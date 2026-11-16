"""Dispatch registered edits into an unchanged driver; observe load/edit/reset/close."""

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

import copy,importlib.util,pickle,sys,time
pass  # Path is provided by the portable boundary above
from platform_io import ROOT,P26,LOCAL,read,write,asset,verify

COPY_FILES=['scene_manifest.json','baseline_freeze.json','baseline_verification.json','initial_context.json',
    'baseline_input_audit.json','map/map.json','boot/manifest.json','scenario/scenario.json','scenario/target_track.npz',
    'checks/cpu_admission.json','checks/native_cpu_replay/result.json']

def load_provider(profile):
    code=Path(profile['code_dir']);spec=importlib.util.spec_from_file_location('_registered_closed_loop_provider',code/'pipeline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path[:0]=[str(code),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
    module.bindings();return module

def stage_work(task,profile,work):
    work=Path(work);release=Path(profile['release_root']);provider=load_provider(profile);provider.check(release)
    if (work/'plan.json').exists():
        plan=read(work/'plan.json')
        from platform_io import fingerprint
        if plan.get('platform_task_digest')!=fingerprint(task):raise ValueError('Work directory belongs to a different task')
        provider.check(work);return provider
    for name in COPY_FILES:
        dst=work/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((release/name).read_bytes())
    plan=copy.deepcopy(read(release/'plan.json'))
    from platform_io import fingerprint
    plan.update(platform_task_digest=fingerprint(task),platform_profile_id=task['simulation']['profile'],
        platform_requested_controls=task['simulation']['control_steps'],platform_owns_only_dispatch_and_lifecycle=True)
    for key,filename in [('scene','scene_manifest.json'),('scenario','scenario/scenario.json'),('baseline','baseline_freeze.json'),('target_track','scenario/target_track.npz')]:
        plan['assets'][key]=asset(work/filename)
    write(work/'plan.json',plan);provider.check(work)
    return provider

def instrument(runtime,native_module,task,work,event):
    """Observe existing edit functions and add mandatory lifecycle verification only."""
    import numpy as np
    from PIL import Image
    from environment import actor_arrays
    work=Path(work);original_factory=native_module.make_types;Backend,Session=original_factory();original_install=runtime.install_trajectory
    context=read(work/'initial_context.json');pose=np.asarray(context['ego_to_world']);timestamp=context['source_timestamp_s']
    lifecycle=work/'platform_lifecycle';expected_edit=task['edits'][0] if task['edits'] else None
    from optional_sensors import Recorder
    recorder=Recorder(task.get('sensors',{}).get('lidar'),work/'optional_sensors',event)
    class PlatformSession(Session):
        def load_scene(self,manifest):
            value=super().load_scene(manifest);event('scene_loader','loaded',dict(manifest=str(manifest),generation=self.generation,revision=self.revision))
            self._platform_original=actor_arrays(self);self._platform_baseline=self._capture('before_edit')
            self.backend.clear_observation_history()
            if self.editor.started or self.backend._last is not None:raise RuntimeError('Platform preview sealed editing phase')
            return value
        def render(self,*args,**kwargs):
            packet=super().render(*args,**kwargs)
            receipt=recorder.record(self,packet)
            if receipt is not None:packet['optional_pointcloud']=dict(provider=receipt['provider'],source_frame=receipt['source_frame'],record_only=True)
            return packet
        def _capture(self,label):
            from platform_io import fingerprint,sha
            raw=self.backend.render(pose,timestamp);paths={}
            for name,camera in raw['cameras'].items():
                path=lifecycle/label/(name+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(camera['rgb']).save(path);paths[name]=asset(path)
            return paths
        def close(self):
            if self.backend is None:return super().close()
            cleanup=dict(reset_attempted=True,reset_verified=False,closed=False)
            problem=None
            try:
                previous=self.generation;self.reset();restored=actor_arrays(self);before=self._platform_original
                actors=set(restored)==set(before) and all(np.array_equal(restored[k],before[k]) for k in before)
                after=self._capture('after_reset');pixels=all(after[k]['sha256']==v['sha256'] for k,v in self._platform_baseline.items())
                cleanup.update(generation_changed=self.generation!=previous,revision_zero=self.revision==0,
                    actor_state_restored=actors,initial_camera_pixels_restored=pixels,remaining_vehicle_programs=len(self.vehicles.programs),
                    generation=self.generation,revision=self.revision)
                cleanup['reset_verified']=all([cleanup['generation_changed'],cleanup['revision_zero'],actors,pixels,len(self.vehicles.programs)==0])
                event('session_lifecycle','reset_verified' if cleanup['reset_verified'] else 'reset_mismatch',cleanup)
            except BaseException as exc:cleanup['error']=repr(exc);problem=exc
            finally:
                try:recorder.flush()
                except BaseException as exc:cleanup['sensor_finalize_error']=repr(exc);problem=problem or exc
                try:super().close();cleanup['closed']=True
                except BaseException as exc:cleanup['close_error']=repr(exc);problem=problem or exc
                cleanup['recovery_required']=not(cleanup['reset_verified'] and cleanup['closed'])
                write(lifecycle/'cleanup.json',cleanup);event('session_lifecycle','closed',cleanup)
            if problem is not None:raise problem
            if not cleanup['reset_verified']:raise RuntimeError('Platform reset did not restore scene state and initial pixels')
    def factory():return Backend,PlatformSession
    def install(session,path,start):
        if expected_edit is None:raise RuntimeError('Driver attempted an edit absent from task instructions')
        expected=work/'scenario/target_track.npz'
        if Path(path).resolve()!=expected.resolve():raise RuntimeError('Driver edit did not use the compiled registered program')
        before=dict(generation=session.generation,revision=session.revision)
        receipt=original_install(session,path,start)
        if receipt['object_id']!=expected_edit['object_id']:raise RuntimeError('Driver edited a different object')
        record=dict(command=expected_edit,actual_receipt=receipt,before=before,
            after=dict(generation=session.generation,revision=session.revision),program=asset(path))
        write(work/'platform_edit_receipt.json',record);event('native_vehicle_trajectory','edited',record)
        return receipt
    native_module.make_types=factory;runtime.install_trajectory=install
    def restore():native_module.make_types=original_factory;runtime.install_trajectory=original_install
    return restore

def execute(task,profile,work,event):
    provider=stage_work(task,profile,work)
    import runtime,native_car_edit
    arm='B' if task['edits'] else 'A';steps=task['simulation']['control_steps'];work=Path(work)
    if (work/arm).exists():raise FileExistsError('Partial or existing drive retained; never rerun controls implicitly')
    restore=instrument(runtime,native_car_edit,task,work,event)
    event('closed_loop_driver','started',dict(profile=task['simulation']['profile'],requested_controls=steps,source_start_frame=profile['source_start_frame']))
    try:report=runtime.run_arm(work,arm,steps,common=None,candidate=None)
    finally:restore()
    provider.check(work);report.update(model_weights_unchanged=True,experiment_plan=asset(work/'plan.json'))
    write(work/arm/'run_report.json',report)
    event('closed_loop_driver','finished',dict(status=report['status'],actual_controls=len(report['trace']),termination=report.get('termination_reason')))
    if report['trace']:
        from scoring import score_arm
        from diagnostics import arm_summary
        event('native_evaluator','started',dict(actual_controls=len(report['trace'])))
        score_arm(work,arm);arm_summary(work/arm);event('native_evaluator','finished',dict(scores=asset(work/arm/'native_eval/native_scores.json')))
    return arm
