"""Explicit driver registration, never an implicit 'latest' controller selection."""

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

import copy
pass  # Path is provided by the portable boundary above
from platform_io import ROOT,P26,LOCAL,read,write,asset,verify
PROFILE='joint_safety_v7_diagnostic'
PROGRAM='scene551_native3_cutin_stop_v1'
RELEASE=P26/'light_joint_safety_v7'
CODE=ROOT/'integrations/light-joint-safety-v7'

class Catalog:
    def __init__(self,path=None):
        self.path=Path(path or LOCAL/'catalog.json');self.data=read(self.path)
        if self.data.get('schema')!='scene_platform_catalog_v1':raise ValueError('Invalid catalog')
    def profile(self,name):
        if name not in self.data['drivers']:raise ValueError('Unregistered exact driver profile')
        return copy.deepcopy(self.data['drivers'][name])
    def program(self,name):
        if name not in self.data['programs']:raise ValueError('Unknown registered edit program')
        return copy.deepcopy(self.data['programs'][name])
    def verify(self,name):
        if name=='legacy_scene_task_v1':
            for a in self.data['legacy']['frozen_sources']:verify(a)
            verify(self.data['legacy']['task_schema']);return self.data['legacy']
        p=self.profile(name)
        for a in p['frozen_sources']:verify(a)
        program=self.program(PROGRAM)
        for a in program['resources']:verify(a)
        return p
    def capabilities(self):
        from sensor_contracts import capabilities as sensor_capabilities
        return dict(schema='scene_platform_capabilities_v1',task_schemas=['scene_task_v1','scene_task_v2'],
            source_migration=self.data.get('source_migration',{}),
            optional_sensors=sensor_capabilities(),
            legacy_modules=dict(task_schema='scene_task_v1',operations=['insert','translate','rotate','remove','set_signal','set_signal_program','set_actor_trajectory','ego_trajectory_preview','ego_reference_tracking','weather_preview'],
                validation='Existing task and module validators remain authoritative; native rigid vehicle programs use v2 only',
                perception_connected=False,closed_loop_UniAD=False,edit_plus_v2_driving_combination_admitted=False),
            drivers=[{k:v for k,v in p.items() if k not in ['frozen_sources','code_dir','release_root']}|dict(profile_id=n) for n,p in self.data['drivers'].items()],
            native_actor_programs=[dict(program_id=n,**{k:v for k,v in p.items() if k!='resources'}) for n,p in self.data['programs'].items()],
            weather_by_task_schema=dict(scene_task_v1=['clear','rainy'],scene_task_v2=['clear']),edit_phase='pre_run_only',LLM_connected=False,simulated_LLM_available=True,
            task_lifecycle='one real scene/backend per episode; verified reset then close',
            actual_driving_requires_GPU=True,formal_evaluation_admitted=False,
            legacy_scene_task_v1_entry=str(P26/'bridge/unified/run_phase26.sh'),
            unsupported_in_this_driver=['rainy closed-loop combination','arbitrary vehicle IDs/programs','mid-run editing','cross-task resident model reuse'])

def register(path=None):
    path=Path(path or LOCAL/'catalog.json')
    if path.exists():
        catalog=Catalog(path);catalog.verify(PROFILE)
        if 'legacy' not in catalog.data:
            catalog.data['legacy']=legacy_registration();write(path,catalog.data)
        return Catalog(path).verify(PROFILE)
    plan=read(RELEASE/'plan.json');summary=read(RELEASE/'B/summary.json')
    if not read(RELEASE/'checks/cpu_admission.json')['passed']:raise ValueError('Driver has no CPU admission')
    if summary['status'] not in ('completed','terminated'):raise ValueError('Driver actual execution not available')
    frozen=[asset(RELEASE/'plan.json'),asset(RELEASE/'checks/cpu_admission.json')]
    frozen.extend(asset(p) for p in sorted(CODE.rglob('*.py')));frozen.append(asset(CODE/'run.sh'))
    profile=dict(scene_id='551',code_dir=str(CODE),release_root=str(RELEASE),frozen_sources=frozen,
        dt_s=.1,source_start_frame=10,source_end_frame=198,max_control_steps=188,initial_speed_mps=4.,
        sensor_policy='source_original_rig',controller_owner='separate driving/patch workstream',
        native_collision_postprocessor_required=True,heavy_obstacle_mpc=False,
        scope='diagnostic engineering integration; no safe-driving or normal-scene nonregression certification',
        existing_GPU_controls=summary['controls'],existing_GPU_anomaly=summary['any_anomaly'],
        existing_GPU_termination=summary['termination_reason'],safe_driving_certified=False)
    program=dict(scene_id='551',object_id='native:RigidNodes:3',driver_profile=PROFILE,
        operation='set_actor_trajectory',interpretation='scripted existing-vehicle trajectory, no insertion',
        resources=[asset(RELEASE/'scenario/target_track.npz'),asset(RELEASE/'scenario/scenario.json')],
        source_activation_s=1.5,local_activation_s=.5,lane_change_duration_s=2.8,stop_local_time_s=5.3)
    write(path,dict(schema='scene_platform_catalog_v1',drivers={PROFILE:profile},programs={PROGRAM:program},legacy=legacy_registration()))
    return profile

def legacy_registration():
    folder=P26/'bridge/agent_json'
    return dict(entry=str(folder/'run_agent_json.py'),task_schema=asset(folder/'schemas/scene_task_v1.schema.json'),
        frozen_sources=[asset(p) for p in sorted(folder.glob('*.py'))]+[asset(folder/'catalog.json')],
        scope='Existing edit, preview and reference tracking modules; no frozen-UniAD closed-loop driving')
