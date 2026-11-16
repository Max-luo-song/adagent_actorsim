"""JSON editing-agent boundary over the real UnifiedSceneSession.

One owner/process, named resources, exactly-once request replay within this
owner, optimistic scene/revision checks, structured receipts and real previews.
No LLM, training, runtime editing, replacement, or driver is implied.
"""

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
import hashlib
import json
from numbers import Real
pass  # Path is provided by the portable boundary above
import re
import threading
import time
import traceback
import numpy as np
from PIL import Image
import agent_bootstrap as bootstrap
from agent_catalog import Catalog
from edit_api import read_command,UnsupportedEditError
from scene_edit_session import UnifiedSceneSession,saf

ACTIONS={'list_scenes','list_assets','get_capabilities','load_scene','list_objects','validate_edit','edit','preview','observe','reset','close','validate_ego_trajectory','preview_ego_trajectory','validate_ego_execution','execute_ego_trajectory','get_ego_state','list_traffic_lights','preview_actor_trajectory','run_scene_task','get_task_contract'}
ID=re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,63}')


class BridgeError(ValueError):
    def __init__(self,code,message,details=None):super().__init__(message);self.code=code;self.details=details


def fields(value,required,optional=()):
    if not isinstance(value,dict) or not set(required)<=set(value) or set(value)-set(required)-set(optional):
        raise BridgeError('invalid_request','missing or unknown fields; required='+','.join(sorted(required)))


def request(value):
    fields(value,{'schema','request_id','action','args'})
    if value['schema']!='agent_request_v1':raise BridgeError('invalid_request','expected agent_request_v1')
    if not isinstance(value['request_id'],str) or not ID.fullmatch(value['request_id']):raise BridgeError('invalid_request','invalid request_id')
    if not isinstance(value['action'],str) or value['action'] not in ACTIONS:raise BridgeError('unsupported_action','unsupported action')
    if not isinstance(value['args'],dict):raise BridgeError('invalid_request','args must be an object')
    try:return _hugsim_expand(json.loads(json.dumps(value,allow_nan=False)))
    except (ValueError,TypeError) as exc:raise BridgeError('invalid_request','request must be finite JSON data') from exc


class AgentBridge:
    def __init__(self,output,*,catalog=None,session=None,execution_backend=None,weather_factory=None):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.catalog=catalog or Catalog();self.session=session or UnifiedSceneSession()
        self.execution_backend=execution_backend or ('framework_renderer' if session is None else 'injected_session_unverified')
        self.scene_id=None;self.bindings={};self.cache={};self.closed=False
        self.lock=threading.RLock()
        self.render_count=0;self.load_count=0;self.reset_count=0
        self.control_count=0;self.execution_count=0;self.driving_state=None
        self.weather='clear';self.weather_factory=weather_factory or (self._weather_backend if session is None else None);self.weather_load_serial=0

    def _weather_backend(self,spec):
        import sys
        path=str(bootstrap.BRIDGE/'weather_runtime')
        if path not in sys.path:sys.path.insert(0,path)
        from runtime_backend import RuntimeWeatherBackend
        output=self.output/'weather'/f'load_{self.weather_load_serial:03d}';self.weather_load_serial+=1
        return RuntimeWeatherBackend(spec,output,protect_signals=True)

    def context(self):
        loaded=self.session.backend is not None
        phase='unloaded'
        if loaded:
            try:phase='observing' if self.session.editor.started else 'pre_run'
            except RuntimeError:phase='failed'
        return dict(scene_id=self.scene_id if loaded else None,
            generation=self.session.generation if loaded else None,
            edit_revision=self.session.revision if loaded else None,
            phase=phase,weather=self.weather if loaded else None,
            catalog_revision=self.catalog.revision,closed=self.closed)

    def require_scene(self):
        if self.scene_id is None or self.session.backend is None:raise BridgeError('scene_not_loaded','load_scene first')

    def require_version(self,args):
        self.require_scene()
        generation=args.get('expected_generation');revision=args.get('expected_revision')
        if not isinstance(generation,str) or not generation:raise BridgeError('invalid_request','expected_generation is required')
        if type(revision) is not int or revision<0:raise BridgeError('invalid_request','expected_revision must be a nonnegative integer')
        if generation!=self.session.generation:raise BridgeError('stale_generation','scene generation changed')
        if revision!=self.session.revision:raise BridgeError('stale_revision','edit revision changed')

    def capabilities(self):
        result=dict(observation_attitude_policies=['initial_ego','recorded_up','gravity_level'],scene_task_default_attitude='gravity_level',task_schema='scene_task_v1',task_result_schema='scene_task_result_v1',scene_task=True,
            weather_presets=['clear','rainy'] if self.weather_factory else ['clear'],
            weather_effect='RGB background style; visible actors and signals preserved',
            unified_driver_adapters=['reference_tracking'],resident_uniad_adapter=False,
            actions=sorted(ACTIONS),edit_phase='pre_run_only',runtime_edit=False,
            online_training=False,asset_generation=False,native_replacement=False,
            pre_run_preview=True,observe_seals_pre_run_edits=True,
            ego_trajectory_preview=True,ego_trajectory_control=True,trajectory_controller_connected=True,
            trajectory_feedback_source='native_motion_state',
            ego_trajectory_coordinates=['start_ego','source_world'],
            perception_connected=False,driver_connected=False,scene_persistent_within_process=True,
            request_replay_scope='this owner process; no cross-process resume',execution_backend=self.execution_backend)
        result['resources']=dict(scenes=self.catalog.list_scenes(),assets=self.catalog.list_assets())
        if self.scene_id is not None:
            result['scene']=self.session.edit_capabilities()
            result['scene_summary']=self.session.spec.summary()
            result['objects']=self.session.list_objects()
        return result

    def compile_edit(self,args):
        fields(args,{'command','expected_generation','expected_revision'})
        self.require_version(args)
        if self.session.editor.started or getattr(self.session.backend,'_last',None) is not None:
            raise BridgeError('edit_requires_pre_run','reset before editing an observed scene')
        value=args['command']
        if not isinstance(value,dict):raise BridgeError('invalid_edit','command must be an object')
        operation=value.get('operation');asset_id=None
        if not isinstance(value.get('object_id'),str):raise BridgeError('invalid_edit','object_id must be a string')
        if not isinstance(operation,str) or operation not in ('translate','rotate','insert','set_pose','remove','set_signal','set_signal_program','restore_signal','set_actor_trajectory','restore_actor_trajectory'):
            raise BridgeError('unsupported_operation','unknown edit operation')
        if operation=='insert':
            fields(value,{'operation','object_id','asset_id'}, {'placement_id','asset_to_world'})
            asset_id=value['asset_id']
            if not isinstance(asset_id,str):raise BridgeError('invalid_edit','asset_id must be a string')
            if ('placement_id' in value)==('asset_to_world' in value):raise BridgeError('invalid_edit','provide exactly one placement_id or asset_to_world')
            try:
                entry=self.catalog.asset(asset_id);self.catalog.verify_asset(asset_id)
            except KeyError as exc:raise BridgeError('unknown_asset',str(exc)) from exc
            except (OSError,ValueError) as exc:raise BridgeError('asset_resource_error',str(exc)) from exc
            pose=self.edit_pose(value,asset_id)
            command=dict(schema='scene_edit_v1',mode='diagnostic_preview',operation=operation,
                object_id=value['object_id'],asset=entry['command']['asset'],
                asset_calibration=entry['command']['asset_calibration'],asset_to_world=pose)
        elif operation=='set_pose':
            fields(value,{'operation','object_id'},{'placement_id','asset_to_world'})
            if ('placement_id' in value)==('asset_to_world' in value):raise BridgeError('invalid_edit','provide exactly one placement_id or asset_to_world')
            asset_id=self.bindings.get(value['object_id'])
            if asset_id is None:raise BridgeError('object_not_found','managed object not created through this bridge')
            command=dict(schema='scene_edit_v1',mode='diagnostic_preview',operation=operation,
                object_id=value['object_id'],asset_to_world=self.edit_pose(value,asset_id))
        else:
            command=dict(value,schema='scene_edit_v1',mode='diagnostic_preview')
            if 'schema' in value or 'mode' in value or 'expected_revision' in value or 'expected_generation' in value:
                raise BridgeError('invalid_edit','protocol metadata belongs outside command')
        try:command=read_command(command)
        except UnsupportedEditError as exc:raise BridgeError('unsupported_operation',str(exc)) from exc
        except ValueError as exc:raise BridgeError('invalid_edit',str(exc)) from exc
        objects={row['object_id']:row for row in self.session.list_objects()}
        oid=command['object_id']
        if operation=='insert':
            if oid in objects:raise BridgeError('object_exists','object_id already exists')
            if 'insert' not in self.session.edit_capabilities()['managed_operations']:raise BridgeError('unsupported_operation','backend cannot insert assets')
        else:
            if oid not in objects:raise BridgeError('object_not_found','object does not exist')
            if operation not in objects[oid]['operations']:raise BridgeError('unsupported_operation','operation not supported by this object')
        if operation in ('set_actor_trajectory','restore_actor_trajectory'):
            try:self.session.editor.actor_paths.validate(command)
            except ValueError as exc:raise BridgeError('invalid_actor_trajectory',str(exc)) from exc
        if operation=='translate' and command['object_id'].startswith('native:SMPLNodes:'):
            manager=self.session.editor.actor_paths
            if manager and int(command['object_id'].split(':')[-1]) in manager.programs:
                raise BridgeError('edit_conflict','restore the actor trajectory before native translation')
        if operation in ('set_signal','set_signal_program','restore_signal'):
            try:self.session.editor.traffic.validate(command)
            except ValueError as exc:raise BridgeError('invalid_signal',str(exc)) from exc
        if operation=='translate':
            _,group,index=oid.split(':');total=np.asarray(self.session.editor.offsets.get((group,int(index)),[0.,0.,0.]))+command['offset_world_m']
            if np.linalg.norm(total)>2:raise BridgeError('edit_limit','cumulative translation exceeds 2m')
        if operation=='rotate':
            _,group,index=oid.split(':');total=self.session.editor.yaws.get((group,int(index)),0)+command['yaw_degrees']
            if abs(total)>45:raise BridgeError('edit_limit','cumulative yaw exceeds 45 degrees')
        command.update(expected_generation=args['expected_generation'],expected_revision=args['expected_revision'])
        return command,asset_id

    def edit_pose(self,value,asset_id):
        if 'placement_id' in value:
            try:pose=self.catalog.placement(asset_id,value['placement_id'],self.scene_id)
            except KeyError as exc:raise BridgeError('unknown_placement',str(exc)) from exc
            except ValueError as exc:raise BridgeError('placement_scene_mismatch',str(exc)) from exc
        else:pose=value['asset_to_world']
        try:return saf.rigid(pose).tolist()
        except (ValueError,TypeError) as exc:raise BridgeError('invalid_pose',str(exc)) from exc

    def observe(self,args,request_id,*,preview=False):
        fields(args,{'expected_generation','expected_revision'},{'source_frame','timestamp_s','ego_to_world','attitude_policy'})
        self.require_version(args)
        editor=self.session.editor
        clear=getattr(self.session.backend,'clear_observation_history',None)
        if preview:
            if editor.started or getattr(self.session.backend,'_last',None) is not None:
                raise BridgeError('preview_requires_pre_run','reset before returning to editable previews')
            if not callable(clear):raise BridgeError('unsupported_preview','backend cannot clear history without deleting edits')
        pose_fields=set(args)-{'expected_generation','expected_revision','attitude_policy'}
        if pose_fields=={'source_frame'}:
            frame=args['source_frame']
            if type(frame) is not int or not 0<=frame<self.session.spec.frames:raise BridgeError('invalid_observation','source_frame outside scene')
            source=self.session.spec.source;inputs=Path(source['inputs']);start=source['start_timestep']
            pose=np.linalg.inv(np.loadtxt(inputs/'ego_pose'/f'{start:03d}.txt'))@np.loadtxt(inputs/'ego_pose'/f'{start+frame:03d}.txt')
            timestamp=frame/self.session.spec.fps;pose_source='recorded_ego_at_requested_frame; preview_only'
        elif pose_fields=={'timestamp_s','ego_to_world'}:
            timestamp=args['timestamp_s']
            if not isinstance(timestamp,Real) or isinstance(timestamp,bool) or not np.isfinite(timestamp):raise BridgeError('invalid_observation','timestamp_s must be finite')
            try:
                pose=saf.rigid(args['ego_to_world']);saf.frame_index(timestamp,self.session.spec.fps,self.session.spec.frames)
            except (ValueError,TypeError) as exc:raise BridgeError('invalid_observation',str(exc)) from exc
            pose_source='caller_ego_to_world'
        else:raise BridgeError('invalid_observation','provide source_frame OR timestamp_s and ego_to_world')
        from ego_attitude import POLICIES,gravity_up,corrected_pose
        policy=args.get('attitude_policy','initial_ego')
        if policy not in POLICIES:raise BridgeError('invalid_attitude_policy','unknown attitude_policy')
        up=gravity_up(self.session.spec)
        recorded=None
        if policy=='recorded_up':
            from ego_trajectory import recorded_pose
            recorded=recorded_pose(self.session.spec,saf.frame_index(timestamp,self.session.spec.fps,self.session.spec.frames))
        pose=corrected_pose(pose,policy,up,recorded)
        folder=self.output/'observations'/request_id;folder.mkdir(parents=True,exist_ok=False)
        if preview:
            try:
                raw=self.session.backend.render(pose,timestamp)
                packet=self.session._packet(raw,timestamp,pose)
            finally:
                try:clear()
                except Exception:
                    editor.closed=True
                    raise
        else:packet=self.session.render(pose,timestamp)
        self.render_count+=1
        environment=dict(weather=self.weather,physical_weather_geometry=False)
        if self.weather=='rainy':
            records=getattr(self.session.backend,'weather_records',[])
            if not records:raise RuntimeError('rainy backend returned no weather evidence')
            record=records[-1]
            environment.update(weather_record_index=record['index'],actor_rgb_preserved=all(c['actor_rgb_preserved'] for c in record['cameras'].values()),depth_unchanged=all(c['depth_unchanged'] for c in record['cameras'].values()),changed_pixels=sum(c['changed_pixels'] for c in record['cameras'].values()))
        result={k:v for k,v in packet.items() if k not in ('cameras','ego_to_world')}
        result.update(environment=environment,ego_to_world=packet['ego_to_world'].tolist(),pose_source=pose_source,cameras={},
            preview_only=preview,observation_history_committed=not preview)
        for name,camera in packet['cameras'].items():
            rgb=folder/f'{name}.png';depth=folder/f'{name}_depth.npy'
            Image.fromarray(camera['rgb']).save(rgb);np.save(depth,camera['depth'])
            result['cameras'][name]=dict(rgb=str(rgb),depth=str(depth),
                rgb_sha256=hashlib.sha256(camera['rgb'].tobytes()).hexdigest(),
                depth_sha256=hashlib.sha256(camera['depth'].tobytes()).hexdigest(),
                shape=list(camera['rgb'].shape),K=camera['K'].tolist(),camera_to_world=camera['camera_to_world'].tolist(),
                camera_to_ego=camera['camera_to_ego'].tolist())
        from ego_attitude import checked_horizon_metrics
        result['attitude']=checked_horizon_metrics(result,up)
        result['attitude_policy']=policy
        (folder/'packet.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        return result

    def execute(self,action,args,request_id):
        if self.closed:raise BridgeError('bridge_closed','this owner is closed')
        if action in ('list_scenes','list_assets','get_capabilities','list_objects','close','get_ego_state','list_traffic_lights','get_task_contract'):
            fields(args,set())
        if action=='get_task_contract':
            from scene_tasks import TASK_SCHEMA,RESULT_SCHEMA
            return dict(task_schema=TASK_SCHEMA,result_schema=RESULT_SCHEMA,capabilities=self.capabilities())
        if action=='run_scene_task':
            fields(args,{'task'},{'expected_generation','expected_revision'})
            if self.scene_id is not None:self.require_version(args)
            elif set(args)-{'task'}:raise BridgeError('invalid_request','unloaded scene task must omit scene version')
            from scene_tasks import execute_task,validate_task
            try:validate_task(args['task'])
            except (ValueError,TypeError) as exc:raise BridgeError('invalid_scene_task',str(exc)) from exc
            result=execute_task(args['task'],self,self.output/'tasks'/request_id,namespace=request_id)
            if result['status']!='completed':raise BridgeError('task_execution_failed','scene task failed; inspect task result',details=result)
            return result
        if action=='get_ego_state':
            self.require_scene();return dict(driving_state=copy.deepcopy(self.driving_state))
        if action=='list_traffic_lights':
            self.require_scene();return dict(traffic_lights=[r for r in self.session.list_objects() if r.get('group') in ('TrafficLightNodes','CalibratedSignals')])
        if action=='list_scenes':return dict(scenes=self.catalog.list_scenes())
        if action=='list_assets':return dict(assets=self.catalog.list_assets())
        if action=='get_capabilities':return self.capabilities()
        if action=='load_scene':
            fields(args,{'scene_id'},{'expected_generation','expected_revision','weather'})
            if not isinstance(args['scene_id'],str):raise BridgeError('invalid_request','scene_id must be a string')
            if self.scene_id is not None or 'expected_generation' in args or 'expected_revision' in args:self.require_version(args)
            weather=args.get('weather','clear')
            if weather not in ('clear','rainy'):raise BridgeError('unsupported_weather','supported weather: clear/rainy')
            if weather=='rainy' and self.weather_factory is None:raise BridgeError('unsupported_weather','rainy adapter unavailable in injected session')
            try:manifest=self.catalog.scene(args['scene_id'])
            except KeyError as exc:raise BridgeError('unknown_scene',str(exc)) from exc
            previous_factory=self.session.factory
            if weather=='rainy':self.session.factory=self.weather_factory
            try:result=self.session.load_scene(manifest)
            except Exception:
                if self.session.backend is None:self.scene_id=None;self.bindings.clear()
                raise
            finally:self.session.factory=previous_factory
            self.weather=weather
            self.scene_id=args['scene_id'];self.bindings.clear();self.driving_state=None;self.load_count+=1
            return dict(description=result,**self.context())
        if action=='list_objects':
            self.require_scene();return dict(objects=self.session.list_objects())
        if action in ('validate_edit','edit'):
            command,asset_id=self.compile_edit(args)
            if action=='validate_edit':return dict(valid=True,compiled_command=command,mutation_performed=False,
                would_edit_revision=self.session.revision+1,visual_quality_assured=False)
            receipt=self.session.edit(command)
            if command['operation']=='insert':self.bindings[command['object_id']]=asset_id
            if command['operation']=='remove':self.bindings.pop(command['object_id'],None)
            return dict(receipt=receipt,**self.context())
        if action in ('validate_ego_execution','execute_ego_trajectory'):
            fields(args,{'expected_generation','expected_revision','trajectory'},
                {'initial_speed_mps','initial_steer_rad','max_tracking_error_m','max_tracking_heading_error_degrees'})
            self.require_version(args)
            if self.session.editor.started or getattr(self.session.backend,'_last',None) is not None:
                raise BridgeError('execution_requires_pre_run','reset before starting a new driving execution')
            from ego_trajectory import compile_trajectory
            from ego_execution import execution_plan,execute_trajectory
            try:
                compiled=compile_trajectory(args['trajectory'],self.session.spec)
                plan=execution_plan(compiled,self.session.spec.vehicle,args)
            except (ValueError,TypeError,OverflowError) as exc:raise BridgeError('invalid_execution',str(exc)) from exc
            if action=='validate_ego_execution':
                return dict(valid=True,mutation_performed=False,control_executed=False,
                    sample_count=compiled['sample_count'],initial_speed_mps=plan['initial_speed_mps'],
                    initial_steer_rad=plan['initial_steer_rad'],dt=plan['dt'])
            return execute_trajectory(self,args,request_id,compiled,plan)
        if action in ('validate_ego_trajectory','preview_ego_trajectory'):
            fields(args,{'expected_generation','expected_revision','trajectory'},{'compare_recorded','level_recorded_baseline'})
            self.require_version(args)
            if 'compare_recorded' in args and type(args['compare_recorded']) is not bool:
                raise BridgeError('invalid_trajectory','compare_recorded must be boolean')
            if self.session.editor.started or getattr(self.session.backend,'_last',None) is not None:
                raise BridgeError('preview_requires_pre_run','reset before previewing an ego trajectory')
            if 'level_recorded_baseline' in args and type(args['level_recorded_baseline']) is not bool:raise BridgeError('invalid_trajectory','level_recorded_baseline must be boolean')
            from ego_trajectory import compile_trajectory,preview_trajectory
            try:compiled=compile_trajectory(args['trajectory'],self.session.spec)
            except (ValueError,TypeError,OverflowError) as exc:raise BridgeError('invalid_trajectory',str(exc)) from exc
            if action=='validate_ego_trajectory':return dict(valid=True,trajectory=compiled,mutation_performed=False)
            return preview_trajectory(self,args,request_id,compiled)
        if action=='preview_actor_trajectory':
            fields(args,{'expected_generation','expected_revision','object_id','start_frame','end_frame','camera_source_frame'},{'render_fps','comparison_baseline','attitude_policy'})
            self.require_version(args)
            if self.session.editor.started or getattr(self.session.backend,'_last',None) is not None:
                raise BridgeError('preview_requires_pre_run','reset before editable actor previews')
            manager=self.session.editor.actor_paths
            if manager is None:raise BridgeError('unsupported_actor_trajectory','native actor path backend unavailable')
            from actor_preview import preview_actor
            return preview_actor(self,args,request_id)
        if action in ('preview','observe'):return self.observe(args,request_id,preview=action=='preview')
        if action=='reset':
            fields(args,{'expected_generation','expected_revision'});self.require_version(args)
            editor=self.session.editor;generation=self.session.generation
            positions=dict(editor.originals);quaternions=dict(editor.original_quaternions);radiance=dict(editor.original_radiance)
            result=self.session.reset();self.bindings.clear();self.driving_state=None;self.reset_count+=1
            backend=self.session.backend;torch=getattr(backend,'torch',None)
            checks=dict(new_generation=self.session.generation!=generation,revision_cleared=self.session.revision==0,
                journal_cleared=not self.session.edit_journal(),managed_objects_cleared=not self.session.objects,
                no_implicit_observation=getattr(backend,'_last',None) is None,
                native_positions_restored=all(torch.equal(backend.trainer.models[g].instances_trans[:,i],v) for (g,i),v in positions.items()),
                native_rotations_restored=all(torch.equal(backend.trainer.models[g].instances_quats[:,i],v) for (g,i),v in quaternions.items()),
                actor_trajectories_cleared=not (self.session.editor.actor_paths and self.session.editor.actor_paths.programs),
                traffic_overrides_cleared=not (self.session.editor.traffic and self.session.editor.traffic.has_programs),
                native_radiance_restored=all(torch.equal(backend.trainer.models[g]._features_rest[state[0]],state[1]) for (g,i),state in radiance.items()))
            if not all(checks.values()):raise BridgeError('reset_verification_failed','reset did not restore all scene state')
            return dict(description=result,reset_audit=checks,**self.context())
        if action=='close':
            self.close();return dict(closed=True)
        raise BridgeError('unsupported_action',action)

    def close(self):
        self.session.close();self.scene_id=None;self.bindings.clear();self.driving_state=None;self.closed=True

    def handle(self,value):
        with self.lock:
            started=time.perf_counter();req=None;fingerprint=None;controls_before=self.control_count
            try:
                req=request(value)
                fingerprint=hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                cached=self.cache.get(req['request_id'])
                if cached is not None:
                    if cached[0]!=fingerprint:raise BridgeError('request_id_conflict','request_id already used for a different request')
                    response=copy.deepcopy(cached[1]);response['replayed']=True
                    response['context']=self.context()
                    response['original_elapsed_s']=response['elapsed_s'];response['elapsed_s']=time.perf_counter()-started
                    self.audit(req,response);return response
                result=self.execute(req['action'],req['args'],req['request_id'])
                response=dict(schema='agent_response_v1',request_id=req['request_id'],status='ok',result=result)
            except Exception as exc:
                code=exc.code if isinstance(exc,BridgeError) else 'backend_error'
                response=dict(schema='agent_response_v1',request_id=req['request_id'] if req else None,status='error',
                    error=dict(code=code,message=str(exc)))
                if isinstance(exc,BridgeError) and exc.details is not None:response['error']['details']=exc.details
                if not isinstance(exc,BridgeError):
                    folder=self.output/'errors';folder.mkdir(exist_ok=True)
                    name=req['request_id'] if req else 'invalid_request'
                    (folder/f'{name}.txt').write_text(traceback.format_exc())
            response.update(replayed=False,context=self.context(),elapsed_s=time.perf_counter()-started,
                training_performed=False,controls_executed=self.control_count-controls_before,control_executed=self.control_count>controls_before,formal_evaluation_admitted=False)
            if req is not None and fingerprint is not None and req['request_id'] not in self.cache:
                self.cache[req['request_id']]=(fingerprint,copy.deepcopy(response))
            self.audit(req,response);return copy.deepcopy(response)

    def audit(self,req,response):
        for name,value in (('requests.jsonl',req),('responses.jsonl',response)):
            with (self.output/name).open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
