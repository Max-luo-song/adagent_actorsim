"""Uniform pre-run editing contract; backend capabilities remain explicit.

No renderer or model is imported here. Native object IDs are scoped to one
scene generation, with model indices (not dataset annotation IDs).
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
import json
from numbers import Real
pass  # Path is provided by the portable boundary above
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'framework_backend'))

NATIVE_GROUPS = ('RigidNodes', 'SMPLNodes', 'DeformableNodes')
from traffic_lights import SIGNAL_OPERATIONS, validate_signal

from actor_trajectories import ACTOR_OPERATIONS, validate_path

FIELDS = dict(SIGNAL_OPERATIONS)
FIELDS.update(ACTOR_OPERATIONS)
FIELDS.update({'rotate': {'yaw_degrees'}, 'translate': {'offset_world_m'}, 'insert': {'asset', 'asset_calibration', 'asset_to_world'},
          'set_pose': {'asset_to_world'}, 'remove': set()})


class UnsupportedEditError(ValueError):
    pass


def read_command(value):
    if not isinstance(value, dict):
        raise ValueError('edit command must be a mapping')
    op = value.get('operation')
    if not isinstance(op, str) or op not in FIELDS:
        raise UnsupportedEditError('supported command vocabulary: ' + ', '.join(FIELDS))
    common = {'schema', 'operation', 'object_id', 'mode'}
    optional = {'expected_revision', 'expected_generation'}
    if not (common | FIELDS[op]) <= set(value) or set(value) - common - FIELDS[op] - optional:
        raise ValueError('missing or unknown edit fields')
    if value['schema'] != 'scene_edit_v1' or value['mode'] != 'diagnostic_preview':
        raise ValueError('scene_edit_v1 requires diagnostic_preview')
    oid = value['object_id']
    if not isinstance(oid, str):
        raise ValueError('object_id must be a string')
    native = re.fullmatch(r'native:(RigidNodes|SMPLNodes|DeformableNodes|TrafficLightNodes):(0|[1-9][0-9]*)', oid)
    managed = re.fullmatch(r'managed:([A-Za-z][A-Za-z0-9_-]{0,63})', oid)
    signal=re.fullmatch(r'signal:([A-Za-z][A-Za-z0-9_-]{0,63})',oid)
    if not native and not managed and not signal:
        raise ValueError('expected native:GROUP:MODEL_INDEX or managed:NAME')
    if op in ACTOR_OPERATIONS:
        if not native or native.group(1)!='SMPLNodes':raise UnsupportedEditError('actor trajectories currently require native SMPLNodes')
        if op=='set_actor_trajectory':validate_path(value['trajectory'])
    elif signal or (native and native.group(1)=='TrafficLightNodes'):
        if op not in SIGNAL_OPERATIONS:raise UnsupportedEditError('traffic lights support signal operations only')
        validate_signal(value)
    elif op in SIGNAL_OPERATIONS:raise UnsupportedEditError('signal operations require a native TrafficLightNodes target')
    elif (op in ('translate', 'rotate')) != bool(native):
        raise UnsupportedEditError('native objects support translate/rigid rotate; managed objects use insert/set_pose/remove')
    for name in ('expected_revision',):
        if name in value and (type(value[name]) is not int or value[name] < 0):
            raise ValueError('expected_revision must be a nonnegative integer')
    if 'expected_generation' in value and (
            not isinstance(value['expected_generation'], str) or not value['expected_generation']):
        raise ValueError('expected_generation must be a nonempty string')
    if op == 'rotate':
        if native.group(1) != 'RigidNodes':
            raise UnsupportedEditError('native rotation supports RigidNodes only')
        yaw = value['yaw_degrees']
        if (not isinstance(yaw, Real) or isinstance(yaw, (bool, np.bool_))
                or not np.isfinite(yaw) or abs(yaw) > 45):
            raise ValueError('diagnostic yaw must be finite and <=45 degrees')
    if op == 'translate':
        offset = value['offset_world_m']
        if not isinstance(offset, (list, tuple)) or len(offset) != 3 or any(
                not isinstance(x, Real) or isinstance(x, (bool, np.bool_)) for x in offset):
            raise ValueError('offset_world_m must contain three real numbers')
        if not np.isfinite(offset).all() or np.linalg.norm(offset) > 2:
            raise ValueError('diagnostic translation must be finite and <=2m')
    # Store independent JSON data; callers cannot later mutate the journal.
    return _hugsim_expand(json.loads(json.dumps(value, allow_nan=False)))


def native_request(command):
    command = read_command(command)
    if command['operation'] not in ('translate', 'rotate'):
        raise UnsupportedEditError('not a native edit command')
    _, group, index = command['object_id'].split(':')
    return dict(schema='framework_native_edit_v1', operation=command['operation'], group=group,
                instance=int(index), **{k: command[k] for k in FIELDS[command['operation']]})


def from_native_request(value):
    if not isinstance(value, dict) or set(value) != {
            'schema', 'operation', 'group', 'instance', 'offset_world_m'}:
        raise ValueError('invalid legacy native edit fields')
    if value['schema'] != 'framework_native_edit_v1' or value['operation'] != 'translate' or type(value['instance']) is not int:
        raise ValueError('invalid legacy native translation')
    return read_command(dict(schema='scene_edit_v1', operation='translate',
        object_id=f"native:{value['group']}:{value['instance']}", mode='diagnostic_preview',
        offset_world_m=value['offset_world_m']))


class SceneEditor:
    def __init__(self, session, managed_edit):
        self.session = session
        self.backend = session.backend
        self.generation = session.generation
        self.managed_edit = managed_edit
        self.framework = hasattr(self.backend, 'trainer')
        self.managed_supported = not self.framework or getattr(self.backend, 'supports_managed_assets', False)
        self.started = False
        self.closed = False
        self.journal = []
        self.originals = {}
        self.offsets = {}
        self.original_quaternions = {}
        self.yaws = {}
        self.original_radiance = {}
        self.radiance_tensors = {}
        self.traffic=None
        if self.framework and 'TrafficLightNodes' in self.backend.trainer.models:
            from traffic_lights import TrafficLights
            self.traffic=TrafficLights(self.backend,self.session.spec)
        self.actor_paths=None
        smpl=getattr(getattr(self.backend,'trainer',None),'models',{}).get('SMPLNodes')
        if self.framework and smpl is not None and callable(getattr(smpl,'get_gaussians',None)) and hasattr(smpl,'point_ids'):
            from actor_trajectories import ActorTrajectories
            self.actor_paths=ActorTrajectories(self.backend,self.session.spec)

    def require_current(self):
        if (self.closed or self.session.backend is not self.backend or
                self.session.generation != self.generation):
            raise RuntimeError('editor belongs to a closed or replaced scene generation')

    def capabilities(self):
        self.require_current()
        return dict(schema='scene_edit_capabilities_v1', generation=self.generation,
            edit_revision=self.session.revision, phase='observing' if self.started else 'pre_run',
            native_translate_groups=list(NATIVE_GROUPS) if self.framework else [],
            actor_trajectory_groups=['SMPLNodes'] if self.actor_paths else [],
            actor_trajectory_operations=list(ACTOR_OPERATIONS) if self.actor_paths else [],
            actor_motion_profiles=['legacy','smooth_v1'] if self.actor_paths else [],
            actor_heading_modes=['waypoint_delta','path_tangent'] if self.actor_paths else [],
            traffic_light_operations=list(SIGNAL_OPERATIONS) if self.traffic else [],
            managed_operations=['insert', 'set_pose', 'remove'] if self.managed_supported else [],
            runtime_edit=False, native_insert=False, native_remove=False, native_rotate=self.framework,
            native_rotate_groups=['RigidNodes'] if self.framework else [],
            rotation_axis='world_z', rotation_pivot='per_frame_root_center',
            max_total_yaw_degrees=45., native_rotation_gpu_verified=False,
            world_frame='first_waymo_ego_z_up_meters', max_total_translation_m=2.,
            unified_interface_gpu_verified=False, formal_evaluation_admitted=False)

    def objects(self):
        self.require_current()
        result = []
        for group, state in getattr(getattr(self.backend, 'env', None), 'models', {}).items():
            if group not in NATIVE_GROUPS or 'instances_trans' not in state:
                continue
            for index in range(state['instances_trans'].shape[1]):
                result.append(dict(object_id=f'native:{group}:{index}', kind='native',
                    group=group, model_index=index, generation=self.generation,
                    operations=((['translate','rotate'] if group=='RigidNodes' else ['translate'])+ (list(ACTOR_OPERATIONS) if group=='SMPLNodes' and self.actor_paths else [])) if self.framework else [],
                    yaw_world_degrees=self.yaws.get((group, index), 0.),
                    translation_world_m=self.offsets.get((group, index), [0., 0., 0.])))
        if self.traffic:
            for row in self.traffic.objects():result.append(dict(row,generation=self.generation))
        for name in self.session.objects:
            result.append(dict(object_id='managed:' + name, kind='managed',
                generation=self.generation, operations=['set_pose', 'remove'] if self.managed_supported else []))
        return copy.deepcopy(result)

    def edit(self, value):
        self.require_current()
        command = read_command(value)
        if self.started or getattr(self.backend, '_last', None) is not None:
            raise RuntimeError('v1 edits are pre-run only; reset before editing again')
        if command.get('expected_revision', self.session.revision) != self.session.revision:
            raise ValueError('stale edit revision')
        if command.get('expected_generation', self.generation) != self.generation:
            raise ValueError('stale scene generation')
        previous_revision = self.session.revision
        if command['operation'] in ACTOR_OPERATIONS:
            if self.actor_paths is None:raise UnsupportedEditError('native actor trajectories unavailable')
            evidence=self.actor_paths.apply(command);self.session.revision+=1
        elif command['operation'] in SIGNAL_OPERATIONS:
            if self.traffic is None:raise UnsupportedEditError('traffic lights unavailable in this backend')
            evidence=self.traffic.apply(command);self.session.revision+=1
        elif command['operation'] in ('translate', 'rotate'):
            if not self.framework:
                raise UnsupportedEditError('native translation requires framework backend')
            from native_edit import translate_native, rotate_native
            request = native_request(command)
            group, index = request['group'], request['instance']
            if group=='SMPLNodes' and self.actor_paths and index in self.actor_paths.programs:
                raise UnsupportedEditError('restore actor trajectory before translating the same actor')
            model = self.backend.trainer.models.get(group)
            if model is None or not 0 <= index < model.instances_trans.shape[1]:
                raise ValueError('native object does not exist in this scene')
            key = (group, index)
            rotate = command['operation'] == 'rotate'
            if rotate:
                total = self.yaws.get(key, 0.) + request['yaw_degrees']
                if abs(total) > 45:
                    raise ValueError('cumulative diagnostic yaw exceeds 45 degrees')
            else:
                total = np.asarray(self.offsets.get(key, [0., 0., 0.])) + request['offset_world_m']
                if np.linalg.norm(total) > 2:
                    raise ValueError('cumulative diagnostic translation exceeds 2m')
            tensor = model.instances_quats if rotate else model.instances_trans
            before = tensor[:, index].detach().clone()
            from native_radiance import radiance_state, restore_radiance
            before_radiance = radiance_state(model, index) if rotate else None
            try:
                evidence = (rotate_native if rotate else translate_native)(self.backend, request)
                clear = getattr(self.backend, 'clear_observation_history', self.backend.reset)
                clear()
            except Exception:
                with self.backend.torch.no_grad():
                    tensor[:, index].copy_(before)
                if rotate:
                    restore_radiance(self.backend.torch, model, before_radiance)
                raise
            if rotate:
                self.original_quaternions.setdefault(key, before)
                if before_radiance is not None:
                    self.original_radiance.setdefault(key, before_radiance)
                    self.radiance_tensors.setdefault(key, model._features_rest)
                self.yaws[key] = float(total)
            else:
                self.originals.setdefault(key, before)
                self.offsets[key] = total.tolist()
            self.session.revision += 1
        else:
            if not self.managed_supported:
                raise UnsupportedEditError('managed assets are not yet supported by framework backend')
            legacy = {k: v for k, v in command.items() if k not in (
                'schema', 'expected_revision', 'expected_generation')}
            legacy['object_id'] = command['object_id'].split(':', 1)[1]
            evidence = self.managed_edit(legacy)
        receipt = dict(schema='scene_edit_receipt_v1', command=command,
            generation=self.generation, previous_revision=previous_revision,
            edit_revision=self.session.revision, evidence=evidence,
            status='diagnostic_unverified', formal_evaluation_admitted=False)
        self.journal.append(copy.deepcopy(receipt))
        return copy.deepcopy(receipt)

    def restore_native(self):
        self.require_current()
        if self.actor_paths:self.actor_paths.clear()
        if self.traffic:self.traffic.clear()
        states = [('instances_trans', self.originals),
                  ('instances_quats', self.original_quaternions)]
        # Validate every shared tensor before restoring any of them.
        for name, originals in states:
            for group, index in originals:
                if self.backend.env.models[group][name] is not getattr(self.backend.trainer.models[group], name):
                    raise RuntimeError('proxy identity changed; cannot restore scene safely')
        for (group, index), tensor in self.radiance_tensors.items():
            if self.backend.trainer.models[group]._features_rest is not tensor:
                raise RuntimeError('SH parameter identity changed; cannot restore safely')
        with self.backend.torch.no_grad() if self.framework else _null_context():
            for name, originals in states:
                for (group, index), value in originals.items():
                    getattr(self.backend.trainer.models[group], name)[:, index].copy_(value)
            from native_radiance import restore_radiance
            for (group, index), state in self.original_radiance.items():
                restore_radiance(self.backend.torch, self.backend.trainer.models[group], state)


def _null_context():
    from contextlib import nullcontext
    return nullcontext()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CPU command validation; no scene/model is loaded')
    parser.add_argument('command', type=Path)
    args = parser.parse_args()
    value = read_command(_hugsim_expand(json.loads(args.command.read_text())))
    print(json.dumps(dict(valid=True, command=value, target_exists_checked=False,
        backend_capability_checked=False, formal_evaluation_admitted=False), indent=2))
