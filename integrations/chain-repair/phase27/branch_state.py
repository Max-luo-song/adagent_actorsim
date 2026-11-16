"""Closed-loop branch state snapshot/restore (phase27 mechanism experiment).

DIAGNOSTIC ONLY: N1 suspended, formal_evaluation_admitted=false. This module
never modifies the main repo, HUGSIM or UniAD; it serializes/restores mutable
state of the *running* closed loop so isolated branches can fork from one
shared state with the driving model fixed.

State contract (02_状态清单与修改位置.md):
  A1 prev_bev, A2 prev_frame_info (keeping `is` identity with prev_bev),
  A3 test_track_instances, A4 module timestamp/scene_token/l2g cache,
  A5 scene_reliability, A6 geometry_refinement_memory, A7 adapter tokens,
  A8 worker loop index/previous_time, A9 RNG (random/numpy/torch/cuda).
Driver state B1-B10 is pickled separately (branch_driver_state_v2).

Immutable scene checkpoints, assets and model weights are shared, never copied.
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

import hashlib
import pickle
import time
pass  # Path is provided by the portable boundary above

import numpy as np

WORKER_SCHEMA = 'branch_worker_state_v1'
DRIVER_SCHEMA = 'branch_driver_state_v2'


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _encode_value(value):
    """Encode one python/tensor value into a serializable record + audit meta."""
    import torch  # local import keeps CPU tests free of CUDA assumptions
    if torch.is_tensor(value):
        device = str(value.device)
        dtype = str(value.dtype)
        tensor = value.detach()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)  # bf16 is a subset of fp32: lossless
        arr = tensor.cpu().numpy()
        return ({'kind': 'tensor', 'data': arr, 'device': device, 'dtype': dtype},
                {'kind': 'tensor', 'shape': list(arr.shape), 'dtype': dtype,
                 'device': device, 'sha256': _sha256_bytes(arr.tobytes()),
                 'nbytes': int(arr.nbytes)})
    if isinstance(value, np.ndarray):
        return ({'kind': 'ndarray', 'data': value},
                {'kind': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype),
                 'sha256': _sha256_bytes(value.tobytes()), 'nbytes': int(value.nbytes)})
    if value is None or isinstance(value, (bool, int, float, str)):
        return {'kind': 'scalar', 'data': value}, {'kind': 'scalar', 'type': type(value).__name__}
    blob = pickle.dumps(value, protocol=4)
    return ({'kind': 'pickle', 'data': blob},
            {'kind': 'pickle', 'type': type(value).__name__,
             'sha256': _sha256_bytes(blob), 'nbytes': len(blob)})


def _decode_value(record):
    import torch
    kind = record['kind']
    if kind == 'tensor':
        tensor = torch.from_numpy(np.ascontiguousarray(record['data']))
        if record.get('dtype') == 'torch.bfloat16':
            tensor = tensor.to(torch.bfloat16)
        if str(record.get('device', '')).startswith('cuda'):
            tensor = tensor.cuda()
        return tensor
    if kind == 'ndarray':
        return record['data']
    if kind == 'scalar':
        return record['data']
    if kind == 'pickle':
        return pickle.loads(record['data'])
    raise ValueError('Unknown record kind: ' + str(kind))


def _encode_instances(instances):
    """UniAD track Instances -> plain dict of encoded fields (no class pickling)."""
    if instances is None:
        return None, {'kind': 'none'}
    encoded, meta = {}, {}
    for name, value in instances.get_fields().items():
        record, m = _encode_value(value)
        encoded[name] = record
        meta[name] = m
    return ({'image_size': tuple(instances.image_size), 'fields': encoded},
            {'kind': 'instances', 'fields': meta, 'count': len(instances)})


def _decode_instances(payload):
    if payload is None:
        return None
    # Import through the same module path the running model uses, so the
    # restored object shares the model's actual class (isinstance-safe).
    try:
        from projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import Instances
    except ImportError:
        from mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import Instances
    inst = Instances(tuple(payload['image_size']))
    for name, record in payload['fields'].items():
        inst.set(name, _decode_value(record))
    return inst


def capture_rng():
    import random
    import torch
    state = {'python': pickle.dumps(random.getstate(), protocol=4),
             'numpy': pickle.dumps(np.random.get_state(), protocol=4),
             'torch_cpu': torch.get_rng_state().cpu().numpy()}
    if torch.cuda.is_available():
        state['torch_cuda'] = [t.cpu().numpy() for t in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    import random
    import torch
    random.setstate(pickle.loads(state['python']))
    np.random.set_state(pickle.loads(state['numpy']))
    torch.set_rng_state(torch.from_numpy(np.ascontiguousarray(state['torch_cpu'])))
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.from_numpy(np.ascontiguousarray(t)) for t in state['torch_cuda']])


def _reliability_state(reliability):
    return {'scene_token': reliability.scene_token,
            'previous_time': reliability.previous_time,
            'tracks': reliability.tracks,
            'next_id': reliability.next_id}


def _gate_state(gate):
    if gate is None:
        return None
    return {'anchors': gate.anchors, 'scene': gate.scene, 'time': gate.time,
            'previous': getattr(gate, 'previous', [])}


def capture_worker_state(core, *, index, previous_time):
    """Capture all mutable UniAD worker state (A1-A9). Runs inside the worker."""
    payload, meta = {}, {}

    payload['prev_bev'], meta['prev_bev'] = _encode_value(core.prev_bev)

    pfi = core.prev_frame_info
    payload['prev_frame_info'], meta['prev_frame_info'] = {}, {}
    for key in ('scene_token', 'prev_pos', 'prev_angle'):
        payload['prev_frame_info'][key], meta['prev_frame_info'][key] = _encode_value(pfi.get(key))
    meta['prev_frame_info']['prev_bev_alias'] = bool(pfi.get('prev_bev') is core.prev_bev)

    payload['test_track_instances'], meta['test_track_instances'] = \
        _encode_instances(core.test_track_instances)

    payload['module_cache'], meta['module_cache'] = {}, {}
    for key in ('timestamp', 'scene_token', 'l2g_t', 'l2g_r_mat'):
        payload['module_cache'][key], meta['module_cache'][key] = \
            _encode_value(getattr(core, key, None))

    payload['adapter'] = {'scene_token': getattr(core, '_adapter_scene_token', None),
                          'timestamp': getattr(core, '_adapter_timestamp', None)}
    meta['adapter'] = {'kind': 'scalars'}

    rel_blob = pickle.dumps(_reliability_state(core.scene_reliability), protocol=4)
    payload['scene_reliability'] = rel_blob
    meta['scene_reliability'] = {'kind': 'pickle', 'sha256': _sha256_bytes(rel_blob),
                                 'nbytes': len(rel_blob)}

    gate = getattr(core, 'geometry_refinement_memory', None)
    gate_blob = pickle.dumps(_gate_state(gate), protocol=4)
    payload['geometry_refinement_memory'] = gate_blob
    meta['geometry_refinement_memory'] = {'kind': 'pickle', 'sha256': _sha256_bytes(gate_blob),
                                          'nbytes': len(gate_blob)}

    payload['rng'] = capture_rng()
    meta['rng'] = {'kind': 'rng_state'}
    payload['loop'] = {'index': int(index),
                       'previous_time': previous_time}
    return payload, meta


def apply_worker_state(core, payload):
    """Restore worker state captured by capture_worker_state. Identity rules:
    core.prev_frame_info['prev_bev'] must be the SAME object as core.prev_bev
    (the per-step temporal_state.json audit asserts this identity)."""
    core.prev_bev = _decode_value(payload['prev_bev'])
    pfi = payload['prev_frame_info']
    core.prev_frame_info['prev_bev'] = core.prev_bev  # re-alias, not a copy
    for key in ('scene_token', 'prev_pos', 'prev_angle'):
        core.prev_frame_info[key] = _decode_value(pfi[key])

    core.test_track_instances = _decode_instances(payload['test_track_instances'])

    for key, record in payload['module_cache'].items():
        setattr(core, key, _decode_value(record))

    adapter = payload['adapter']
    core._adapter_scene_token = adapter['scene_token']
    core._adapter_timestamp = adapter['timestamp']

    rel = pickle.loads(payload['scene_reliability'])
    reliability = core.scene_reliability
    reliability.scene_token = rel['scene_token']
    reliability.previous_time = rel['previous_time']
    reliability.tracks = rel['tracks']
    reliability.next_id = rel['next_id']

    gate = pickle.loads(payload['geometry_refinement_memory'])
    if gate is not None:
        from pseudo_lidar_adapter import TemporalRefinementGate
        memory = TemporalRefinementGate()
        memory.anchors = gate['anchors']
        memory.scene = gate['scene']
        memory.time = gate['time']
        memory.previous = gate.get('previous', [])
        core.geometry_refinement_memory = memory
    elif hasattr(core, 'geometry_refinement_memory'):
        delattr(core, 'geometry_refinement_memory')

    restore_rng(payload['rng'])
    return dict(payload['loop'])


def save_worker_snapshot(directory, core, *, index, previous_time, extra=None):
    """Write worker snapshot to directory/worker_snapshot.pt + .json (atomic)."""
    import torch
    began = time.monotonic()
    payload, meta = capture_worker_state(core, index=index, previous_time=previous_time)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'worker_snapshot.pt'
    tmp = directory / 'worker_snapshot.pt.tmp'
    torch.save(payload, tmp)
    tmp.replace(target)
    total_bytes = target.stat().st_size
    gpu = {}
    if torch.cuda.is_available():
        gpu = {'memory_allocated_bytes': int(torch.cuda.memory_allocated()),
               'max_memory_allocated_bytes': int(torch.cuda.max_memory_allocated())}
    document = {'schema': WORKER_SCHEMA, 'created_after_index': int(index) - 1,
                'next_index': int(index), 'previous_time': previous_time,
                'fields': meta, 'snapshot_bytes': total_bytes,
                'capture_wall_seconds': time.monotonic() - began,
                'gpu_memory': gpu}
    if extra:
        document.update(extra)
    _atomic_json(directory / 'worker_snapshot.json', document)
    return document


def load_worker_snapshot(directory):
    import torch
    directory = Path(directory)
    document = json_load(directory / 'worker_snapshot.json')
    if document.get('schema') != WORKER_SCHEMA:
        raise ValueError('Unexpected worker snapshot schema')
    try:
        payload = torch.load(directory / 'worker_snapshot.pt',
                             map_location='cpu', weights_only=False)
    except TypeError:
        # Legacy torch (UniAD worker env) has no weights_only kwarg; the
        # snapshot is a trusted local artifact produced by our own worker.
        payload = torch.load(directory / 'worker_snapshot.pt', map_location='cpu')
    return document, payload


def _atomic_json(path, data):
    import json
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def json_load(path):
    import json
    return _hugsim_expand(json.loads(Path(path).read_text()))


def save_driver_snapshot(path, **fields):
    """Pickle the harness-side closed-loop state (B1-B10). Keyword payload must
    include next_index/token/motion/warm/controller/obstacle_memory/depth_memory/
    road_memory/navigation_anchor/committed_route/mode/report/minimum_gap/rng/
    session/request/experiment_signature/implementation."""
    payload = dict(fields)
    payload['schema'] = DRIVER_SCHEMA
    path = Path(path)
    tmp = path.with_suffix('.pkl.tmp')
    with tmp.open('wb') as handle:
        pickle.dump(payload, handle, protocol=4)
    tmp.replace(path)
    return path.stat().st_size


def load_driver_snapshot(path, *, implementation, experiment_signature):
    path = Path(path).resolve()
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if payload.get('schema') != DRIVER_SCHEMA:
        raise ValueError('Unexpected driver snapshot schema')
    if payload.get('implementation') != implementation:
        raise ValueError('Branch implementation fingerprint differs; refusing to mix code versions')
    if payload.get('experiment_signature') != experiment_signature:
        raise ValueError('Branch experiment signature differs; refusing to mix inputs')
    return payload


def verify_worker_snapshot_hashes(document, payload):
    """Re-hash encoded tensor/ndarray fields and compare to the snapshot meta."""
    mismatches = []
    def check(meta, record, name):
        if not isinstance(meta, dict) or meta.get('kind') not in ('tensor', 'ndarray'):
            return
        arr = record['data']
        if _sha256_bytes(arr.tobytes()) != meta['sha256']:
            mismatches.append(name)
    check(document['fields'].get('prev_bev'), payload['prev_bev'], 'prev_bev')
    for key, meta in document['fields'].get('prev_frame_info', {}).items():
        if key in payload['prev_frame_info']:
            check(meta, payload['prev_frame_info'][key], 'prev_frame_info.' + key)
    inst_meta = document['fields'].get('test_track_instances', {})
    if inst_meta.get('kind') == 'instances':
        for name, meta in inst_meta.get('fields', {}).items():
            check(meta, payload['test_track_instances']['fields'][name],
                  'test_track_instances.' + name)
    for key, meta in document['fields'].get('module_cache', {}).items():
        check(meta, payload['module_cache'][key], 'module_cache.' + key)
    return mismatches
