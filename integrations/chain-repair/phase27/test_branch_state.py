"""CPU tests for the phase27 branch snapshot/restore contract. No GPU, no scene."""

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

import os
import pickle
import sys
import tempfile
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
os.environ.setdefault('BASE', str(_REPO))
sys.path[:0] = [str(_HERE), str(_REPO / 'waymo23-native-import')]

import branch_state

RESULTS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        RESULTS.append((name, False, repr(exc)))
        print(f'FAIL {name}: {exc!r}')
    else:
        RESULTS.append((name, True, None))
        print(f'pass {name}')


def test_tensor_roundtrip():
    import torch
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    record, meta = branch_state._encode_value(value)
    assert meta['kind'] == 'tensor' and meta['shape'] == [2, 3, 4]
    back = branch_state._decode_value(record)
    assert torch.equal(back, value)


def test_bfloat16_roundtrip():
    import torch
    value = torch.randn(8, dtype=torch.bfloat16)
    record, meta = branch_state._encode_value(value)
    assert meta['dtype'] == 'torch.bfloat16' and record['data'].dtype == np.float32
    back = branch_state._decode_value(record)
    assert back.dtype == torch.bfloat16 and torch.equal(back, value)


def test_scalar_and_pickle_roundtrip():
    for value in (None, True, 7, 1.25, 'token-xyz'):
        record, _ = branch_state._encode_value(value)
        assert branch_state._decode_value(record) == value
    value = {'anchors': [{'box': np.ones(3), 'time': 0.5}], 'scene': 's'}
    record, meta = branch_state._encode_value(value)
    assert meta['kind'] == 'pickle'
    back = branch_state._decode_value(record)
    assert back['scene'] == 's' and np.array_equal(back['anchors'][0]['box'], np.ones(3))


class _StubInstances:
    def __init__(self, image_size):
        self._image_size = tuple(image_size)
        self._fields = {}

    @property
    def image_size(self):
        return self._image_size

    def get_fields(self):
        return self._fields

    def set(self, name, value):
        if self._fields:
            assert len(value) == len(self)
        self._fields[name] = value

    def __len__(self):
        return len(next(iter(self._fields.values()))) if self._fields else 0


def test_instances_roundtrip():
    import torch
    inst = _StubInstances((1, 1))
    inst.set('query', torch.randn(5, 256))
    inst.set('track_id', torch.arange(5))
    inst.set('mem_bank', torch.randn(5, 4, 256))
    encoded, meta = branch_state._encode_instances(inst)
    assert meta['kind'] == 'instances' and meta['count'] == 5
    original_import = branch_state._decode_instances

    def decode_with_stub(payload):
        out = _StubInstances(payload['image_size'])
        for name, record in payload['fields'].items():
            out.set(name, branch_state._decode_value(record))
        return out

    back = decode_with_stub(encoded)
    assert back.image_size == (1, 1)
    assert torch.equal(back.get_fields()['query'], inst.get_fields()['query'])
    assert torch.equal(back.get_fields()['track_id'], inst.get_fields()['track_id'])
    # Real UniAD Instances decode path runs in the GPU worker; covered there.


def test_rng_roundtrip():
    import random
    import torch
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    state = branch_state.capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(4))
    random.seed(999); np.random.seed(999); torch.manual_seed(999)
    branch_state.restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def _fake_core():
    import torch
    reliability = SimpleNamespace(scene_token='tok', previous_time=0.5,
                                  tracks=[{'id': 1, 'box': np.zeros(9)}], next_id=2)
    core = SimpleNamespace()
    core.prev_bev = torch.randn(10, 1, 8)
    core.prev_frame_info = {'prev_bev': core.prev_bev, 'scene_token': 'tok',
                            'prev_pos': np.array([1., 2., 0.]), 'prev_angle': 0.3}
    core.test_track_instances = None
    core.timestamp = 0.5
    core.scene_token = 'tok'
    core.l2g_t = np.zeros(3)
    core.l2g_r_mat = np.eye(3)
    core._adapter_scene_token = 'tok'
    core._adapter_timestamp = 0.5
    core.scene_reliability = reliability
    return core


def test_worker_state_roundtrip_identity():
    import torch
    core = _fake_core()
    payload, meta = branch_state.capture_worker_state(core, index=6, previous_time=0.5)
    assert meta['prev_frame_info']['prev_bev_alias'] is True
    # Mutate everything, then restore.
    core.prev_bev = None
    core.prev_frame_info.update(prev_bev=None, scene_token='other', prev_angle=9.)
    core.timestamp = None
    core._adapter_timestamp = None
    core.scene_reliability.tracks = []
    loop = branch_state.apply_worker_state(core, payload)
    assert loop == {'index': 6, 'previous_time': 0.5}
    assert core.prev_frame_info['prev_bev'] is core.prev_bev  # identity contract
    assert core.prev_frame_info['scene_token'] == 'tok'
    assert core.prev_frame_info['prev_angle'] == 0.3
    assert core.timestamp == 0.5 and core._adapter_timestamp == 0.5
    assert core.scene_reliability.next_id == 2 and len(core.scene_reliability.tracks) == 1
    assert core.test_track_instances is None


def test_worker_snapshot_files_and_tamper_detection():
    with tempfile.TemporaryDirectory() as tmp:
        core = _fake_core()
        document = branch_state.save_worker_snapshot(tmp, core, index=6, previous_time=0.5)
        assert document['schema'] == branch_state.WORKER_SCHEMA
        assert document['next_index'] == 6 and document['created_after_index'] == 5
        loaded_doc, payload = branch_state.load_worker_snapshot(tmp)
        assert branch_state.verify_worker_snapshot_hashes(loaded_doc, payload) == []
        payload['prev_bev']['data'][0, 0, 0] += 1.0  # tamper
        mismatches = branch_state.verify_worker_snapshot_hashes(loaded_doc, payload)
        assert 'prev_bev' in mismatches


def test_driver_snapshot_roundtrip():
    from live_env import Motion
    from planning_guard import ObstacleMemory, RecoveryController
    motion = Motion(dt=.1, wheelbase=2.7, start_speed=2.)
    motion.advance(dict(acc=0.1, steer_rate=0.01))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'driver_snapshot.pkl'
        implementation = {'a.py': 'x' * 64}
        branch_state.save_driver_snapshot(path,
            next_index=6, token='tok-1', motion=dict(vars(motion)), warm=np.zeros((6, 2)),
            controller=RecoveryController(), obstacle_memory=ObstacleMemory(),
            depth_memory={'stub': True}, road_memory={'stub': [1, 2]},
            navigation_memory=None, navigation_anchor=SimpleNamespace(points=np.ones((3, 3))),
            committed_route=SimpleNamespace(world=None), mode='normal / MPC',
            report={'trace': [{'step': 5}]}, minimum_gap=3.25,
            rng=branch_state.capture_rng(), session={'revision': 0}, edits=[],
            request={'schema': 'unified_three_closed_dynamic_v1'},
            experiment_signature='sig-1', implementation=implementation)
        payload = branch_state.load_driver_snapshot(path, implementation=implementation,
                                                    experiment_signature='sig-1')
        assert payload['next_index'] == 6 and payload['token'] == 'tok-1'
        assert payload['motion']['speed'] == motion.speed
        restored = Motion()
        restored.__dict__.update(payload['motion'])
        assert vars(restored) == vars(motion)
        try:
            branch_state.load_driver_snapshot(path, implementation={'a.py': 'y' * 64},
                                              experiment_signature='sig-1')
        except ValueError:
            pass
        else:
            raise AssertionError('implementation mismatch must be rejected')
        try:
            branch_state.load_driver_snapshot(path, implementation=implementation,
                                              experiment_signature='sig-2')
        except ValueError:
            pass
        else:
            raise AssertionError('experiment signature mismatch must be rejected')


def test_branch_worker_importable():
    import importlib.util
    spec = importlib.util.spec_from_file_location('branch_worker', _HERE / 'branch_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # heavy imports live inside serve()/main()
    assert hasattr(module, 'serve') and hasattr(module, 'main')


def test_driver_state_classes_pickle():
    sys.path.insert(0, str(_HERE.parent / 'phase11'))
    from navigation_anchor import NavigationAnchor  # phase11 reviewed copies
    from committed_route import CommittedRoute
    from road_prediction_memory import RoadPredictionMemory
    vehicle = SimpleNamespace(dt=.1, wheelbase_m=2.7, half_length_m=2.3,
                              emergency_braking_mps2=4.)
    for obj in (NavigationAnchor(2.3), CommittedRoute(), RoadPredictionMemory(vehicle)):
        blob = pickle.dumps(obj, protocol=4)
        pickle.loads(blob)


def main():
    check('tensor_roundtrip', test_tensor_roundtrip)
    check('bfloat16_roundtrip', test_bfloat16_roundtrip)
    check('scalar_and_pickle_roundtrip', test_scalar_and_pickle_roundtrip)
    check('instances_roundtrip', test_instances_roundtrip)
    check('rng_roundtrip', test_rng_roundtrip)
    check('worker_state_roundtrip_identity', test_worker_state_roundtrip_identity)
    check('worker_snapshot_files_and_tamper_detection', test_worker_snapshot_files_and_tamper_detection)
    check('driver_snapshot_roundtrip', test_driver_snapshot_roundtrip)
    check('branch_worker_importable', test_branch_worker_importable)
    check('driver_state_classes_pickle', test_driver_state_classes_pickle)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f'\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed')
    if failed:
        print('FAILED: ' + ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
