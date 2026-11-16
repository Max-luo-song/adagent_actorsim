"""Build and enqueue phase-26 stage-1 GPU requests (diagnostic, uncertified).

Jobs (gpu_test_service allowlisted `render_scene` dispatch only):
1. srveh552-smoke-20261027-v1: unified interface smoke (insert/move/remove/
   reset/step) of the exported RigidNodes asset on scene 552 at the existing
   sensor-audit anchor (frame 60).
2. srveh552-closed-realcar-20261027-v1: diagnostic closed loop, OLD 3DRealCar
   asset, same anchor/steps as the exported-asset arm.
3. srveh552-closed-rigid25-20261027-v1: same closed loop with the exported
   RigidNodes vehicle (true size).
4. srveh552-closed-rigid25sz-20261027-v1: matched-size ablation arm (labelled;
   support wlh matched to the old asset because length/height differ >10%).

All arms share scene manifest, start frame, steps, dt, seed, guard policy and
the world anchor matrix; only the asset differs. Results are diagnostic,
uncertified, and never enter formal evaluation records.
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

import json
import shutil
import sys
pass  # Path is provided by the portable boundary above

BASE = Path(_hugsim_expand('${HUGSIM_ROOT}'))
BRIDGE = Path(__file__).resolve().parent
REQ = BRIDGE/'requests'
sys.path.insert(0, str(BASE/'waymo23-native-import'))
from gpu_test_service import enqueue  # noqa: E402

OLD_ASSET = BASE/'data/3DRealCar/2024_06_04_13_44_39'
OLD_CAL = BASE/'waymo23-native-import/configs/asset_calibrations/2024_06_04_13_44_39-support-v1.json'
NEW_ASSET = BRIDGE/'assets/waymo552_rigid_id25_v1'
ABL_ASSET = BRIDGE/'assets/waymo552_rigid_id25_v1_matched_size_ablation'
AUDIT_552 = BASE/'outputs/n1_cross_scene_audit_20261020_015102/source_01/audit_report.json'
MANIFEST_552 = BASE/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase4/dynamic552_frame80/scene_manifest.json'
PHASE4_INSERT = BASE/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase4/dynamic552_frame80/insert_request.json'


def write(name, value):
    path = REQ/name
    path.write_text(json.dumps(value, indent=2))
    return path


def insert_request(asset, cal, anchor_pose):
    return dict(operation='insert', object_id='preview_car', mode='diagnostic_preview',
                asset=str(asset), asset_calibration=str(cal), asset_to_world=anchor_pose)


def main():
    REQ.mkdir(exist_ok=True)
    for p in (OLD_CAL, AUDIT_552, MANIFEST_552, PHASE4_INSERT,
              NEW_ASSET/'gs.pth', NEW_ASSET/'asset_calibration.json',
              ABL_ASSET/'gs.pth', ABL_ASSET/'asset_calibration.json'):
        if not Path(p).is_file():
            raise FileNotFoundError(str(p))
    anchor = _hugsim_expand(json.loads(PHASE4_INSERT.read_text()))['asset_to_world']

    # The smoke validates the audit source_report against the manifest's
    # source_report, so it must use the original 552 manifest (render_scale
    # 0.25), not the phase4 closed-loop manifest.
    smoke = write('smoke552.json', dict(
        schema='unified_scene_smoke_v1',
        cases=[dict(scene_manifest=str(BASE/'integrations/sr-hugsim/configs/waymo552.json'),
                    sensor_audit=str(AUDIT_552))],
        asset=str(NEW_ASSET), asset_calibration=str(NEW_ASSET/'asset_calibration.json')))
    smoke_dispatch = write('smoke552_dispatch.json', {'unified_interface_probe': str(smoke)})

    arms = [('realcar', OLD_ASSET, OLD_CAL), ('rigid25', NEW_ASSET, NEW_ASSET/'asset_calibration.json'),
            ('rigid25sz', ABL_ASSET, ABL_ASSET/'asset_calibration.json')]
    closed_dispatch = {}
    for name, asset, cal in arms:
        ins = write(f'closed552_insert_{name}.json', insert_request(asset, cal, anchor))
        req = write(f'closed552_{name}.json', dict(
            schema='unified_three_closed_dynamic_v1', scene_manifest=str(MANIFEST_552),
            insert_request=str(ins), start_frame=80, steps=117,
            time_policy='source_time_only'))
        closed_dispatch[name] = write(f'closed552_{name}_dispatch.json', {'unified_open_closed_probe': str(req)})

    jobs = [('srveh552-smoke-20261027-v1', smoke_dispatch),
            ('srveh552-closed-realcar-20261027-v1', closed_dispatch['realcar']),
            ('srveh552-closed-rigid25-20261027-v1', closed_dispatch['rigid25']),
            ('srveh552-closed-rigid25sz-20261027-v1', closed_dispatch['rigid25sz'])]
    for identity, dispatch in jobs:
        try:
            path = enqueue(identity, 'render_scene', dict(source_report=str(dispatch)))
            print('ENQUEUED', identity, '->', path)
        except ValueError as exc:
            if 'already exists' in str(exc):
                print('SKIP existing', identity)
            else:
                raise
    print('GPU worker is stopped; restart with:')
    print('  bash', BASE/'waymo23-native-import/run_gpu_test_service.sh')


if __name__ == '__main__':
    main()
