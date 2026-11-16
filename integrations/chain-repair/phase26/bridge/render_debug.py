"""Fixed-state decomposed render diagnosis (GPU terminal; diagnostic only).

State: arm_B step 10 (closed-loop ego pose + inserted asset anchor pose).
Renders: baseline (no asset) / composite (with asset) / asset-pixels isolated
via the semantic channel (RigidNodes class 13) / reference render at the
asset's native observed pose (frame 140). Saves raw PNG + alpha/depth/semantic
arrays + exact transforms. Never runs closed loop, never evaluates policy.
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
import sys
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(_hugsim_expand('${HUGSIM_ROOT}'))
sys.path[:0] = [str(ROOT/'integrations/sr-hugsim'), str(ROOT/'waymo23-native-import')]

from scene_api import SceneSession, SceneSpec  # noqa: E402

OUT = HERE/'render_debug'
OUT.mkdir(exist_ok=False)

ARM_B = HERE/'closed_arms/arm_B'
report = _hugsim_expand(json.loads((ARM_B/'run_report.json').read_text()))
step10 = report['observations'][10]
ego_pose = np.asarray(step10['closed_ego_to_world'])
timestamp = float(step10['timestamp'])
insert = _hugsim_expand(json.loads((HERE/'requests/closed552_insert_rigid25.json').read_text()))

manifest = _hugsim_expand(json.loads((HERE/'requests/scene_manifest_552_full.json').read_text()))
spec = SceneSpec.read(HERE/'requests/scene_manifest_552_full.json')

transforms = dict(step='arm_B/10', ego_to_world=ego_pose.tolist(), timestamp_s=timestamp,
                  asset_to_world=insert['asset_to_world'],
                  note='diagnostic fixed state; uncertified')
(OUT/'transforms.json').write_text(json.dumps(transforms, indent=2))

def save(tag, packet):
    from PIL import Image
    for name, view in packet['cameras'].items():
        Image.fromarray(view['rgb']).save(OUT/f'{tag}_{name}.png')
    arrays = {}
    for name, view in packet['cameras'].items():
        for k in ('rgb', 'depth'):
            arrays[f'{name}__{k}'] = view[k]
        for k in ('depth_alpha', 'surface_depth', 'surface_valid'):
            if k in view:
                arrays[f'{name}__{k}'] = view[k]
    np.savez_compressed(OUT/f'{tag}.npz', **arrays)

with SceneSession() as session:
    session.load_scene(HERE/'requests/scene_manifest_552_full.json')
    # 1) background only
    base = session.render(ego_pose, timestamp)
    save('1_background', base)
    # 2)+3) composite with asset
    session.edit(insert)
    comp = session.render(ego_pose, timestamp)
    save('3_composite', comp)
    # asset pixel isolation: semantic channel is in the raw obs; the session
    # packet drops semantics, so re-derive the asset mask from rgb/depth change
    m = {}
    for name in spec.camera_order:
        diff = np.any(comp['cameras'][name]['rgb'] != base['cameras'][name]['rgb'], axis=-1)
        dchg = np.abs(comp['cameras'][name]['depth']-base['cameras'][name]['depth']) > 1e-3
        m[name] = diff | dchg
    np.savez_compressed(OUT/'2_asset_mask.npz', **{k: v for k, v in m.items()})
    from PIL import Image
    for name in spec.camera_order:
        rgb = comp['cameras'][name]['rgb'].copy()
        rgb[~m[name]] = 0
        Image.fromarray(rgb).save(OUT/f'2_asset_only_{name}.png')
        alpha = comp['cameras'][name].get('depth_alpha')
        if alpha is not None:
            Image.fromarray((np.clip(alpha, 0, 1)*255).astype(np.uint8)).save(OUT/f'2_alpha_{name}.png')
            Image.fromarray(m[name].astype(np.uint8)*255).save(OUT/f'2_mask_{name}.png')
    # 4) reference: same asset at its native observed pose (frame 140 overlay anchor)
    session.reset()
    audit140 = _hugsim_expand(json.loads((HERE/'requests/sensor_audit552_frame140_overlay.json').read_text()))
    from asset_placement import terrain_aligned_pose
    import asset_calibration
    cal = asset_calibration.load(HERE/'assets/waymo552_rigid_id25_v1/asset_calibration.json',
                                 HERE/'assets/waymo552_rigid_id25_v1')
    pose140 = terrain_aligned_pose(audit140['proposed_anchor_source_xyz'], audit140['heading_source_radians'],
                                   audit140['ground_fit'], cal['support']['bottom_y'])
    ego140 = np.asarray(_hugsim_expand(json.loads((HERE/'requests/transforms140.json').read_text()))['ego_to_world']) if (HERE/'requests/transforms140.json').exists() else None
    if ego140 is None:
        inputs = Path(spec.source['inputs'])
        e0 = np.loadtxt(inputs/'ego_pose/000.txt')
        ego140 = np.linalg.inv(e0) @ np.loadtxt(inputs/'ego_pose/140.txt')
    session.edit(dict(operation='insert', object_id='preview_car', mode='diagnostic_preview',
                      asset=insert['asset'], asset_calibration=insert['asset_calibration'],
                      asset_to_world=pose140.tolist()))
    ref = session.render(ego140, 14.0)
    save('4_reference_native_pose', ref)
    transforms['reference_pose_frame'] = 140
    transforms['reference_asset_to_world'] = pose140.tolist()
    (OUT/'transforms.json').write_text(json.dumps(transforms, indent=2))

print('render_debug complete ->', OUT)
