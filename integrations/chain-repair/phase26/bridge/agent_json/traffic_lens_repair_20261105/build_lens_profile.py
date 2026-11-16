
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

pass  # Path is provided by the portable boundary above
import json,numpy as np,hashlib
p=Path('outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26');a=p/'bridge/agent_json';d=a/'traffic_preparation_20261105';cal=_hugsim_expand(json.loads((d/'lamp_multiframe_calibration.json').read_text()))
up=np.asarray(cal['gravity_up_source']);green=np.array(cal['lights']['green']['center']);spacing=cal['lights']['yellow']['offset_up_m'];root=Path('outputs/waymo552_three_setup_20261018_043131/inputs');first=np.loadtxt(root/'ego_pose/000.txt')
cv=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
positions=[]
for f in cal['green_training_frames']:
 c=np.linalg.inv(first)@np.loadtxt(root/'ego_pose'/f'{f:03d}.txt')@np.loadtxt(root/'extrinsics/0.txt')@cv;positions.append(c[:3,3])
normal=np.mean(positions,axis=0)-green;normal-=up*np.dot(normal,up);normal/=np.linalg.norm(normal)
checkpoint=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth'));st=checkpoint.stat()
profile=dict(schema='calibrated_signal_lenses_v1',profile_id='scene552_front_v1',scene=552,checkpoint=str(checkpoint),checkpoint_size=st.st_size,checkpoint_mtime_ns=st.st_mtime_ns,
 green_center_source=green.tolist(),up_source=up.tolist(),normal_towards_viewer_source=normal.tolist(),lens_spacing_m=spacing,lens_radius_m=.145,
 surface_front_bias_m=.12,gaussian_spacing_m=.035,gaussian_sigma_m=.021,gaussian_opacity=.995,
 calibration_source='Multi-frame original camera0 green observations plus observed yellow; red slot inferred by equal spacing and checked against dark red lens.',
 green_fit_frames=cal['green_training_frames'],green_holdout_frames=cal['green_holdout_frames'],green_holdout_errors_px={str(f):cal['green_reprojection_error_px'][str(f)] for f in cal['green_holdout_frames']},
 yellow_offset_errors_px=cal['lights']['yellow']['reprojection_error_px'],red_slot_directly_observed_lit=False,
 source_sha256={str((root/'images'/f'{f:03d}_0.jpg').resolve()):hashlib.sha256((root/'images'/f'{f:03d}_0.jpg').read_bytes()).hexdigest() for f in [100,110,112,120,130,140,160,180]},
 visual_gpu_verified=False,semantic_control_connected=False)
profiles=p/'bridge/framework_backend/signal_profiles';profiles.mkdir(exist_ok=True);(profiles/'scene552_front_v1.json').write_text(json.dumps(profile,indent=2))
print('profile saved; spacing',spacing,'holdout',profile['green_holdout_errors_px'])
