
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

import hashlib,json
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parent
RUN=OUT.parent;J=RUN/'joint'
def read(n):return _hugsim_expand(json.loads((J/n).read_text()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
originals={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and OUT not in p.parents}
accept=read('acceptance.json');probe=read('native_rotation_probe/report.json')
report=read('closed_loop/run_report.json');weather=read('weather/report.json')['records']
packets=read('weather_packet_audit.json');receipts=read('weather_input_receipts.json')
checks={}
checks['original_acceptance_all_20']=accept['passed'] and len(accept['checks'])==20 and all(accept['checks'].values())
checks['command_rigid1_yaw15']=accept['unified_edit_command']['object_id']=='native:RigidNodes:1' and accept['unified_edit_command']['yaw_degrees']==15
checks['rotation_geometry_and_proxy']=all(probe[k] for k in ('rotation_matrices_exact','all_root_positions_unchanged','other_instance_rotations_unchanged','proxy_tensor_shared','unit_quaternions','local_gaussians_unchanged','oriented_proxy_matches_rotation','other_groups_unchanged'))
counts={}
for cam,c in probe['cameras'].items():
 a=rgb(J/f'native_rotation_probe/{cam}_before.png');b=rgb(J/f'native_rotation_probe/{cam}_after.png')
 ma=np.asarray(Image.open(J/f'native_rotation_probe/{cam}_before_mask.png'));mb=np.asarray(Image.open(J/f'native_rotation_probe/{cam}_after_mask.png'))
 counts[cam]=dict(rgb_changed_pixels=int(np.any(a!=b,axis=-1).sum()),mask_changed_pixels=int(np.count_nonzero(ma!=mb)))
 checks[f'{cam}_saved_probe_counts_match']=all(counts[cam][k]==c[k] for k in counts[cam])
checks['five_controls_six_observations']=len(report['trace'])==5 and len(report['observations'])==len(weather)==len(packets)==6
checks['one_edit_revision']=all(p['edit_revision']==1 and p['edited'] for p in packets)
checks['one_rotate_journal']=len(read('unified_edit_journal.json'))==1 and read('unified_edit_journal.json')[0]['command']==accept['unified_edit_command']
checks['reset_all_eight_fields']=len(read('unified_reset_audit.json'))==1 and len(read('unified_reset_audit.json')[0])==8 and all(read('unified_reset_audit.json')[0].values())
all_pixels=True;all_receipts=True;all_masks=True
for i,r in enumerate(weather):
 for cam,c in r['cameras'].items():
  clear=rgb(J/f'weather/step_{i:03d}/{cam}_clear.png');wet=rgb(J/f'weather/step_{i:03d}/{cam}_weather.png')
  mask=np.asarray(Image.open(J/f'weather/step_{i:03d}/{cam}_mask.png'))>0
  all_pixels &= hashlib.sha256(wet.tobytes()).hexdigest()==c['rgb_sha256']==packets[i]['rgb_sha256'][cam]
  all_masks &= bool(np.array_equal(clear[mask],wet[mask]))
  if i<5:all_receipts &= receipts[f'step_{i:03d}']['rgb_sha256'][cam]==c['rgb_sha256']
checks['18_saved_weather_views_match_packet_hashes']=all_pixels
checks['18_views_dynamic_pixels_preserved']=all_masks
checks['15_uniad_input_receipts_match_weather']=all_receipts and len(receipts)==5
checks['five_worker_replies_ok']=all(read(f'closed_loop/uniad_worker/done_{i:03d}.json')['status']=='ok' for i in range(5))
checks['executed_poses_match_weather']=all(np.allclose(r['ego_to_world'],o['closed_ego_to_world'],atol=1e-6,rtol=0) for r,o in zip(weather,report['observations']))
checks['causal_history']=all(not r['audit']['future_frames_used'] and max(r['audit']['window_timestamps'])<=r['timestamp'] for r in weather)
checks['source_fingerprints_match']=all(Path(p).is_file() and sha(Path(p))==h for p,h in read('implementation.json').items())
checks['no_reference_box_overlap']=all(not o['all_actor_reference']['overlap'] for o in report['observations'])
checks['diagnostic_no_formal_admission']=accept['formal_evaluation_admitted'] is False and accept['safety_equivalence_to_mpc'] is False
# Saved clear-render contact sheet, same model/camera/pose/time; no invented frames.
a=Image.open(J/'native_rotation_probe/CAM_FRONT_before.png').convert('RGB')
b=Image.open(J/'native_rotation_probe/CAM_FRONT_after.png').convert('RGB')
w=960;h=round(a.height*w/a.width)
sheet=Image.new('RGB',(w,2*(h+30)),(20,20,20));draw=ImageDraw.Draw(sheet)
for i,(im,label) in enumerate(((a,'BEFORE: original vehicle heading | scene552 frame80'),(b,'AFTER: RigidNodes:1 world-Z yaw +15 deg | same camera/pose/time'))):
 y=i*(h+30);draw.text((10,y+8),label,fill='white');sheet.paste(im.resize((w,h)),(0,y+30))
sheet.save(OUT/'rigid1_yaw15_before_after.jpg',quality=94)
checks['original_files_unchanged']=all(sha(RUN/p)==h for p,h in originals.items())
summary=dict(passed=all(checks.values()),checks=checks,pixel_counts=counts,sealed_files=len(originals),
 simulation_seconds=.5,control='iLQR tracking passthrough',speed_initial_mps=report['trace'][0]['state_before'][3],
 speed_final_mps=report['trace'][-1]['state_after'][3],min_actor_gap_m=min(o['all_actor_reference']['nearest']['gap_m'] for o in report['observations']),
 formal_evaluation_admitted=False,safety_certified=False)
(OUT/'original_sha256.json').write_text(json.dumps(originals,indent=2))
(OUT/'review.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));assert summary['passed']
