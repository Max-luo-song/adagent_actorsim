"""Compare actual old/new B outcomes; preserve causal scope and unequal support."""

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
import numpy as np
from common import read,write,verify,asset,sha

def compare_native(root):
 root=Path(root);plan=read(root/'plan.json');verify(plan['collision_off_baseline_report']);off=Path(plan['collision_off_baseline_root'])/'B';on=root/'B';reports={n:read(p/'run_report.json') for n,p in [('off',off),('on',on)]};rows=[];changes=[]
 for r in reports['on']['trace']:
  i=r['index'];audit=r['native_collision_audit'];verify(audit);a=read(audit['path'])
  if a['enabled']!=(i>5) or a['native_code_modified']:raise ValueError('Native collision rollout policy violated')
  plans=on/f'uniad_worker/step_{i:03d}/native_collision_plans.npz'
  if sha(plans)!=a['plans_sha256']:raise ValueError('Native collision before/after archive changed')
  with np.load(plans) as z:pre=z['before'][0].copy();post=z['after'][0].copy()
  np.testing.assert_array_equal(post,np.asarray(r['raw_uniad'],dtype=post.dtype));changes.append(a)
  if i<len(reports['off']['trace']):
   old=reports['off']['trace'][i];same=bool(np.allclose(old['ego_to_world'],r['ego_to_world'],rtol=0,atol=1e-9))
   rows.append(dict(index=i,time_s=r['timestamp_s'],same_actual_ego_pose=same,same_camera_images=old['image_sha256']==r['image_sha256'],native_enabled=a['enabled'],native_method_calls=a['native_method_calls'],native_trajectory_change_m=a['trajectory_max_change_m'],off_speed_mps=old['motion_state'][3],on_speed_mps=r['motion_state'][3],off_acc=old['action']['acc_mps2'],on_acc=r['action']['acc_mps2'],on_pre_collision_end=pre[-1].tolist(),on_native_processed_end=post[-1].tolist(),on_adopted_end=r['adopted'][-1],post_divergence_visual_change_not_attributed_only_to_native_stage=True))
 summary={}
 for n,report in reports.items():
  score=read((off if n=='off' else on)/'native_eval/native_scores.json')
  summary[n]=dict(status=report['status'],termination_reason=report.get('termination_reason'),controls=len(report['trace']),collision=report.get('terminal_collision',{}).get('collision',False),last_observation_time_s=report['observations'][-1]['timestamp'],terminal_speed_mps=report['observations'][-1]['motion_state'][3],native_executed_scores=score.get('scores',{}).get('executed_path'),score_scope='Actual supported future windows only; differing episode duration is not a matched-score certificate')
 out=dict(schema='native_collision_off_on_B_v6',off_report=asset(off/'run_report.json'),on_report=asset(on/'run_report.json'),summaries=summary,enabled_frames=sum(a['enabled'] for a in changes),native_method_called_frames=sum(a['native_method_calls']>0 for a in changes),native_optimizer_required_frames=sum(a['native_filter_requires_solver'] for a in changes),native_reference_changed_frames=sum(a['trajectory_max_change_m']>1e-5 for a in changes),occupancy_unavailable_enabled_frames=sum(a['enabled'] and not a['occupancy_available'] for a in changes),first_changed_frame=next((r['index'] for r in rows if r['native_trajectory_change_m']>1e-5),None),new_full_source_window_completed=reports['on']['status']=='completed' and len(reports['on']['trace'])==plan['controls'],generalization_certified=False,rows=rows)
 write(root/'comparison_Boff_Bon.json',out);media=render_pair(root,off,on,reports);return {k:v for k,v in dict(out,media=media).items() if k!='rows'}

def render_pair(root,off,on,reports):
 from PIL import Image,ImageDraw
 import imageio_ffmpeg,subprocess
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 stage=Path(root)/'presentation_Boff_Bon';stage.mkdir(exist_ok=True)
 if (stage/'artifacts.json').exists():
  cached=read(stage/'artifacts.json')
  for entry in [cached['video'],cached['plot'],*cached['source_reports']]:verify(entry)
  if not cached['full_decode_passed']:raise ValueError('Cached pair video not decoded')
  return cached
 frames=stage/'frames';frames.mkdir(exist_ok=True);arms={'off':off,'on':on};count=max(len(r['observations']) for r in reports.values())
 for i in range(count):
  sheet=Image.new('RGB',(1440,680),'#151515');draw=ImageDraw.Draw(sheet);draw.text((8,5),f'Local time {i*.1:.1f}s | same scripted target | actual closed-loop observations',fill='white')
  for row,(name,path) in enumerate(arms.items()):
   draw.text((8,30+row*320),'B '+('v9 per-candidate risk memory' if name=='on' else 'v8 joint local repair'),fill='white')
   for col,cam in enumerate(['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']):
    file=path/f'steps/{i:03d}'/(cam+'.png')
    if file.exists():
     with Image.open(file) as im:im.thumbnail((480,280));sheet.paste(im,(col*480,55+row*320))
    else:draw.text((col*480+10,150+row*320),'Episode ended; no fabricated frame',fill='white')
  sheet.save(frames/f'{i:04d}.png')
 video=stage/'risk_memory_v8_v9_3CAM.mp4'
 if video.exists():raise FileExistsError('Pair video retained; no overwrite')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate','10','-i',str(frames/'%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True,timeout=180)
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True,timeout=60)
 fig,axes=plt.subplots(2,1,figsize=(12,7))
 for name,r in reports.items():
  obs=r['observations'];xy=np.array([o['pose'] for o in obs]);axes[0].plot(xy[:,0,3],xy[:,1,3],label='B '+name);axes[1].plot([o['timestamp'] for o in obs],[o['motion_state'][3] for o in obs],label='B '+name)
 axes[0].set(xlabel='Source forward x (m)',ylabel='Source left y (m)');axes[0].axis('equal');axes[1].set(xlabel='Local time (s)',ylabel='Actual ego speed (m/s)')
 for ax in axes:ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(stage/'executed_path_and_speed.png',dpi=160);plt.close(fig);result=dict(video=asset(video),plot=asset(stage/'executed_path_and_speed.png'),frames=count,full_decode_passed=True,source_reports=[asset(p/'run_report.json') for p in arms.values()]);write(stage/'artifacts.json',result);return result
