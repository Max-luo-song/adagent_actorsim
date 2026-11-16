"""Registered before/after evidence: saved v1 10Hz renders vs new 3D renders."""

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

import json,subprocess
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw
import agent_bootstrap as bootstrap
from ego_trajectory import ffmpeg_path

BASELINES={'scene552_pedestrian_v1':bootstrap.PHASE/'bridge/unified/agent_json_runs/20261105T082358Z/session/actor_trajectories/pedestrian_crossing/packets.json'}


def load_baseline(name,spec,object_id,start,end,camera_frame):
    if name not in BASELINES:raise ValueError('unknown registered pedestrian comparison baseline')
    path=BASELINES[name];value=_hugsim_expand(json.loads(path.read_text()))
    if not value['completed'] or value['object_id']!=object_id or value['camera_source_frame']!=camera_frame:raise ValueError('baseline actor/camera mismatch or incomplete run')
    frames=value['frames']
    if frames[0]['new']['scene']!=spec.source['scene'] or frames[0]['new']['source_frame']!=start or frames[-1]['new']['source_frame']!=end:raise ValueError('baseline scene/interval mismatch')
    for pair in frames:
        for camera in pair['new']['cameras'].values():
            if not Path(camera['rgb']).is_file():raise ValueError('baseline original RGB file unavailable')
    return dict(path=str(path),manifest=value)


def compare_motion(baseline,frames,output,fps,crop):
    original=baseline['manifest']['frames'];timestamps=np.array([p['new']['timestamp_s'] for p in original]);rows=[]
    # Hold the actual earlier 10Hz frames; never synthesize before-state motion.
    for pair in frames:
        packet=pair['new'];index=int(np.searchsorted(timestamps,packet['timestamp_s']+1e-8,side='right')-1);old=original[index]['new']
        if old['source_frame']!=packet['source_frame'] or old['scene']!=packet['scene']:raise ValueError('before/after source clock mismatch')
        for cam,c in packet['cameras'].items():
            a=old['cameras'][cam]
            if not np.allclose(a['K'],c['K'],atol=1e-8,rtol=0) or not np.allclose(a['camera_to_world'],c['camera_to_world'],atol=1e-8,rtol=0):raise ValueError('before/after camera calibration mismatch')
        rows.append(dict(before=old,after=packet))
    out=Path(output);folder=out/'motion_comparison_frames';folder.mkdir()
    for index,row in enumerate(rows):
        image=Image.new('RGB',(1280,544),(20,20,20));draw=ImageDraw.Draw(image)
        draw.text((8,8),'Pedestrian motion conditioning | same camera and source-scene frame',fill='white')
        for column,(key,label) in enumerate([('before','BEFORE: recorded v1, 10Hz (held samples)'),('after',f'AFTER: smooth path / heading / gait, {fps:g}Hz 3D')]):
            draw.text((column*640+8,30),label,fill='white')
            with Image.open(row[key]['cameras']['CAM_FRONT']['rgb']) as source:
                im=source.convert('RGB').crop(crop);im.thumbnail((640,480));image.paste(im,(column*640,54))
        image.save(folder/f'{index:05d}.png')
    video=out/'motion_smoothing_comparison.mp4';ffmpeg=str(ffmpeg_path())
    subprocess.run([ffmpeg,'-v','error','-threads','1','-filter_threads','1','-n','-framerate',str(fps),'-i',str(folder/'%05d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','18',str(video)],check=True)
    subprocess.run([ffmpeg,'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    result=dict(passed=True,video=str(video),baseline_manifest=baseline['path'],frames=len(rows),comparison='before 10Hz actual RGB held; after newly rasterized 3D poses',source_background_fps=10,after_fps=fps,visual_naturalness_certified=False)
    (out/'motion_comparison.json').write_text(json.dumps(result,indent=2))
    return result
