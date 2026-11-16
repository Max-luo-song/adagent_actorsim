"""Present existing paired held-out renders. Does not render, train, or drive."""

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

import argparse,json,hashlib,subprocess
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg

def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(a):
    if sha(a['path'])!=a['sha256']:raise ValueError('Input artifact changed: '+a['path'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('output');args=parser.parse_args()
    run=Path(args.run).resolve();out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    rows=read(run/'render/manifest.json')['rows'];evaluation=run/'structure/evaluation';receipt=read(run/'structure/train/receipt.json')
    grouped={}
    for row in rows:grouped.setdefault(row['source_frame'],{}).setdefault(row['camera_id'],{})[row['kind']]=row
    frames=sorted({r['source_frame'] for r in rows if r['kind']=='novel' and r['split']=='eval'})
    audit=[]
    for frame in frames:
        original=grouped[frame][0]['original'];novel=grouped[frame][0]['novel']
        delta=np.linalg.inv(np.array(original['ego_to_world']))@np.array(novel['ego_to_world'])
        audit.append(dict(source_frame=frame,timestamp_s=novel['timestamp_s'],
            local_forward_left_up_m=delta[:3,3].tolist(),relative_yaw_deg=float(np.degrees(np.arctan2(delta[1,0],delta[0,0])))))
    out.mkdir(parents=True);sheets=out/'frames';sheets.mkdir();inputs=[]
    for i,(frame,state) in enumerate(zip(frames,audit)):
        canvas=Image.new('RGB',(1440,1150),'#161b22');draw=ImageDraw.Draw(canvas)
        draw.text((12,8),'TRAJECTORY COMPARISON | archived held-out renders, NOT closed-loop driving',fill='white')
        draw.text((12,27),f'source frame={frame}; t={state["timestamp_s"]:.1f}s; new pose left shift={state["local_forward_left_up_m"][1]:.3f} m; yaw difference={state["relative_yaw_deg"]:.4f} deg',fill='white')
        draw.text((12,48),'Lateral offset from recorded pose (metres); horizontal axis: source time 0..6s',fill='white')
        draw.line((40,170,1390,170),fill='#78baff',width=3)
        coords=[(40+r['timestamp_s']/6*1350,170-r['local_forward_left_up_m'][1]*95) for r in audit[:i+1]]
        if len(coords)>1:draw.line(coords,fill='#ffbc65',width=4)
        draw.text((8,166),'0m',fill='#78baff');draw.text((8,72),'1m',fill='#ffbc65')
        for col,title in enumerate(['A: RECORDED TRAJECTORY / original model','B: NEW TRAJECTORY / original model','C: NEW TRAJECTORY / distilled model']):
            draw.text((col*480+8,192),title,fill=['#78baff','#ffbc65','#8ce5a2'][col])
        for rowindex,camera in enumerate([1,0,2]):
            pair=grouped[frame][camera];a,b=pair['original'],pair['novel']
            if a['checkpoint_sha256']!=b['checkpoint_sha256']:raise ValueError('A/B source model differs')
            for record in [a,b]:verify(record['before']);inputs.append(record['before'])
            after=evaluation/(b['id']+'_after.png')
            if not after.exists() or b['id'] in receipt['trained_sample_ids']:raise ValueError('Missing held-out after image or train/eval leakage')
            inputs.append(dict(path=str(after),sha256=sha(after)))
            for col,path in enumerate([a['before']['path'],b['before']['path'],after]):
                y=220+rowindex*300;draw.text((col*480+8,y),f'camera {camera}',fill='white')
                with Image.open(path) as im:
                    im=im.convert('RGB');im.thumbnail((480,275));canvas.paste(im,(col*480,y+18))
        draw.text((12,1125),'A vs B shows viewpoint change. B vs C shows distillation only. Images are sampled every 0.5s.',fill='white')
        canvas.save(sheets/f'{i:04d}.png')
        if i==len(frames)-1:canvas.save(out/'trajectory_comparison_last.png')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();video=out/'recorded_vs_novel_before_after.mp4'
    subprocess.run([ffmpeg,'-v','error','-n','-framerate','2','-i',str(sheets/'%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True)
    subprocess.run([ffmpeg,'-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
    result=dict(schema='trajectory_distillation_presentation_v1',source_manifest=dict(path=str(run/'render/manifest.json'),sha256=sha(run/'render/manifest.json')),
        frames=audit,frame_count=len(frames),video=str(video),preview=str(out/'trajectory_comparison_last.png'),full_decode_passed=True,
        reused_existing_renders=True,new_GPU_render=False,new_training=False,closed_loop_driving=False,
        trajectory=dict(maximum_observed_left_m=max(r['local_forward_left_up_m'][1] for r in audit),
            maximum_observed_relative_yaw_deg=max(abs(r['relative_yaw_deg']) for r in audit)),inputs=inputs)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','frames']},indent=2))

if __name__=='__main__':main()
