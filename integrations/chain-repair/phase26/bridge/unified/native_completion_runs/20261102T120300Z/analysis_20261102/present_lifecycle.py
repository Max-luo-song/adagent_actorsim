"""Present saved lifecycle frames and explicitly labelled log-only resets."""

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
import json
pass  # Path is provided by the portable boundary above
import subprocess

from PIL import Image, ImageDraw, ImageFont

RUN = Path(__file__).resolve().parent.parent
SOURCE = RUN / 'lifecycle'
OUT = SOURCE / 'presentation_20261102'
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code/HUGSIM').is_dir())
FONT = ROOT / 'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
FFMPEG = ROOT / 'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')


def main():
    report = _hugsim_expand(json.loads((SOURCE / 'probe/report.json').read_text()))
    assert report['passed'] and report['checks']['same_backend_two_episodes']
    OUT.mkdir(exist_ok=False)
    frames = OUT / 'frames'
    frames.mkdir()
    font = ImageFont.truetype(str(FONT), 20)
    large = ImageFont.truetype(str(FONT), 34)
    count = 0
    hashes = {}
    segments = []

    def emit(sheet, seconds, label):
        nonlocal count
        # Input is a storyboard of measured still frames, NOT synthesized motion.
        for _ in range(seconds * 2):
            sheet.save(frames / f'{count:04d}.png')
            count += 1
        segments.append(dict(label=label, playback_seconds=seconds))

    def card(title, lines):
        sheet = Image.new('RGB', (1920, 840), '#10141b')
        draw = ImageDraw.Draw(sheet)
        draw.text((60,110), title, font=large, fill='white')
        for i,line in enumerate(lines):
            draw.text((60,220+i*52),line,font=font,fill='#88ddff')
        return sheet

    def observation(load, index, title, subtitle):
        sheet = Image.new('RGB', (1920, 840), '#10141b')
        draw = ImageDraw.Draw(sheet)
        draw.text((16,8),title,font=font,fill='white')
        draw.text((16,36),subtitle+' | saved stills held for viewing; no driving in this test',font=font,fill='#ffd080')
        folder = SOURCE / f'weather_load_{load}/step_{index:03d}'
        for row,kind in enumerate(('clear','weather')):
            for col,cam in enumerate(CAMERAS):
                path = folder / f'{cam}_{kind}.png'
                hashes[str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as image:
                    view=image.convert('RGB').resize((640,360))
                x,y=col*640,72+row*384
                draw.text((x+10,y),cam+' | '+kind.upper(),font=font,fill='#88ddff')
                sheet.paste(view,(x,y+24))
        return sheet

    emit(card('Same-session two-episode edit / reset - actual GPU evidence',[
        'Same UnifiedSceneSession + same framework backend + same weather worker for episodes 1 and 2.',
        'Episode 1: native bus RigidNodes:1 translated source-world Y +0.5m.',
        'Reset, then episode 2: DeformableNodes:1 translated source-world Y +0.5m.',
        'Each episode has only 2 measured observations, t=8.0s and 8.1s, with fixed ego pose.',
        'Reset checks are log evidence; reset-only RGB frames were not separately saved.']),3,'introduction')
    fronts=[]
    for ep in (0,1):
        obj=report['episodes'][ep]['command']['object_id']
        for j in (0,1):
            index=ep*2+j
            sheet=observation(0,index,f'Episode {ep+1}: {obj} | source-world Y +0.5m',
                f't={8.+j*.1:.1f}s | weather epoch {report["episodes"][ep]["weather_audits"][j]["epoch"]}')
            emit(sheet,2,f'episode{ep+1}_observation{j}')
            if j==0:
                sheet.save(OUT / f'episode{ep+1}.png')
                with Image.open(SOURCE/f'weather_load_0/step_{index:03d}/CAM_FRONT_clear.png') as im:
                    fronts.append(im.convert('RGB').resize((960,540)))
        emit(card(f'Episode {ep+1}: RESET - runtime checks passed',[
            'Original native trajectories restored; RGB/mask equality checked during execution.',
            'Revision and journal cleared; new scene generation; weather history reset.',
            'Old editor and old-generation commands rejected; failed commands did not commit.',
            'This card describes recorded checks; no separate reset-only RGB frame was saved.']),2,f'episode{ep+1}_reset_log_only')
    emit(card('Reload the same scene552 - runtime checks passed',[
        'Old backend closed; old editor rejected; new empty generation and weather worker.',
        'Next frame is a real saved unedited observation after load_scene.',
        'This verifies same-scene reload; a different scene was not tested.']),2,'same_scene_reload_log')
    emit(observation(1,0,'After reload: scene552, no active edits','t=8.0s | fresh weather history'),2,'reloaded_observation')
    compare = Image.new('RGB',(1920,594),'#10141b')
    draw=ImageDraw.Draw(compare)
    for ep,image in enumerate(fronts):
        compare.paste(image,(ep*960,54))
        label='Episode 1: BUS translated +0.5m' if ep==0 else 'Episode 2: bus restored; Deformable translated +0.5m'
        draw.text((ep*960+12,14),label,font=font,fill='white')
    compare.save(OUT/'two_episode_front_comparison.png')
    video=OUT/'two_episode_edit_reset.mp4'
    subprocess.run([str(FFMPEG),'-v','error','-n','-framerate','2','-i',str(frames/'%04d.png'),
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    subprocess.run([str(FFMPEG),'-v','error','-i',str(video),'-f','null','-'],check=True)
    manifest=dict(source_run=str(RUN),source_report_sha256=hashlib.sha256((SOURCE/'probe/report.json').read_bytes()).hexdigest(),
        source_image_sha256=hashes,rendered_again=False,storyboard=True,measured_episode_frames=[2,2],
        ego_pose_fixed=True,driving_test=False,reset_frames_saved=False,video_full_decode_passed=True,
        playback_fps=2,playback_frames=count,playback_seconds=count/2,segments=segments,
        different_scene_verified=False,formal_evaluation_admitted=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(str(video))


if __name__=='__main__':main()
