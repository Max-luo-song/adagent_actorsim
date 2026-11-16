"""CPU presentation from actual clear/weather frames; never synthesizes motion."""

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

import argparse
import hashlib
import json
pass  # Path is provided by the portable boundary above
import subprocess

from PIL import Image, ImageDraw, ImageFont

CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slowdown', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.slowdown <= 10:
        parser.error('--slowdown must be in 1..10')
    run = args.run.resolve()
    report = _hugsim_expand(json.loads((run / 'closed_loop/run_report.json').read_text()))
    weather_path = run / 'weather/report.json'
    records = _hugsim_expand(json.loads(weather_path.read_text()))['records']
    # Find the repository by its installed encoder rather than a hard-coded depth.
    root = next(p for p in Path(__file__).resolve().parents if
                (p / 'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2').is_file())
    ffmpeg = root / 'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    font_path = root / 'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
    font = ImageFont.truetype(str(font_path), 20)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frames = out / 'frames'
    frames.mkdir()
    source_hashes = {}

    def read(path):
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            return image.convert('RGB').resize((640, 360))

    count = 0
    for i, record in enumerate(records):
        sheet = Image.new('RGB', (1920, 840), '#10141b')
        draw = ImageDraw.Draw(sheet)
        elapsed = record['timestamp'] - records[0]['timestamp']
        observations = report['observations']
        speed = observations[i]['closed_speed_mps'] if i < len(observations) else float('nan')
        draw.text((16,8), f'Scene552 | native bus edit + animated actors + rainy RGB + UniAD/iLQR feedback | t={elapsed:.1f}s | {speed:.2f} m/s', font=font, fill='white')
        draw.text((16,36), f'Status: {report["status"]} | controls {len(report["trace"])}/{report["steps_requested"]} | playback {args.slowdown}x slower | diagnostic, driving safety unverified',font=font,fill='#ffd080')
        folder = run / 'weather' / f'step_{i:03d}'
        for row, kind in enumerate(('clear', 'weather')):
            for col, cam in enumerate(CAMERAS):
                x, y = col * 640, 72 + row * 384
                draw.text((x+10,y), cam + (' | CLEAR render' if kind=='clear' else ' | RAINY / model observation'),font=font,fill='#66ddff')
                sheet.paste(read(folder / f'{cam}_{kind}.png'), (x,y+24))
        for repeat in range(args.slowdown):
            sheet.save(frames / f'{count:05d}.png')
            count += 1
        if i in (0, len(records)-1):
            sheet.save(out / ('first_frame.png' if i == 0 else 'last_frame.png'))
    probe = run / 'native_edit_probe'
    if probe.is_dir():
        sheet = Image.new('RGB', (1920, 408), '#10141b')
        draw = ImageDraw.Draw(sheet)
        for col, label in enumerate(('before', 'after')):
            with Image.open(probe / f'CAM_FRONT_{label}.png') as im:
                view = im.convert('RGB').resize((960, 360))
            sheet.paste(view,(col*960,48))
            draw.text((col*960+16,12), 'Native bus | ' + label + (' source-world Y +0.5m' if label=='after' else ' edit'),font=font,fill='white')
        sheet.save(out / 'edit_before_after.png')
    video = out / 'scene_weather_three_views.mp4'
    subprocess.run([str(ffmpeg), '-v','error','-n','-framerate','10','-i',str(frames/'%05d.png'),
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    subprocess.run([str(ffmpeg),'-v','error','-i',str(video),'-f','null','-'],check=True)
    manifest = dict(source_run=str(run), rendered_again=False, frames=count, playback_fps=10,
        simulation_status=report['status'], controls_executed=len(report['trace']),
        controls_requested=report['steps_requested'],
        slowdown=args.slowdown, actual_simulation_seconds=records[-1]['timestamp']-records[0]['timestamp'],
        playback_seconds=count/10, video_full_decode_passed=True,
        source_report_sha256=hashlib.sha256((run/'closed_loop/run_report.json').read_bytes()).hexdigest(),
        source_weather_report_sha256=hashlib.sha256(weather_path.read_bytes()).hexdigest(),
        source_image_sha256=source_hashes, formal_evaluation_admitted=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(str(video))


if __name__ == '__main__':
    main()
