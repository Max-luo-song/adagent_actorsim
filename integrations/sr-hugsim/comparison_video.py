"""CPU-only comparison presentation, without simulator/model imports."""

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
import subprocess
BASE=Path(__file__).resolve().parents[2]
CAMERA='CAM_FRONT_RIGHT'
THREE_CAMERAS=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')


def compose_three_closed(packet, elapsed, speed, mode, background_note='source time'):
    """Three synchronized closed-loop cameras; labels never cover RGB pixels."""
    from PIL import Image, ImageDraw, ImageFont
    views=[Image.fromarray(packet['cameras'][name]['rgb']) for name in THREE_CAMERAS]
    if any(view.size != views[0].size for view in views):
        raise ValueError('Three-view camera image dimensions differ')
    w,h=views[0].size
    canvas=Image.new('RGB',(3*w,h+80),(16,20,27))
    draw=ImageDraw.Draw(canvas)
    font_path=BASE/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
    try: font=ImageFont.truetype(str(font_path),20)
    except (ImportError,OSError): font=ImageFont.load_default()
    for index,(name,view) in enumerate(zip(THREE_CAMERAS,views)):
        canvas.paste(view,(index*w,80))
        draw.text((index*w+14,6),name+' | CLOSED LOOP',font=font,fill='white')
        draw.text((index*w+14,34),f't={elapsed:.1f}s | speed={speed:.2f} m/s | {mode}',font=font,fill='#66ddff')
        draw.text((index*w+14,56),background_note,font=font,fill='#dddddd')
    return canvas


def compose(left, right, elapsed, open_speed, closed_speed, mode):
    from PIL import Image, ImageDraw, ImageFont
    if left.size != right.size:
        raise ValueError('Comparison image dimensions differ')
    w, h = left.size
    canvas = Image.new('RGB', (2*w, h+80), (16, 20, 27))
    canvas.paste(left, (0, 80)); canvas.paste(right, (w, 80))
    draw = ImageDraw.Draw(canvas)
    font_path = BASE/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
    try:
        font = ImageFont.truetype(str(font_path), 20)
    except (ImportError, OSError):
        font = ImageFont.load_default()
    draw.text((14, 6), 'LEFT: Open-loop EDITED | recorded ego trajectory', font=font, fill='white')
    draw.text((w+14, 6), 'RIGHT: Closed-loop EDITED | UniAD + MPC + vehicle feedback', font=font, fill='white')
    draw.text((14, 34), f'{CAMERA} | t={elapsed:.1f}s | speed={open_speed:.2f} m/s', font=font, fill='#66ddff')
    draw.text((w+14, 34), f't={elapsed:.1f}s | speed={closed_speed:.2f} m/s | {mode}', font=font, fill='#66ddff')
    draw.text((14, 56), 'Same inserted white car / same simulation clock / frozen weights / diagnostic preview', font=font, fill='#dddddd')
    return canvas


def encode(output, fps, count, filename='172_edited_open_left_closed_right.mp4'):
    ffmpeg = BASE/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    video = output/filename
    subprocess.run([str(ffmpeg), '-y', '-v', 'error', '-framerate', str(fps), '-i',
        str(output/'comparison_frames/%03d.png'), '-c:v', 'libx264', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    decoded = subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(video), '-progress', 'pipe:1',
        '-f', 'null', '-'], check=True, capture_output=True, text=True)
    frames = [int(line.split('=')[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    if not frames or frames[-1] != count:
        raise RuntimeError('Encoded comparison frame count mismatch')
    from PIL import Image
    chosen = [0, count//2, count-1]
    samples = [Image.open(output/f'comparison_frames/{i:03d}.png').convert('RGB') for i in chosen]
    montage = Image.new('RGB', (samples[0].width, sum(p.height for p in samples)))
    y = 0
    for im in samples: montage.paste(im, (0,y)); y += im.height
    montage.save(output/'first_middle_last.png')
    return dict(path=str(video), frames=frames[-1], fps=fps, duration_s=count/fps,
        width=samples[0].width, height=samples[0].height, full_decode_passed=True)

