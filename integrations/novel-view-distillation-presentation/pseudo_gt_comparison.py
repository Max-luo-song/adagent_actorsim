"""Repackage existing held-out images without changing frozen training inputs."""

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

import imageio_ffmpeg
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(run):
    run = Path(run).resolve()
    stage = run / 'pseudo_gt_presentation'
    stage.mkdir()
    (stage / 'frames').mkdir()
    manifest = run / 'pseudo_gt/manifest.json'
    data = _hugsim_expand(json.loads(manifest.read_text()))
    assert data['completed']
    rows = [r for r in data['rows'] if r['kind'] == 'novel' and r['split'] == 'eval']
    by = {}
    for r in rows:
        key = (r['source_frame'], r['camera_id'])
        assert key not in by, key
        assert r['scene_variant'] == 'position_swapped' and r['edit_revision'] == 1
        by[key] = r
    frames = sorted({f for f, _ in by})
    assert frames
    sources = []
    for index, frame in enumerate(frames):
        sheet = Image.new('RGB', (1440, 1050), '#151b22')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f'POSITION SWAP | frame={frame} t={frame/10:.1f}s | held-out views; same edited state in all columns', fill='white')
        for col, title in enumerate(['PSEUDO GT / teacher output', 'BEFORE distillation', 'AFTER distillation']):
            draw.text((col*480+8, 35), title, fill=['#8dc4ff', '#ffb968', '#9fe5a3'][col])
        for row, cam in enumerate([1, 0, 2]):
            r = by[frame, cam]
            paths = []
            for field in ['target', 'before']:
                item = r[field]
                assert digest(item['path']) == item['sha256'], item['path']
                paths.append(Path(item['path']))
            paths.append(run / 'structure/evaluation' / f'{r["id"]}_after.png')
            sizes = []
            for col, path in enumerate(paths):
                with Image.open(path) as image:
                    sizes.append(image.size)
                    image = image.convert('RGB')
                    image.thumbnail((480, 290))
                    sheet.paste(image, (col*480, 90+row*315))
                draw.text((col*480+8, 70+row*315), f'camera {cam}', fill='white')
                sources.append(dict(frame=frame, camera_id=cam, column=col, path=str(path), sha256=digest(path)))
            assert len(set(sizes)) == 1, (frame, cam, sizes)
        draw.text((10, 1025), 'Pseudo GT is generated supervision, not real ground truth. Full images shown; training/metrics use valid static masks.', fill='white')
        sheet.save(stage / 'frames' / f'{index:04d}.png')
        if index == len(frames)-1:
            sheet.save(stage / 'comparison_last.png')
    video = stage / 'pseudo_gt_before_after.mp4'
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, '-v', 'error', '-n', '-framerate', '2', '-i', str(stage/'frames/%04d.png'), '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', str(video)], check=True)
    subprocess.run([ff, '-v', 'error', '-threads', '1', '-i', str(video), '-f', 'null', '-'], check=True)
    report = dict(completed=True, frames=len(frames), fps=2, full_decode_passed=True,
                  comparison='same edited new view: pseudo GT / before / after',
                  video=dict(path=str(video), sha256=digest(video)),
                  pseudo_gt_manifest_sha256=digest(manifest), sources=sources,
                  training_rerun=False, gpu_render_executed=False, closed_loop_driving=False)
    (stage/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    build(parser.parse_args().run)
