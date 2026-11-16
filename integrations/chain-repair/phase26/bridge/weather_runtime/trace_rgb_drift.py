"""Trace plain frame80 rendering across fresh processes; no weather or edits.

GPU execution is explicit. Compare reports to separate camera, scene geometry,
appearance and raster differences. Existing GPU evidence remains untouched.
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

import argparse
import hashlib
import importlib.metadata
import json
import os
pass  # Path is provided by the portable boundary above
import random
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


def digest_array(value):
    a = np.ascontiguousarray(value)
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def image_difference(reference, current):
    a, b = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if a.shape != b.shape:
        raise ValueError('Image dimensions differ')
    d = np.abs(a - b)
    # First-order diagnostic, not an image registration or a causal proof.
    gray = a.mean(-1)
    gy, gx = np.gradient(gray)
    x = np.stack([gx[2:-2, 2:-2].ravel(), gy[2:-2, 2:-2].ravel()], -1)
    y = (b - a).mean(-1)[2:-2, 2:-2].ravel()
    shift = np.linalg.lstsq(x, y, rcond=None)[0]
    energy = float(np.mean(y * y))
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                changed_pixel_fraction=float(np.any(d > 0, -1).mean()),
                first_order_displacement_xy_px=shift.tolist(),
                residual_energy_ratio=float(np.mean((y - x @ shift) ** 2) / energy)
                if energy else 0.,
                displacement_is_diagnostic_only=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compare', type=Path)
    p.add_argument('--disable-tf32', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, formal_evaluation_admitted=False,
                  diagnostic_uncertified=True, seed=args.seed, cameras={},
                  environment={k: os.environ.get(k) for k in (
                      'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG',
                      'PYTHONPATH', 'CUDA_HOME', 'PHASE26_PYTHON')})
    backend = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')
        sys.path[:0] = [str(ROOT / 'integrations/sr-hugsim'),
                       str(BRIDGE / 'framework_backend')]
        from scene_api import SceneSpec
        from backend import FrameworkSceneBackend
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if args.disable_tf32:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        report['runtime'] = dict(python=sys.executable, torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            matmul_precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark)
        report['packages'] = {}
        for package in ('gsplat', 'numpy', 'Pillow', 'torch', 'opencv-python'):
            try:
                report['packages'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report['packages'][package] = None
        spec = SceneSpec.read(JOB / 'chain_repair/phase4/dynamic552_frame80/scene_manifest.json')
        inp = Path(spec.source['inputs'])
        start = spec.source['start_timestep']
        pose = np.linalg.inv(np.loadtxt(inp / 'ego_pose' / f'{start:03d}.txt')) @ np.loadtxt(
            inp / 'ego_pose' / f'{start + 80:03d}.txt')
        report.update(pose=pose.tolist(), timestamp_s=8., checkpoint=str(spec.source['checkpoint']))
        backend = FrameworkSceneBackend(spec)
        captures = []
        original = backend.trainer.render_gaussians

        def traced(gs, cam, **kwargs):
            arrays = {key: getattr(gs, key).detach().cpu().numpy()
                      for key in ('means', 'scales', 'quats', 'rgbs', 'opacities')}
            c2w = cam.camtoworlds.detach().cpu().numpy()
            k = cam.Ks.detach().cpu().numpy()
            view = torch.linalg.inv(cam.camtoworlds).detach().cpu().numpy()
            index = len(captures)
            np.savez(args.output / f'camera_{index}.npz', c2w=c2w, K=k, view=view)
            captures.append(dict(actual_c2w=c2w.tolist(), actual_K=k.tolist(),
                                 actual_view=view.tolist(),
                                 gaussians={key: digest_array(a) for key, a in arrays.items()}))
            return original(gs=gs, cam=cam, **kwargs)

        backend.trainer.render_gaussians = traced
        raw = backend.render(pose, 8.)
        if len(captures) != 3:
            raise RuntimeError(f'Expected three rasters, got {len(captures)}')
        historical = BRIDGE / 'unified/weather_window_runs/20261101T112744Z_rainy_3x3/inputs'
        old_report = _hugsim_expand(json.loads((args.compare / 'report.json').read_text())) if args.compare else None
        for i, cam in enumerate(CAMERAS):
            rgb = raw['cameras'][cam]['rgb']
            Image.fromarray(rgb).save(args.output / f'{cam}.png')
            entry = captures[i]
            entry['rgb'] = digest_array(rgb)
            with Image.open(historical / f'080_{i}_framework.png') as im:
                entry['historical_difference'] = image_difference(np.array(im), rgb)
            if args.compare:
                with Image.open(args.compare / f'{cam}.png') as im:
                    entry['fresh_process_difference'] = image_difference(np.array(im), rgb)
                old = old_report['cameras'][cam]
                entry['trace_comparison'] = dict(
                    camera_max_abs={key: float(np.max(np.abs(np.asarray(entry[key]) - old[key])))
                                    for key in ('actual_c2w', 'actual_K', 'actual_view')},
                    gaussian_hash_equal={key: value == old['gaussians'][key]
                                         for key, value in entry['gaussians'].items()})
            report['cameras'][cam] = entry
        source_files = [Path(__file__), BRIDGE / 'framework_backend/backend.py']
        source_files += list((BRIDGE.parent / 'framework_repo/models').rglob('*.py'))
        source_files += list((BRIDGE.parent / 'framework_repo/datasets').rglob('*.py'))
        source_files.append(Path(spec.source['checkpoint']).parent / 'config.yaml')
        report['source_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in source_files}
        report['complete'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if backend is not None:
            backend.close()
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
