"""Validate real frozen material weights against the archived source (CPU or CUDA).

Loads the *actual* VanillaGaussians class body without optional renderer imports.
This is a component probe, not a claim of full MultiTrainer/runtime validation.
"""
from __future__ import annotations

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

import ast
import hashlib
import importlib.util
import json
import logging
import math
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[2]
REPO = BASE/'integrations/sr-origin-main'
LEGACY = Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/models'))


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pbr = import_file('sr_legacy_pbr_probe', REPO/'models/legacy_pbr.py')


def compile_nodes(path, nodes, namespace):
    prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix]+nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def vanilla_class(spherical_harmonics):
    path = REPO/'models/gaussians/vanilla.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VanillaGaussians')
    ns = dict(torch=torch, nn=nn, Parameter=nn.Parameter, Tensor=torch.Tensor,
              math=math, F=F, logger=logging.getLogger(__name__),
              num_sh_bases=lambda d: (d+1)**2, spherical_harmonics=spherical_harmonics,
              init_material=pbr.init_material, load_material=pbr.load_material,
              compute_pbr_color=pbr.compute_pbr_color, MATERIAL_SHAPES=pbr.MATERIAL_SHAPES)
    return compile_nodes(path, [node], ns)['VanillaGaussians']


def cpu_sh(degree, directions, coeffs):
    # Independent pre-existing legacy evaluator; new formula calls gsplat in production.
    import sys
    if str(BASE/'waymo23-native-import') not in sys.path:
        sys.path.insert(0, str(BASE/'waymo23-native-import'))
    from scene_math import sh_eval
    if coeffs.shape[-2] < (degree+1)**2:
        raise ValueError('Insufficient SH coefficients')
    return sh_eval(coeffs[..., :(degree+1)**2, :], directions)


def archived_formula(spherical_harmonics):
    path = LEGACY/'gaussians/vanilla.py'
    tree = ast.parse(path.read_text())
    parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VanillaGaussians')
    node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_pbr_color')
    return compile_nodes(path, [node], dict(torch=torch, F=F, spherical_harmonics=spherical_harmonics))['compute_pbr_color']


def archived_lighting():
    path = LEGACY/'modules.py'
    tree = ast.parse(path.read_text())
    names = {'XYZ_Encoder','SinusoidalEncoder','MLP','GlobalLightingModule'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    return compile_nodes(path, nodes, dict(torch=torch, nn=nn, Tensor=torch.Tensor))['GlobalLightingModule']


def background_from_state(state, ctrl, device='cpu', sh=cpu_sh):
    cls = vanilla_class(sh)
    model = cls(class_name='Background', ctrl=ctrl, device=torch.device(device))
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    model.eval().requires_grad_(False)
    return model


class Config(dict):
    def __getattr__(self, key):
        return self[key]


def validate(source_report, device='cpu', samples=2048):
    import numpy as np
    import yaml
    source = _hugsim_expand(json.loads(Path(source_report).read_text()))
    cfg = yaml.safe_load((Path(source['inputs'])/'config.yaml').read_text())
    checkpoint = torch.load(source['checkpoint'], map_location=device, mmap=True, weights_only=False)
    states = checkpoint['models']
    ctrl = Config(cfg['trainer']['gaussian_ctrl_general_cfg'] | cfg['model']['Background']['ctrl'])
    sh = cpu_sh
    if str(device).startswith('cuda'):
        from gsplat import spherical_harmonics
        sh = spherical_harmonics
    background = background_from_state(states['Background'], ctrl, device, sh)
    light_state = states['GlobalLighting']
    count = light_state['appearance_embedding.weight'].shape[0]
    params = cfg['model']['GlobalLighting']['params']
    light = pbr.GlobalLightingModule(class_name='GlobalLighting', n=count, device=torch.device(device), **params).to(device)
    result = light.load_state_dict(light_state, strict=True)
    light.eval().requires_grad_(False)
    golden_light = archived_lighting()(class_name='GlobalLighting', n=count, device=torch.device(device), **params).to(device)
    golden_light.load_state_dict(light_state, strict=True)
    golden_light.eval().requires_grad_(False)
    # Verify every background parameter, including unused metallic/sharpness.
    loaded = background.state_dict()
    if set(loaded) != set(states['Background']):
        raise RuntimeError('Background parameter keys were dropped or added')
    for key, value in loaded.items():
        if not torch.equal(value, states['Background'][key]):
            raise RuntimeError('Checkpoint value changed: '+key)
    index = torch.linspace(0, background.num_points-1, min(samples, background.num_points), device=device).long()
    subset = SimpleNamespace(**{key: value[index] for key,value in loaded.items()},
                             num_points=len(index), num_asg_lobes=background.num_asg_lobes)
    formula = archived_formula(sh)
    frames = source['frames']
    cameras = cfg['data']['pixel_source']['cameras']
    cases = []
    start = source['start_timestep']
    first = np.loadtxt(Path(source['inputs'])/'ego_pose'/f'{start:03d}.txt')
    cv = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1.]])
    with torch.inference_mode():
        for frame in (0, frames//2, frames-1):
            ego = np.linalg.inv(first) @ np.loadtxt(Path(source['inputs'])/'ego_pose'/f'{start+frame:03d}.txt')
            for ordinal, camera in enumerate(cameras):
                image_id = frame*len(cameras)+ordinal
                coeff = pbr.lighting_for_frame(light, frame, image_id, count, device)
                time = torch.tensor([[frame/max(1,count-1)]], device=device)
                golden_coeff = golden_light(time,torch.tensor([image_id],device=device)).squeeze(0)
                c2w = torch.tensor(ego @ np.loadtxt(Path(source['inputs'])/'extrinsics'/f'{camera}.txt') @ cv,
                                   dtype=torch.float32,device=device)
                cam = SimpleNamespace(camtoworlds=c2w)
                actual = pbr.compute_pbr_color(subset,cam,coeff,sh)
                expected = formula(subset,cam,golden_coeff)
                light_error = float((coeff-golden_coeff).abs().max())
                color_error = float((actual-expected).abs().max())
                if not torch.isfinite(actual).all() or not torch.allclose(actual,expected,atol=1e-6,rtol=1e-6) or not torch.allclose(coeff,golden_coeff,atol=1e-6,rtol=1e-6):
                    raise RuntimeError(f'Legacy appearance mismatch: frame={frame} camera={camera}')
                cases.append(dict(frame=frame,camera=camera,image_id=image_id,lighting_max_abs_error=light_error,
                                  pbr_rgb_max_abs_error=color_error))
    return dict(scene=source['scene'], device=str(device), checkpoint=source['checkpoint'],
        background_points=background.num_points, background_keys=list(loaded),
        all_background_values_equal=True, strict_background_load=True, strict_lighting_load=True,
        lighting_images=count, color_samples_per_view=len(index), cases=cases,
        training_started=False, checkpoint_written=False, full_trainer_imported=False,
        full_renderer_verified=False, integration_complete=False,
        source_hashes={str(path.relative_to(REPO)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (REPO/'models/legacy_pbr.py',REPO/'models/gaussians/vanilla.py',REPO/'models/trainers/base.py')})


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args=parser.parse_args()
    report=validate(args.source_report,args.device)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
