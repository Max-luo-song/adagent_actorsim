"""Read-only SR integration inspection. Does not import SR, CUDA, or training code."""

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

import argparse,ast,json,subprocess
pass  # Path is provided by the portable boundary above


def exported_symbol(repo,dotted,seen=None):
    seen=set() if seen is None else seen
    if dotted in seen:return False
    seen.add(dotted);module,_,name=dotted.rpartition('.')
    file=repo/Path(*module.split('.')).with_suffix('.py');package=False
    if not file.is_file():file=repo/Path(*module.split('.'))/'__init__.py';package=True
    if not file.is_file():return False
    tree=ast.parse(file.read_text())
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name:return True
        if isinstance(node,ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name)!=name:continue
                if node.level:
                    base=module.split('.') if package else module.split('.')[:-1]
                    if node.level>1:base=base[:-(node.level-1)]
                    target='.'.join(base+([node.module] if node.module else [])+[alias.name])
                else:target=(node.module+'.' if node.module else '')+alias.name
                return exported_symbol(repo,target,seen)
    return False


def inspect(repo,config=None,dataset="waymo"):
    repo=Path(repo).resolve()
    def tree(relative):return ast.parse((repo/relative).read_text())
    def methods(relative,klass):
        t=tree(relative);c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==klass)
        return [n.name for n in c.body if isinstance(n,ast.FunctionDef)]
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    report=dict(repo=str(repo),commit=revision,role='scene_reconstruction_editing_rendering_backend',
                driving_policy_model=False,training_started=False,sr_python_imported=False,integration_complete=False,
                online_entry='sim_render/cam/cam_inference_online.py:Renderer.render_single_frame',
                online_output='JPEG file path per camera; not a lossless RGB/depth/calibration packet',
                waymo_online_loader_implemented='load_specified_cameras' in methods('datasets/waymo/waymo_sourceloader.py','WaymoPixelSource'),
                editing_entry='models/trainers/base.py:BasicTrainer.edit_gaussians',
                checkpoint_files=[str(p.relative_to(repo)) for p in repo.rglob('*.pth') if '.git' not in p.parts],
                required_user_inputs=['matching trained scene checkpoint','checkpoint-adjacent config.yaml','processed scene data root','intended dataset and cameras'])
    report['target_dataset']=dataset
    source_classes={'waymo':'WaymoPixelSource','chery':'CheryPixelSource','qcraft':'QcraftPixelSource'}
    report['target_online_loader_implemented']='load_specified_cameras' in methods(f'datasets/{dataset}/{dataset}_sourceloader.py',source_classes[dataset])
    report['blockers']=[]
    if dataset=='waymo' and not report['waymo_online_loader_implemented']:report['blockers'].append('Waymo inherits load_specified_cameras raising NotImplementedError; online interface needs adaptation')
    if config:
        import yaml
        cfg=yaml.safe_load(Path(config).read_text());types=[cfg['trainer']['type']]+[v['type'] for v in cfg.get('model',{}).values() if isinstance(v,dict) and 'type' in v]
        missing=[t for t in types if not exported_symbol(repo,t)]
        report['compatibility_probe']=dict(config=str(Path(config).resolve()),configured_types=types,missing_symbols=missing,actual_checkpoint_load_tested=False)
        pbr=[name for name,value in cfg.get('model',{}).items() if isinstance(value,dict) and value.get('ctrl',{}).get('use_pbr',False)]
        report['compatibility_probe']['legacy_pbr_nodes']=pbr
        if pbr and not any(isinstance(n,ast.FunctionDef) and n.name=='compute_pbr_color' for n in ast.walk(tree('models/gaussians/vanilla.py'))):
            report['blockers'].append('Legacy PBR configuration requires a material compatibility backend; current VanillaGaussians does not implement compute_pbr_color')
        if missing:report['blockers'].append('Probe configuration references missing symbols: '+', '.join(missing))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--probe-config');p.add_argument('--dataset',choices=('waymo','chery','qcraft'),default='waymo');p.add_argument('--output',required=True);a=p.parse_args()
    report=inspect(a.repo,a.probe_config,a.dataset);Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
