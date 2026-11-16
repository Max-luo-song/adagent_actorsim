"""Import production SR classes and strictly load real material weights; no AST shim."""

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
import json
pass  # Path is provided by the portable boundary above
import sys
import time

BASE=Path(__file__).resolve().parents[2]
REPO=BASE/'integrations/sr-origin-main'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    began=time.monotonic()
    sys.path.insert(0,str(REPO))
    from models.modules import GlobalLightingModule
    from models.gaussians.vanilla import VanillaGaussians
    from omegaconf import OmegaConf
    import torch
    print("production material classes imported", flush=True)
    results=[]
    for name in ('waymo172_three_setup_20261016_132050_295391','waymo552_three_setup_20261018_043131'):
        source=_hugsim_expand(json.loads((BASE/'outputs'/name/'source_report.json').read_text()))
        cfg=OmegaConf.load(Path(source['inputs'])/'config.yaml')
        checkpoint=torch.load(source['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        state=checkpoint['models']
        ctrl=OmegaConf.merge(cfg.trainer.gaussian_ctrl_general_cfg,cfg.model.Background.ctrl)
        background=VanillaGaussians(class_name='Background',ctrl=ctrl,device=torch.device('cpu'))
        bg_result=background.load_state_dict(state['Background'],strict=True)
        count=state['GlobalLighting']['appearance_embedding.weight'].shape[0]
        lighting=GlobalLightingModule(class_name='GlobalLighting',n=count,device=torch.device('cpu'),**cfg.model.GlobalLighting.params)
        light_result=lighting.load_state_dict(state['GlobalLighting'],strict=True)
        for label,model,original in (('Background',background,state['Background']),('GlobalLighting',lighting,state['GlobalLighting'])):
            model.eval().requires_grad_(False)
            loaded=model.state_dict()
            if set(loaded)!=set(original) or any(not torch.equal(v,original[k]) for k,v in loaded.items()):
                raise RuntimeError('Production weight mismatch: '+label)
        results.append(dict(scene=source['scene'],strict_background_load=True,strict_lighting_load=True,
                            all_values_equal=True,background_keys=len(background.state_dict()),lighting_images=count))
        print(source['scene'],'production material class loading passed',flush=True)
        del background,lighting,checkpoint,state
    report=dict(status='production_material_load_passed',results=results,wall_seconds=time.monotonic()-began,
                ast_used=False,training_started=False,full_trainer_imported=False,full_renderer_verified=False,
                integration_complete=False)
    Path(args.output).write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
