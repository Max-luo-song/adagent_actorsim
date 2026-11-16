"""Optional offline RGB repair, using the locally archived official pipeline.

Preserve original, repaired RGB and change mask; no training, no geometry edits.
Run in a separate, pinned dependency target/environment, never modify UniAD env.
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
import importlib.metadata as md
import json
import sys
pass  # Path is provided by the portable boundary above
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    required={'diffusers':'0.25.1','transformers':'4.38.0','peft':'0.9.0','huggingface-hub':'0.25.1'}
    for package,version in required.items():
        if md.version(package)!=version: raise RuntimeError(f'{package} must be {version}; use isolated target')
    root=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/Difix3D'))
    sys.path.insert(0,str(root/'src'))
    import torch
    from PIL import Image
    from pipeline_difix import DifixPipeline
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True,exist_ok=False)
    before=Image.open(args.input).convert('RGB')
    # Pipeline requires dimensions divisible by 8; retain raw input separately.
    original=before.copy()
    before=before.resize((before.width//8*8,before.height//8*8))
    pipe=DifixPipeline.from_pretrained(str(root/'difix'),local_files_only=True).to('cuda')
    result=pipe('remove degradation',image=before,height=before.height,width=before.width,
                num_inference_steps=1,timesteps=[199],
                guidance_scale=0.0,generator=torch.Generator('cuda').manual_seed(42)).images[0]
    original.save(args.output/'original.png');before.save(args.output/'before.png')
    result.save(args.output/'fixed.png')
    diff=np.abs(np.asarray(result,dtype=float)-np.asarray(before,dtype=float)).max(-1)
    Image.fromarray(((diff>3)*255).astype('uint8')).save(args.output/'changed_rgb_mask.png')
    report=dict(formal_evaluation_admitted=False,diagnostic_uncertified=True,
        evidence_level='single-image GPU call only; not closed-loop admission',
        model='local nvidia/difix',input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        packages=required,depth_modified=False,collision_geometry_modified=False,
        generated_rgb_low_trust=True,mask_semantics='RGB change threshold; not geometric completion mask')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
